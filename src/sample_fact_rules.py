# -*- coding: utf-8 -*-
"""
Deterministic (fully factual) transition-rule sampling with an LLM + hard stock-law pass.

Goal:
- Use LLM to propose deterministic expressions (no probabilities) for next-step updates.
- Then enforce the "inventory conservation law" uniformly:
    Inv_{t+1} = min(Capacity?, max(0, Inv_t + ArrivingOrders - DownstreamDemand))
  for any stock-like property; capacity clamp applied only if a capacity-like self property exists.

Inputs:
  outputs/final_structure.pruned.json    # pruned Self vocabulary per agent
  outputs/final_state_space.json         # must include Self and Relations (ArrivingOrders/DownstreamDemand preferred)

Outputs:
  outputs/final_transition_rules.json
  outputs/transition_rules_debug.json

Requires:
  pip install openai
  env DASHSCOPE_API_KEY or configs/keys.json with {"DASHSCOPE_API_KEY": "..."}
"""

import os, re, json, time
from typing import Dict, Any, List
from openai import OpenAI

# --------- LLM config (DashScope OpenAI-compatible) ----------
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"
SYSTEM = (
    "You are a precise supply chain expert. You produce ONLY valid JSON. "
    "No prose. No code fences. Deterministic arithmetic only."
)

# --------- Paths ----------
HERE = os.path.dirname(__file__)
OUTDIR = os.path.join(HERE, "..", "outputs")
PATH_PRUNED = os.path.join(OUTDIR, "final_structure.pruned.json")
PATH_STATE  = os.path.join(OUTDIR, "final_state_space.json")
PATH_OUT    = os.path.join(OUTDIR, "final_transition_rules.json")
PATH_DBG    = os.path.join(OUTDIR, "transition_rules_debug.json")

# --------- Helpers ----------
def _get_api_key() -> str:
    key = os.getenv("DASHSCOPE_API_KEY")
    if key:
        return key
    try:
        with open(os.path.join(HERE, "..", "configs", "keys.json"), "r", encoding="utf-8") as f:
            j = json.load(f)
            return j.get("DASHSCOPE_API_KEY", "")
    except Exception:
        return ""

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _strip_fences(s: str) -> str:
    s = re.sub(r"```(?:json)?\s*", "", s, flags=re.I)
    s = s.replace("```", "")
    return s.strip()

def _try_parse_json(text: str) -> Any:
    txt = _strip_fences(text)
    m = re.search(r"\{.*\}\s*$", txt, flags=re.S)
    if m:
        txt = m.group(0)
    return json.loads(txt)

# Conservative stems
STOCK_STEMS   = ["inventory", "stock", "stored", "shelf"]
CAP_STEMS     = ["capacity", "storage", "store", "shelf", "warehouse"]
ASSET_NEG_GUARD = ["asset", "price", "cost", "debt", "payable", "receivable", "liability"]

def _is_stock_like(name: str) -> bool:
    low = name.lower()
    if any(bad in low for bad in ASSET_NEG_GUARD):
        return False
    return any(stem in low for stem in STOCK_STEMS)

def _is_capacity_like(name: str) -> bool:
    low = name.lower()
    return any(stem in low for stem in CAP_STEMS)

def _pick_capacity(self_props: List[str]) -> str:
    # Prefer something包含 'Capacity' 的；否则任何命中 CAP_STEMS 的
    for p in self_props:
        if "Capacity" in p:
            return p
    for p in self_props:
        if _is_capacity_like(p):
            return p
    return ""

def _build_prompt(agent: str, self_props: List[str], relations: List[str]) -> str:
    """
    Hard constraints:
      - Deterministic only; allowed ops: +, -, min(), max(); integers/floats; parentheses ok.
      - Only use provided variable names (Self + Relations). NO new symbols.
      - For stock-like properties, follow the example exactly.
    """
    rels = relations or []
    if "ArrivingOrders" not in rels:
        rels = rels + ["ArrivingOrders"]  # we will zero-fill later if missing at runtime
    if "DownstreamDemand" not in rels:
        rels = rels + ["DownstreamDemand"]

    example = (
        'Example (Retailer):\n'
        'Self: ["OnHandInventory","StoreCapacity"]\n'
        'Relations: ["ArrivingOrders","DownstreamDemand"]\n'
        'Return:\n'
        '{\n'
        '  "rules":[\n'
        '    {"write":"OnHandInventory","expr":"min(StoreCapacity, max(0, OnHandInventory + ArrivingOrders - DownstreamDemand))"},\n'
        '    {"write":"StoreCapacity","expr":"StoreCapacity"}\n'
        '  ]\n'
        '}'
    )

    lines = []
    lines.append("Design fully-deterministic next-step update rules for a supply-chain agent.")
    lines.append("Use ONLY these variables (no new names):")
    lines.append(f"- Self: {self_props}")
    lines.append(f"- Relations: {rels}")
    lines.append("Rules:")
    lines.append("- Allowed only +, -, min(), max(), parentheses, numeric literals.")
    lines.append("- NO probabilities, NO random, NO new symbols.")
    lines.append("- For stock-like properties (names that imply inventory/stock/shelf), follow the inventory conservation pattern like the example below.")
    lines.append("- For other properties (price/cost/capacity/assets), keep identity unless a deterministic formula using ONLY provided symbols is obvious.")
    lines.append("- Output ONE JSON object: {\"rules\":[{\"write\":\"Prop\",\"expr\":\"EXPR\"}, ...]}")
    lines.append(example)
    lines.append("")
    lines.append(f"Agent: {agent}")
    return "\n".join(lines)

def _call_llm(prompt: str) -> Dict[str, Any]:
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY")
    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1200
    )
    return _try_parse_json(resp.choices[0].message.content)

def _sanitize_and_patch(agent: str,
                        self_props: List[str],
                        relations: List[str],
                        rules: List[Dict[str, str]]) -> (List[Dict[str, str]], Dict[str, Any]):
    """
    - Drop any rule whose expr references unknown symbols.
    - Ensure every Self property has a rule (fill identity if missing).
    - Hard enforce stock-law for stock-like properties.
    - If Relations are missing, replace missing Rel by 0 in the expression.
    """
    dbg = {"patched": [], "identity_injected": [], "invalid_expr_fixed": []}

    allowed = set(self_props) | set(relations)
    # If relations are missing, we still allow using the names but we will replace them with 0 at evaluate time.
    has_arr = "ArrivingOrders" in relations
    has_dem = "DownstreamDemand" in relations

    def safe_vars(expr: str) -> List[str]:
        # extremely simple token pick: alnum and underscores treated as names
        names = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr or "")
        # ignore function names min/max
        return [n for n in names if n not in ("min", "max")]

    # Index by write
    by_write = {r.get("write",""): r.get("expr","") for r in (rules or [])}

    # ensure each self prop has a rule
    for p in self_props:
        if p not in by_write:
            by_write[p] = p  # identity
            dbg["identity_injected"].append(p)

    # capacity pick (for clamp)
    cap = _pick_capacity(self_props)

    final_rules = []
    for p, expr in by_write.items():
        expr = (expr or "").strip()
        if not expr:
            expr = p

        # unknown symbol guard
        used = set(safe_vars(expr))
        unknown = [u for u in used if (u not in allowed and u not in ("ArrivingOrders", "DownstreamDemand"))]
        if unknown:
            dbg["invalid_expr_fixed"].append({"write": p, "expr": expr, "unknown": unknown})
            expr = p  # fallback

        # Hard stock-law pass
        if _is_stock_like(p):
            base = f"{p} + {'ArrivingOrders' if has_arr else '0'} - {'DownstreamDemand' if has_dem else '0'}"
            base = f"max(0, {base})"
            if cap:
                expr = f"min({cap}, {base})"
            else:
                expr = base
            dbg["patched"].append({"write": p, "expr": expr, "reason": "stock-law"})

        final_rules.append({"write": p, "expr": expr})

    # stable order: Self order
    name2rule = {r["write"]: r for r in final_rules}
    ordered = [{"write": p, "expr": name2rule[p]["expr"]} for p in self_props if p in name2rule]
    return ordered, dbg

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    pruned = _load_json(PATH_PRUNED) if os.path.exists(PATH_PRUNED) else _load_json(os.path.join(OUTDIR,"final_structure.json"))
    state  = _load_json(PATH_STATE)

    out = {}
    debug = {"agents": {}, "time": int(time.time())}

    for agent, self_props in pruned.items():
        # fallback: if state space missing this agent, assume empty relations
        st = state.get(agent, {})
        relations = st.get("Relations", [])
        prompt = _build_prompt(agent, self_props, relations)
        try:
            js = _call_llm(prompt)
            rules_llm = js.get("rules", [])
        except Exception as e:
            rules_llm = []
            debug["agents"].setdefault(agent, {})["llm_error"] = str(e)

        # sanitize + hard stock-law
        rules, patch_dbg = _sanitize_and_patch(agent, self_props, relations, rules_llm)
        out[agent] = {"rules": rules}
        debug["agents"].setdefault(agent, {})["patch"] = patch_dbg
        debug["agents"][agent]["relations"] = relations
        debug["agents"][agent]["capacity_picked"] = _pick_capacity(self_props)

    with open(PATH_OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(PATH_DBG, "w", encoding="utf-8") as f:
        json.dump(debug, f, ensure_ascii=False, indent=2)
    print(f"Wrote {PATH_OUT}")
    print(f"Wrote {PATH_DBG}")

if __name__ == "__main__":
    main()


