# -*- coding: utf-8 -*-
"""
Strict and robust LLM proposals for State Space and Action Space.

This version wires:
- STATE: Neighbors come ONLY from 1-hop neighbors' PUBLIC properties
         (read from configs/network.json + configs/visibility.json; fallback = all-public).
- STATE: All vocab strictly from outputs/final_structure.json; no synonyms/new tokens.
- RELATIONS: restricted to {"DownstreamDemand","ArrivingOrders"} (0–2 items).
- ACTION: simplified to three types per worklist:
          {OrderPlacement, Produce, SupplierSelection}
          with SupplierSelection params locked to ["AddSupplierId","RemoveSupplierId"].

Drop-in replacement. No other modules need to change to run one sampling pass.
"""

import os
import re
import json
import ast
import time
from typing import Dict, Any, List
from openai import OpenAI

from src.topology import neighbor_public_whitelist  # 1-hop PUBLIC whitelist

# ----------------------------- LLM config -----------------------------
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"
SYSTEM = "You are a supply chain expert. Output ONLY valid JSON. No extra text."

# ================================ Utils ================================
def _get_api_key() -> str:
    key = os.getenv("DASHSCOPE_API_KEY")
    if key:
        return key
    try:
        here = os.path.dirname(__file__)
        with open(os.path.join(here, "..", "configs", "keys.json"), "r", encoding="utf-8") as f:
            j = json.load(f)
            return j.get("DASHSCOPE_API_KEY", "")
    except Exception:
        return ""

def _debug_dump_raw(kind: str, text: str) -> str:
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"llm_raw_{kind}_{int(time.time())}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

def _strip_code_fences(s: str) -> str:
    s = re.sub(r"```(?:json)?\s*", "", s, flags=re.I)
    s = s.replace("```", "")
    return s.strip()

def _smart_to_ascii_quotes(s: str) -> str:
    return (s.replace("“", '"').replace("”", '"')
             .replace("‘", "'").replace("’", "'"))

def _remove_trailing_commas(s: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", s)

def _first_complete_json_block(text: str) -> str:
    s = text
    start = s.find("{")
    if start < 0:
        return ""
    stack = []
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                stack.append("{")
            elif ch == "}":
                if stack:
                    stack.pop()
                    if not stack:
                        return s[start:i+1]
    return ""

def _json_like_cleanup(s: str) -> str:
    s = _strip_code_fences(s)
    s = _smart_to_ascii_quotes(s)
    s = _remove_trailing_commas(s)
    return s.strip()

def _try_parse_json(text: str, *, kind: str) -> Any:
    blob = _first_complete_json_block(text) or text
    blob = _json_like_cleanup(blob)
    try:
        return json.loads(blob)
    except Exception:
        pass
    pyish = (blob.replace("true", "True")
                  .replace("false", "False")
                  .replace("null", "None"))
    try:
        obj = ast.literal_eval(pyish)
        json.dumps(obj, ensure_ascii=False)
        return obj
    except Exception:
        pass
    try:
        obj = json.loads(_remove_trailing_commas(blob))
        return obj
    except Exception:
        path = _debug_dump_raw(kind, text)
        raise ValueError(f"Failed to parse LLM JSON. Raw dumped to: {path}")

def load_final_properties() -> Dict[str, List[str]]:
    """
    Authoritative vocabulary for Self properties per agent.
    Prefer pruned structure if available.
    """
    here = os.path.dirname(__file__)
    outdir = os.path.join(here, "..", "outputs")
    pruned = os.path.join(outdir, "final_structure.pruned.json")
    raw    = os.path.join(outdir, "final_structure.json")
    path = pruned if os.path.isfile(pruned) else raw
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dedup_keep_order(xs: List[str]) -> List[str]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

# ============================ STATE proposal ============================
def _neighbor_prop_whitelist(props: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Build per-agent whitelist for Neighbors from 1-hop neighbors' PUBLIC properties.
    Uses configs/network.json + configs/visibility.json (fallback = default chain, all-public).
    """
    wl: Dict[str, List[str]] = {}
    for ag in props.keys():
        wl[ag] = neighbor_public_whitelist(ag, props)
    return wl

def _build_state_prompt_strict(props: Dict[str, List[str]]) -> str:
    """
    Prompt enforces:
      - Self from agent's OWN list (from final_structure.json)
      - Neighbors from agent's NEIGHBOR WHITELIST (1-hop neighbors' PUBLIC properties)
      - Relations subset of {"DownstreamDemand","ArrivingOrders"}
    """
    neigh_wl = _neighbor_prop_whitelist(props)

    lines: List[str] = []
    lines.append("Design the STATE SPACE for each agent in a supply chain.")
    lines.append('Return ONLY ONE JSON: {"Agent":{"Self":[...],"Neighbors":[...],"Relations":[...]}, ...}')
    lines.append("")
    lines.append("STRICT RULES (do not break):")
    lines.append("1) Self: choose 2–6 items ONLY from that agent's OWN list below.")
    lines.append("2) Neighbors: choose 0–4 items ONLY from THIS AGENT'S NEIGHBOR-WHITELIST below (1-hop upstream & downstream, PUBLIC only).")
    lines.append('3) Relations: choose 0–2 items ONLY from {"DownstreamDemand","ArrivingOrders"}.')
    lines.append("4) Use EXACT tokens; NO new names; NO explanations; NO code fences.")
    lines.append("")
    lines.append("Agent OWN lists (for Self):")
    for ag, ps in props.items():
        lines.append(f"- {ag}: [{', '.join(ps)}]")
    lines.append("")
    lines.append("Per-agent NEIGHBOR-WHITELIST (1-hop neighbors' PUBLIC properties):")
    for ag, wl in neigh_wl.items():
        lines.append(f"- {ag}: [{', '.join(wl)}]")
    lines.append("")
    lines.append('Example: {"Retailer":{"Self":["ShelfInventory","OperatingDebt"],'
                 '"Neighbors":["WholesaleHoldingCost"],"Relations":["DownstreamDemand"]}, ...}')
    return "\n".join(lines)

def llm_propose_state_space() -> Dict[str, Any]:
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY")

    props = load_final_properties()
    neigh_wl = _neighbor_prop_whitelist(props)  # build whitelist once
    prompt = _build_state_prompt_strict(props)

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1200,
    )
    data = _try_parse_json(resp.choices[0].message.content, kind="state")

    cleaned: Dict[str, Any] = {}
    for ag, own_list in props.items():
        spec = data.get(ag, {})
        self_in  = spec.get("Self", [])
        neigh_in = spec.get("Neighbors", [])
        rel_in   = spec.get("Relations", [])

        # Strict filtering
        self_ok  = [x for x in self_in  if x in own_list]
        neigh_ok = [x for x in neigh_in if x in set(neigh_wl.get(ag, []))]
        rel_ok   = [x for x in rel_in   if x in {"DownstreamDemand", "ArrivingOrders"}]

        # De-dup + clamp
        self_ok  = _dedup_keep_order(self_ok)[:6]
        neigh_ok = _dedup_keep_order(neigh_ok)[:4]
        rel_ok   = _dedup_keep_order(rel_ok)[:2]

        # Ensure Self minimum size = 2 (fallback from own_list)
        if len(self_ok) < 2:
            for x in own_list:
                if x not in self_ok:
                    self_ok.append(x)
                if len(self_ok) >= 2:
                    break

        cleaned[ag] = {"Self": self_ok, "Neighbors": neigh_ok, "Relations": rel_ok}
    return cleaned

# ============================ ACTION proposal (simplified) ============================

_ALLOWED_ACTIONS = {
    "OrderPlacement", "Produce", "SupplierSelection"
}

_PARAM_WHITELIST = {
    "Quantity", "To", "AddSupplierId", "RemoveSupplierId"
}

# Default params (used as fallback only)
_DEFAULT_PARAMS = {
    "OrderPlacement": ["To", "Quantity"],
    "Produce": ["Quantity"],
    "SupplierSelection": ["AddSupplierId", "RemoveSupplierId"]
}

# Role priors (guidance only)
ROLE_PRIORS = {
    "Supplier": ["SupplierSelection"],
    "Manufacturer": ["Produce", "OrderPlacement", "SupplierSelection"],
    "Distributor":  ["OrderPlacement", "SupplierSelection"],
    "Wholesaler":   ["OrderPlacement", "SupplierSelection"],
    "Retailer":     ["OrderPlacement", "SupplierSelection"],
}

# Role minimum backstop (hard): ensure minimal sensible action per agent
ROLE_MUST_AT_LEAST_ONE = {
    "Manufacturer": {"Produce"},
    "Distributor": {"OrderPlacement"},
    "Wholesaler": {"OrderPlacement"},
    "Retailer": {"OrderPlacement"}
    # Supplier has no hard minimum here; SupplierSelection is optional.
}

_REQUIRED_PARAMS = {
    "OrderPlacement": {"To", "Quantity"},
}

# Explicit fixed params for SupplierSelection
_EXPLICIT_SUPPLIER_SELECTION = ["AddSupplierId", "RemoveSupplierId"]

_PARAM_ALIASES = {
    "Destination": "To",
    "Target": "To",
    "Receiver": "To",
    "Amount": "Quantity",
    "Qty": "Quantity"
}

def _alias_param(p: str) -> str:
    return _PARAM_ALIASES.get(p, p)

def _enforce_required_params(action_type: str, params: List[str]) -> List[str]:
    """
    Normalize & enforce action parameters:
      - alias to canonical names
      - whitelist + de-dup
      - add per-action required params from _REQUIRED_PARAMS
      - force explicit param set for SupplierSelection: ["AddSupplierId","RemoveSupplierId"]
      - cap length (<=3) for other actions
      - fallback to _DEFAULT_PARAMS if empty
    """
    # 1) Alias normalization + whitelist filter
    params = [_alias_param(p) for p in params if isinstance(p, str)]
    seen = set()
    filtered: List[str] = []
    for p in params:
        if p in _PARAM_WHITELIST and p not in seen:
            seen.add(p)
            filtered.append(p)

    # 2) Add per-action required params (if declared)
    for r in _REQUIRED_PARAMS.get(action_type, set()):
        if r not in seen:
            filtered.append(r)
            seen.add(r)

    # 3) Hard-lock SupplierSelection
    if action_type == "SupplierSelection":
        filtered = _EXPLICIT_SUPPLIER_SELECTION[:]
    else:
        # Keep at most 3 params for other actions
        filtered = filtered[:3]

    # 4) Fallback if empty
    if not filtered:
        filtered = _DEFAULT_PARAMS.get(action_type, [])[:3]

    return filtered

def _build_action_prompt(props: Dict[str, List[str]]) -> str:
    lines: List[str] = []
    lines.append("Design role-appropriate ACTION TYPES for each agent in a supply chain.")
    lines.append('Return ONLY ONE JSON: {"Agent":[{"type":"ActionName","params":["ParamA","ParamB"]}, ...], ...}')
    lines.append("")
    lines.append("STRICT RULES:")
    lines.append("- Use DOUBLE QUOTES for all keys and strings.")
    lines.append("- Provide 2–3 actions per agent.")
    lines.append("- Action names MUST be chosen from:")
    lines.append("  [OrderPlacement, Produce, SupplierSelection]")
    lines.append("- Parameter names MUST be chosen from:")
    lines.append("  [Quantity, To, AddSupplierId, RemoveSupplierId]")
    lines.append("- No comments, no trailing commas, no code fences, no explanations.")
    lines.append("")
    lines.append("Role priors (guidance only):")
    for ag, pri in ROLE_PRIORS.items():
        lines.append(f"- {ag}: [{', '.join(pri)}]")
    lines.append("")
    lines.append("Agent properties (context only):")
    for ag, ps in props.items():
        lines.append(f"- {ag}: [{', '.join(ps)}]")
    return "\n".join(lines)

def _ensure_role_minimum(actions_by_agent: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure each agent has at least one sensible action; de-dup; cap to 2–3.
    """
    fixed = {}
    for agent, acts in actions_by_agent.items():
        seen = {a.get("type", "") for a in acts if isinstance(a, dict)}
        need = ROLE_MUST_AT_LEAST_ONE.get(agent, set())
        # Inject a minimum action if none of the required ones present
        if need and not (seen & need):
            req = sorted(list(need))[0]
            acts = list(acts) + [{"type": req, "params": _DEFAULT_PARAMS.get(req, [])[:3]}]

        # De-duplicate by type, and lock SupplierSelection params if present
        uniq, s2 = [], set()
        for a in acts:
            t = a.get("type", "")
            if t and t not in s2:
                if t == "SupplierSelection":
                    a = {"type": t, "params": _EXPLICIT_SUPPLIER_SELECTION[:]}
                uniq.append(a)
                s2.add(t)

        # Ensure lower bound (2) by adding a generic OrderPlacement when applicable
        if len(uniq) < 2:
            if agent != "Supplier" and "OrderPlacement" not in s2:
                uniq.append({"type": "OrderPlacement", "params": _DEFAULT_PARAMS["OrderPlacement"]})

        fixed[agent] = uniq[:3]  # cap to 3 actions
    return fixed

def llm_propose_action_space() -> Dict[str, Any]:
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY")

    props = load_final_properties()
    prompt = _build_action_prompt(props)

    client = OpenAI(api_key=api_key, base_url=BASE_URL)

    extra_kwargs = {}
    if os.getenv("FORCE_JSON_MODE", "0") == "1":
        extra_kwargs["response_format"] = {"type": "json_object"}

    last_err = None
    for attempt in range(3):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1500,
            **extra_kwargs,
        )
        raw = resp.choices[0].message.content
        try:
            js = _try_parse_json(raw, kind="action")

            cleaned: Dict[str, Any] = {}
            for ag in props.keys():
                acts = []
                for a in js.get(ag, []):
                    t = a.get("type")
                    if t not in _ALLOWED_ACTIONS:
                        continue
                    raw_params = a.get("params", [])
                    fixed_params = _enforce_required_params(t, raw_params)
                    acts.append({"type": t, "params": fixed_params})
                cleaned[ag] = acts

            cleaned = _ensure_role_minimum(cleaned)
            return cleaned

        except Exception as e:
            last_err = e
            prompt = prompt + "\n\nREMINDER: Return ONE VALID JSON only. No comments, no trailing commas, double quotes only."
            continue

    # Fallback (safe minimal set aligned with the simplified action space)
    fallback = {
        "Supplier":    [{"type": "SupplierSelection", "params": _EXPLICIT_SUPPLIER_SELECTION[:]}],
        "Manufacturer":[{"type": "Produce", "params": _DEFAULT_PARAMS["Produce"]},
                        {"type": "OrderPlacement", "params": _DEFAULT_PARAMS["OrderPlacement"]},
                        {"type": "SupplierSelection", "params": _EXPLICIT_SUPPLIER_SELECTION[:]}],
        "Distributor": [{"type": "OrderPlacement", "params": _DEFAULT_PARAMS["OrderPlacement"]},
                        {"type": "SupplierSelection", "params": _EXPLICIT_SUPPLIER_SELECTION[:]}],
        "Wholesaler":  [{"type": "OrderPlacement", "params": _DEFAULT_PARAMS["OrderPlacement"]},
                        {"type": "SupplierSelection", "params": _EXPLICIT_SUPPLIER_SELECTION[:]}],
        "Retailer":    [{"type": "OrderPlacement", "params": _DEFAULT_PARAMS["OrderPlacement"]},
                        {"type": "SupplierSelection", "params": _EXPLICIT_SUPPLIER_SELECTION[:]}],
    }
    _debug_dump_raw("action_fallback", f"[ParseFailed] {str(last_err)}")
    return fallback









