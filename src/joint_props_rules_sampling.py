# -*- coding: utf-8 -*-
"""
Joint sampling for property definitions and deterministic transition rules.

This script:
- Loads final_structure.pruned.json (or final_structure.json) as the authoritative
  Self property list per agent.
- Loads final_state_space.json for Relations (e.g., ArrivingOrders, DownstreamDemand).
- Loads the supply-chain topology from configs/network.json (or uses default chain).
- For EACH agent, runs a small MH-like loop where an LLM jointly proposes:
    * a category, **domain-specific type**, and English definition for every Self property;
    * for each relational property, a from/to list + English definition;
    * deterministic next-step update rules for ALL Self properties.
- The code then sanitizes and patches rules:
    * removes unknown symbols;
    * enforces an inventory conservation law for stock-like properties.

Outputs:
    outputs/final_property_definitions.json
    outputs/final_transition_rules.json
    outputs/transition_rules_debug.json

You can run:
    python -m src.joint_props_rules_sampling
"""

import os
import re
import json
import time
import math
import random
from typing import Dict, Any, List, Tuple
from collections import defaultdict

from openai import OpenAI
from src.topology import load_network
from src.joint_rules_scoring import score_joint_sample

# -------------------------- Paths --------------------------

HERE = os.path.dirname(__file__)
OUTDIR = os.path.join(HERE, "..", "outputs")
PATH_PRUNED = os.path.join(OUTDIR, "final_structure.pruned.json")
PATH_STRUCT = os.path.join(OUTDIR, "final_structure.json")
PATH_STATE = os.path.join(OUTDIR, "final_state_space.json")

PATH_OUT_DEFS = os.path.join(OUTDIR, "final_property_definitions.json")
PATH_OUT_RULES = os.path.join(OUTDIR, "final_transition_rules.json")
PATH_OUT_DEBUG = os.path.join(OUTDIR, "transition_rules_debug.json")

# --------------------- LLM / API helpers -------------------

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"
SYSTEM = (
    "You are a precise supply chain expert. You output ONLY valid JSON. "
    "No prose. No code fences. Deterministic arithmetic only."
)

ALLOWED_CATEGORIES = {"Stock", "Capacity", "Cost", "Asset", "Price", "Other"}
# New set of domain-specific semantic types (the "firewall" concepts)
ALLOWED_TYPES = {
    "DiscreteVolume",  # e.g., Inventory, units
    "MonetaryValue",  # e.g., Price, Cost, Cash
    "TimeValue",  # e.g., LeadTime, DeliveryDelay
    "BooleanFlag",  # e.g., IsStockout, HasCapacity
    "Ratio",  # e.g., ServiceLevel, MarkupRate
}


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


def _strip_code_fences(s: str) -> str:
    s = re.sub(r"```(?:json)?\s*", "", s, flags=re.I)
    s = s.replace("```", "")
    return s.strip()


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
                        return s[start: i + 1]
    return ""


def _json_like_cleanup(s: str) -> str:
    s = _strip_code_fences(s)
    # make sure smart quotes are normalized
    s = (
        s.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    # best-effort removal of dangling commas
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s.strip()


def _try_parse_json(text: str) -> Any:
    blob = _first_complete_json_block(text) or text
    blob = _json_like_cleanup(blob)
    try:
        return json.loads(blob)
    except Exception:
        # last resort: try again after removing trailing commas
        blob2 = re.sub(r",\s*([}\]])", r"\1", blob)
        return json.loads(blob2)


def _call_llm(prompt: str) -> Dict[str, Any]:
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY (env or configs/keys.json)")

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1800,
    )
    raw = resp.choices[0].message.content
    return _try_parse_json(raw)


# --------------------- Data loading helpers ---------------------


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_properties() -> Dict[str, List[str]]:
    """
    Authoritative Self properties per agent.
    Prefer pruned structure if present.
    """
    if os.path.exists(PATH_PRUNED):
        return _load_json(PATH_PRUNED)
    if os.path.exists(PATH_STRUCT):
        return _load_json(PATH_STRUCT)
    raise FileNotFoundError(
        "Neither final_structure.pruned.json nor final_structure.json found in outputs/."
    )


def _load_state_space() -> Dict[str, Any]:
    if not os.path.exists(PATH_STATE):
        raise FileNotFoundError(
            "final_state_space.json not found in outputs/. "
            "Run state_action_mcmc.py first or provide this file manually."
        )
    return _load_json(PATH_STATE)


def _neighbors_from_topology() -> Dict[str, Dict[str, List[str]]]:
    """
    Build upstream / downstream neighbor lists for each agent from the topology.
    """
    net = load_network()
    agents = net.get("agents", [])
    edges = net.get("edges", [])
    upstream = {a: set() for a in agents}
    downstream = {a: set() for a in agents}
    for e in edges:
        src = e.get("from")
        dst = e.get("to")
        if src in agents and dst in agents:
            downstream[src].add(dst)
            upstream[dst].add(src)
    out = {}
    for a in agents:
        out[a] = {
            "upstream": sorted(upstream[a]),
            "downstream": sorted(downstream[a]),
        }
    return out


# ------------------- Stock / capacity heuristics -------------------


STOCK_STEMS = ["inventory", "stock", "stored", "shelf"]
CAP_STEMS = ["capacity", "storage", "store", "shelf", "warehouse"]
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
    """
    Prefer a property containing 'Capacity'; otherwise any capacity-like property.
    """
    for p in self_props:
        if "Capacity" in p:
            return p
    for p in self_props:
        if _is_capacity_like(p):
            return p
    return ""


def _safe_vars(expr: str) -> List[str]:
    """
    Extract variable-like tokens from an expression.
    Ignores function names min/max.
    """
    names = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr or "")
    return [n for n in names if n not in ("min", "max")]


# ------------------- Prompt builder -------------------


def _build_prompt_for_agent(
        agent: str,
        self_props: List[str],
        relations: List[str],
        neigh_meta: Dict[str, List[str]],
) -> str:
    """
    Build a joint prompt: property typing + definitions + deterministic rules.
    Modified to include domain-specific 'type' field.
    """
    ups = neigh_meta.get("upstream", [])
    downs = neigh_meta.get("downstream", [])

    lines: List[str] = []
    lines.append("You design the STATE PROPERTIES and DETERMINISTIC UPDATE RULES")
    lines.append("for a single agent in a supply chain simulation.")
    lines.append("")
    lines.append(f"Agent name: {agent}")
    lines.append(f"Self properties of this agent (owned state): {self_props}")
    lines.append(f"Relational properties visible at this agent: {relations}")
    lines.append(f"Upstream neighbors (who send products/orders to {agent}): {ups}")
    lines.append(f"Downstream neighbors (who receive products/orders from {agent}): {downs}")
    lines.append("")
    lines.append("TASK 1: For each Self property, assign:")
    lines.append('- a category from ["Stock","Capacity","Cost","Asset","Price","Other"];')
    lines.append(
        f"- **a precise domain-specific 'type'** from {list(ALLOWED_TYPES)} (this acts as a semantic guardrail);")
    lines.append("- a short, clear, one-sentence English definition.")
    lines.append("")
    lines.append("TASK 2: For each relational property (name in the Relations list):")
    lines.append(
        "- Specify who it flows FROM and TO, using a 'from' field (list of agent names) "
        "and a 'to' field (single agent name)."
    )
    lines.append(
        "- **Assign a flow 'type'** from the semantic list (e.g., 'DiscreteVolume' for quantities, 'MonetaryValue' for payments).")
    lines.append(
        "- Write a short English definition explaining what quantity or information it represents."
    )
    lines.append(
        "- Typically, ArrivingOrders is shipments from upstream neighbors to this agent; "
        "DownstreamDemand is orders from downstream neighbors to this agent."
    )
    lines.append("")
    lines.append("TASK 3: Deterministic next-step update RULES for every Self property.")
    lines.append(
        "- Each rule writes exactly one Self property and uses only allowed variables: "
        "Self properties + relational properties listed above."
    )
    lines.append(
        "- Allowed operations: +, -, min(), max(), parentheses, numeric constants. "
        "NO random, NO new symbols."
    )
    lines.append(
        "- For inventory/stock-like properties (type: DiscreteVolume), "
        "you MUST follow an inventory conservation pattern similar to:"
    )
    lines.append(
        '  OnHandInventory_next = min(CapacityVar, max(0, OnHandInventory + ArrivingOrders - DownstreamDemand))'
    )
    lines.append(
        "- If there is no natural capacity property, you may omit the min(capacity, ...) clamp and only use max(0,...)."
    )
    lines.append("")
    lines.append("Return ONLY ONE JSON object with EXACTLY this schema:")
    lines.append("{")
    lines.append('  "properties": [')
    lines.append(
        '    {"name": "OnHandInventory", '
        '"category": "Stock", '
        '"type": "DiscreteVolume", '  # Added 'type' field
        '"definition": "Current discrete inventory volume of this agent (number of units on hand)."}'
    )
    lines.append("  ],")
    lines.append('  "relations": [')
    lines.append(
        '    {"name": "ArrivingOrders", '
        '"from": ["UpstreamAgent1"], '
        '"to": "AgentName", '
        '"type": "DiscreteVolume", '  # Added 'type' field
        '"definition": "Units shipped by upstream agents that arrive at this agent in the current period."}'
    )
    lines.append("  ],")
    lines.append('  "rules": [')
    lines.append(
        '    {"write": "OnHandInventory", '
        '"expr": "min(StorageCapacity, max(0, OnHandInventory + ArrivingOrders - DownstreamDemand))"}'
    )
    lines.append("  ]")
    lines.append("}")
    lines.append("")
    lines.append("Do NOT include comments or code fences. Only this JSON object.")

    return "\n".join(lines)


# ------------------- Candidate processing -------------------


def _process_candidate_for_agent(
        agent: str,
        self_props: List[str],
        relations: List[str],
        cand: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, str]], List[Dict[str, Any]], List[Dict[str, str]], Dict[str, Any]]:
    """
    Turn raw LLM candidate into:
      - prop_defs: {name: {"category": str, "type": str, "definition": str}}
      - rel_defs:  [{"name": str, "from": [...], "to": str, "type": str, "definition": str}, ...]
      - sanitized_rules: [{"write": str, "expr": str}, ...]
      - patch_debug: dict with patch info
    Modified to capture and normalize the new 'type' field.
    """
    # ----- properties -----
    prop_defs: Dict[str, Dict[str, str]] = {}
    for item in cand.get("properties", []):
        name = item.get("name")
        if not isinstance(name, str):
            continue
        if name not in self_props:
            continue

        cat = item.get("category", "Other")
        if cat not in ALLOWED_CATEGORIES:
            cat = "Other"

        p_type = item.get("type", "MonetaryValue")  # Default assumption
        if p_type not in ALLOWED_TYPES:
            # Fallback based on category heuristics if type is invalid/missing
            if cat in {"Stock", "Capacity"}:
                p_type = "DiscreteVolume"
            elif cat in {"Cost", "Asset", "Price"}:
                p_type = "MonetaryValue"
            else:
                p_type = "MonetaryValue"

        definition = (item.get("definition") or "").strip()
        prop_defs[name] = {"category": cat, "type": p_type, "definition": definition}

    # ----- relations -----
    rel_defs: List[Dict[str, Any]] = []
    seen_rel = set()
    for item in cand.get("relations", []):
        name = item.get("name")
        if not isinstance(name, str) or (name not in relations):
            continue

        r_type = item.get("type", "DiscreteVolume")  # Default flow type
        if r_type not in ALLOWED_TYPES:
            r_type = "DiscreteVolume"  # Fallback

        frm = item.get("from", [])
        to = item.get("to", agent)
        if isinstance(frm, str):
            frm = [frm]
        if not isinstance(frm, list):
            frm = []
        frm_clean = [str(x) for x in frm]
        to_clean = str(to)
        definition = (item.get("definition") or "").strip()
        key = (name, to_clean)
        if key in seen_rel:
            continue
        seen_rel.add(key)
        rel_defs.append(
            {
                "name": name,
                "from": frm_clean,
                "to": to_clean,
                "type": r_type,  # Added 'type' field
                "definition": definition,
            }
        )

    # ----- rules (raw) -----
    rules_raw: List[Dict[str, str]] = []
    for item in cand.get("rules", []):
        w = item.get("write")
        e = item.get("expr")
        if not isinstance(w, str):
            continue
        if w not in self_props:
            continue
        rules_raw.append({"write": w, "expr": (e or "").strip()})

    # sanitize + patch
    sanitized_rules, patch_debug = _sanitize_and_patch_rules(agent, self_props, prop_defs, relations, rules_raw)
    return prop_defs, rel_defs, sanitized_rules, patch_debug


def _sanitize_and_patch_rules(
        agent: str,
        self_props: List[str],
        prop_defs: Dict[str, Dict[str, str]],  # Pass property definitions for type info
        relations: List[str],
        rules_raw: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """
    - Deduplicate by write;
    - Drop expressions using unknown symbols;
    - Ensure each Self property has a rule (identity fallback);
    - For stock-like properties, enforce inventory conservation law;
    - If ArrivingOrders/DownstreamDemand are missing from Relations, treat them as 0.
    """
    dbg = {"identity_injected": [], "invalid_expr_fixed": [], "patched_stock_law": []}

    allowed_names = set(self_props) | set(relations)
    has_arr = "ArrivingOrders" in relations
    has_dem = "DownstreamDemand" in relations

    # index by write; last one wins
    by_write: Dict[str, str] = {}
    for r in rules_raw:
        w = str(r.get("write", "")).strip()
        e = (r.get("expr") or "").strip()
        if not w or w not in self_props:
            continue
        by_write[w] = e

    # ensure each self property has at least an identity rule
    for p in self_props:
        if p not in by_write:
            by_write[p] = p
            dbg["identity_injected"].append(p)

    cap = _pick_capacity(self_props)

    final_rules: List[Dict[str, str]] = []
    for p, expr in by_write.items():
        expr = (expr or "").strip()
        if not expr:
            expr = p

        # unknown symbol guard
        used = set(_safe_vars(expr))
        unknown = [u for u in used if (u not in allowed_names)]
        if unknown:
            dbg["invalid_expr_fixed"].append({"write": p, "expr": expr, "unknown": unknown})
            expr = p  # fallback to identity

        # Get semantic type for hard patching
        p_type = prop_defs.get(p, {}).get("type")

        # hard stock-law patch for properties of type DiscreteVolume/Stock
        if p_type == "DiscreteVolume" or _is_stock_like(p):
            base = f"{p} + {'ArrivingOrders' if has_arr else '0'} - {'DownstreamDemand' if has_dem else '0'}"
            # DiscreteVolume must be non-negative (max(0, ...))
            base = f"max(0, {base})"
            if cap:
                # Stock/Capacity must be bounded (min(Cap, ...))
                expr = f"min({cap}, {base})"
            else:
                expr = base
            dbg["patched_stock_law"].append({"write": p, "expr": expr, "type": p_type})

        final_rules.append({"write": p, "expr": expr})

    # stable order: follow Self property list order
    name2rule = {r["write"]: r for r in final_rules}
    ordered = [{"write": p, "expr": name2rule[p]["expr"]} for p in self_props if p in name2rule]
    return ordered, dbg


# ------------------- Scoring for MH loop -------------------


def _score_candidate(
        agent: str,
        self_props: List[str],
        relations: List[str],
        neigh_meta: Dict[str, List[str]],
        prop_defs: Dict[str, Dict[str, str]],
        rel_defs: List[Dict[str, Any]],
        rules: List[Dict[str, str]],
) -> float:
    """
    Thin wrapper that adapts the per-agent candidate into the joint
    scoring format expected by score_joint_sample().
    """
    # The structure must be compatible with the joint scoring function
    props_packet = {
        agent: {
            "properties": prop_defs,
            "relations": rel_defs,
        }
    }
    rules_packet = {
        agent: {
            "rules": rules,
        }
    }
    # score_joint_sample now receives the new 'type' field implicitly
    score, _debug = score_joint_sample(props_packet, rules_packet)
    return score


# ------------------- MH loop per agent -------------------


def _run_agent_mcmc(
        agent: str,
        self_props: List[str],
        relations: List[str],
        neigh_meta: Dict[str, List[str]],
        n_rounds: int = 6,
        beta: float = 1.0,
) -> Tuple[Dict[str, Dict[str, str]], List[Dict[str, Any]], List[Dict[str, str]], Dict[str, Any]]:
    """
    MH-like loop over joint (definitions + rules) proposals for one agent.

    Returns the best candidate's:
      - property definitions (now includes 'type')
      - relation definitions (now includes 'type')
      - sanitized rules
      - debug info
    """
    debug_agent = {"mcmc": [], "patch": {}}

    # initial proposal
    prompt = _build_prompt_for_agent(agent, self_props, relations, neigh_meta)
    cand_raw = _call_llm(prompt)
    prop_defs, rel_defs, rules, patch_dbg = _process_candidate_for_agent(
        agent, self_props, relations, cand_raw
    )
    curr_score = _score_candidate(agent, self_props, relations, neigh_meta, prop_defs, rel_defs, rules)

    best = {
        "score": curr_score,
        "prop_defs": prop_defs,
        "rel_defs": rel_defs,
        "rules": rules,
        "patch": patch_dbg,
    }

    debug_agent["mcmc"].append({"iter": 0, "score": curr_score, "accepted": True})
    debug_agent["patch"] = patch_dbg

    current = (prop_defs, rel_defs, rules)

    for it in range(1, n_rounds):
        cand_raw = _call_llm(prompt)
        prop_defs_new, rel_defs_new, rules_new, patch_dbg_new = _process_candidate_for_agent(
            agent, self_props, relations, cand_raw
        )
        cand_score = _score_candidate(
            agent, self_props, relations, neigh_meta, prop_defs_new, rel_defs_new, rules_new
        )

        acc_ratio = min(1.0, math.exp(beta * (cand_score - curr_score)))
        accepted = random.random() < acc_ratio

        if accepted:
            current = (prop_defs_new, rel_defs_new, rules_new)
            curr_score = cand_score
            patch_dbg = patch_dbg_new

        # track best
        if cand_score > best["score"]:
            best = {
                "score": cand_score,
                "prop_defs": prop_defs_new,
                "rel_defs": rel_defs_new,
                "rules": rules_new,
                "patch": patch_dbg_new,
            }

        debug_agent["mcmc"].append(
            {
                "iter": it,
                "score": cand_score,
                "accepted": accepted,
                "acc_ratio": acc_ratio,
            }
        )

    return (
        best["prop_defs"],
        best["rel_defs"],
        best["rules"],
        debug_agent,
    )


# ------------------- Main orchestration -------------------


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    props_by_agent = _load_properties()
    state_space = _load_state_space()
    neighbors = _neighbors_from_topology()

    all_prop_defs: Dict[str, Dict[str, Any]] = {}
    all_rules: Dict[str, Dict[str, Any]] = {}
    debug_all: Dict[str, Any] = {"time": int(time.time()), "agents": {}}

    for agent, self_props in props_by_agent.items():
        # find relations from state space (may be missing)
        st = state_space.get(agent, {})
        rels = st.get("Relations", [])
        neigh_meta = neighbors.get(agent, {"upstream": [], "downstream": []})

        (
            prop_defs,
            rel_defs,
            rules,
            debug_agent,
        ) = _run_agent_mcmc(agent, self_props, rels, neigh_meta, n_rounds=6, beta=1.0)

        # store property definitions + relation definitions
        # These now include the 'type' field
        all_prop_defs[agent] = {
            "properties": prop_defs,  # {name: {category, type, definition}}
            "relations": rel_defs,  # list of {name, from, to, type, definition}
        }

        # store rules in old-compatible format
        all_rules[agent] = {"rules": rules}

        debug_agent["neighbors"] = neigh_meta
        debug_all["agents"][agent] = debug_agent

    # write outputs
    with open(PATH_OUT_DEFS, "w", encoding="utf-8") as f:
        json.dump(all_prop_defs, f, ensure_ascii=False, indent=2)
    with open(PATH_OUT_RULES, "w", encoding="utf-8") as f:
        json.dump(all_rules, f, ensure_ascii=False, indent=2)
    with open(PATH_OUT_DEBUG, "w", encoding="utf-8") as f:
        json.dump(debug_all, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {PATH_OUT_DEFS}")
    print(f"Wrote: {PATH_OUT_RULES}")
    print(f"Wrote: {PATH_OUT_DEBUG}")


if __name__ == "__main__":
    main()
