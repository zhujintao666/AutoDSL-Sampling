# -*- coding: utf-8 -*-
"""
Scoring for joint (properties + relations + deterministic rules) proposals
on small subgraphs.

Interface:
    score_joint_sample(prop_packet, rules_packet) -> (score: float, debug: dict)

prop_packet example (UPDATED to include 'type'):
{
  "Supplier": {
    "properties": {
      "Inventory": {"category": "Stock", "type": "DiscreteVolume", "definition": "..."},
      ...
    },
    "relations": [
      {"name": "ArrivingOrders", "from": ["Manufacturer"], "to": "Supplier", "type": "DiscreteVolume", "definition": "..."},
      ...
    ]
  },
  ...
}
"""

from __future__ import annotations

import re
from typing import Dict, Any, List, Tuple

from src.topology import load_network

# The new semantic types must be accessible for scoring logic
ALLOWED_CATEGORIES = {"Stock", "Capacity", "Cost", "Asset", "Price", "Other"}
ALLOWED_TYPES = {
    "DiscreteVolume",
    "MonetaryValue",
    "TimeValue",
    "BooleanFlag",
    "Ratio",
}


# ----------------- Small helpers -----------------


def _safe_vars(expr: str) -> List[str]:
    """Extract variable-like tokens from an expression, ignoring common fn names."""
    names = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr or "")
    return [n for n in names if n not in {"min", "max"}]


STOCK_TOKENS = ["inventory", "stock", "stored", "onhand", "on_hand", "shelf"]
CAP_TOKENS = ["capacity", "storage", "store", "warehouse"]
CASH_TOKENS = ["cash"]
PRICE_TOKENS = ["price"]
COST_TOKENS = ["cost"]


def _is_stock_like(name: str) -> bool:
    low = name.lower()
    return any(tok in low for tok in STOCK_TOKENS)


def _is_capacity_like(name: str) -> bool:
    low = name.lower()
    return any(tok in low for tok in CAP_TOKENS)


def _is_cash_like(name: str) -> bool:
    low = name.lower()
    return any(tok in low for tok in CASH_TOKENS)


def _is_price_like(name: str) -> bool:
    low = name.lower()
    return any(tok in low for tok in PRICE_TOKENS)


def _is_cost_like(name: str) -> bool:
    low = name.lower()
    return any(tok in low for tok in COST_TOKENS)


def _neighbors_from_topology() -> Dict[str, Dict[str, List[str]]]:
    """
    Build upstream / downstream neighbor lists from global network topology.
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


# ----------------- Core scoring -----------------


def _score_properties_for_agent(agent: str, blk: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Score property coverage, category coherence, and now, type alignment.
    """
    props = blk.get("properties", {}) or {}
    rels = blk.get("relations", []) or {}

    s = 0.0
    dbg = {"props": {}, "rels": {}}

    # property coverage + category coherence + TYPE COHERENCE
    for pname, meta in props.items():
        meta = meta or {}
        cat = meta.get("category", "Other")
        p_type = meta.get("type")  # NEW: Get semantic type
        definition = (meta.get("definition") or "").strip()

        ps = 0.0
        if definition:
            ps += 1.0
        if cat in ALLOWED_CATEGORIES:
            ps += 0.5

        # NEW: Reward for having a valid semantic type
        if p_type in ALLOWED_TYPES:
            ps += 0.5

        # semantic alignment between name and category (legacy)
        if cat == "Stock" and _is_stock_like(pname):
            ps += 0.3
        if cat == "Capacity" and _is_capacity_like(pname):
            ps += 0.3
        if cat == "Asset" and _is_cash_like(pname):
            ps += 0.3
        if cat == "Price" and _is_price_like(pname):
            ps += 0.3
        if cat == "Cost" and _is_cost_like(pname):
            ps += 0.3

        # NEW: Semantic alignment between type and category/name (future: this will replace legacy checks)
        if p_type == "DiscreteVolume" and cat in {"Stock", "Capacity"}:
            ps += 0.3
        if p_type == "MonetaryValue" and cat in {"Price", "Cost", "Asset"}:
            ps += 0.3

        dbg["props"][pname] = ps
        s += ps

    # relations are partially scored globally; ensure definition and type presence
    for r in rels:
        name = (r or {}).get("name")
        if not isinstance(name, str):
            continue
        r_type = r.get("type")  # NEW: Get relational type
        rd = (r.get("definition") or "").strip()
        rs = 0.0
        if rd:
            rs += 0.5
        if r_type in ALLOWED_TYPES:  # NEW: Reward for valid relational type
            rs += 0.5

        dbg["rels"][name] = rs
        s += rs

    return s, dbg


def _score_relations_topology(
        agent: str,
        blk: Dict[str, Any],
        neighbor_meta: Dict[str, List[str]],
) -> Tuple[float, Dict[str, Any]]:
    """Score whether relation from/to directions align with topology."""
    rels = blk.get("relations", []) or []
    ups = set(neighbor_meta.get("upstream", []))
    downs = set(neighbor_meta.get("downstream", []))
    neigh = ups | downs

    s = 0.0
    dbg: Dict[str, float] = {}

    for r in rels:
        name = (r or {}).get("name")
        if not isinstance(name, str):
            continue

        r_type = r.get("type")  # NEW: Get type
        frm = r.get("from", [])
        if isinstance(frm, str):
            frm = [frm]
        frm_set = set(str(x) for x in frm)
        to = str(r.get("to", ""))

        rs = 0.0
        # "to" should be the current agent
        if to == agent:
            rs += 0.3

        if name == "ArrivingOrders":
            # ArrivingOrders typically carries DiscreteVolume or FlowQuantity type
            if r_type == "DiscreteVolume":
                rs += 0.2

            # ideally from upstream neighbors
            if frm_set & ups:
                rs += 0.7
            # also accept "from neighbors" in general (weaker reward)
            elif frm_set & neigh:
                rs += 0.3
        elif name == "DownstreamDemand":
            # DownstreamDemand typically carries DiscreteVolume or FlowQuantity type
            if r_type == "DiscreteVolume":
                rs += 0.2

            # ideally from downstream neighbors
            if frm_set & downs:
                rs += 0.7
            elif frm_set & neigh:
                rs += 0.3
        else:
            # unknown relation names: small reward if they at least come from neighbors
            if frm_set & neigh:
                rs += 0.3

        dbg[name] = rs
        s += rs

    return s, dbg


def _get_prop_type_map(props_blk: Dict[str, Any]) -> Dict[str, str]:
    """Helper to map property name to its semantic type."""
    prop_map = {}
    for pname, meta in (props_blk.get("properties", {}) or {}).items():
        prop_map[pname] = meta.get("type", "MonetaryValue")
    return prop_map


def _capacity_vars_for_agent(props_blk: Dict[str, Any]) -> List[str]:
    """Collect names of capacity-like properties for rule scoring."""
    out: List[str] = []
    prop_map = _get_prop_type_map(props_blk)
    for pname, meta in (props_blk.get("properties", {}) or {}).items():
        meta = meta or {}
        cat = meta.get("category")
        p_type = prop_map.get(pname)
        if cat == "Capacity" or _is_capacity_like(pname) or p_type == "DiscreteVolume":
            out.append(pname)
    return out


def _price_vars_for_agent(props_blk: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    prop_map = _get_prop_type_map(props_blk)
    for pname, meta in (props_blk.get("properties", {}) or {}).items():
        meta = meta or {}
        cat = meta.get("category")
        p_type = prop_map.get(pname)
        if cat == "Price" or _is_price_like(pname) or (p_type == "MonetaryValue" and "Price" in pname):
            out.append(pname)
    return out


def _cost_vars_for_agent(props_blk: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    prop_map = _get_prop_type_map(props_blk)
    for pname, meta in (props_blk.get("properties", {}) or {}).items():
        meta = meta or {}
        cat = meta.get("category")
        p_type = prop_map.get(pname)
        if cat == "Cost" or _is_cost_like(pname) or (p_type == "MonetaryValue" and "Cost" in pname):
            out.append(pname)
    return out


def _score_rules_for_agent(
        agent: str,
        props_blk: Dict[str, Any],
        rules_blk: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """
    Score rules, incorporating semantic type information.
    """
    rules = rules_blk.get("rules", []) or []

    s = 0.0
    dbg: Dict[str, float] = {}

    prop_type_map = _get_prop_type_map(props_blk)
    cap_vars = _capacity_vars_for_agent(props_blk)
    price_vars = _price_vars_for_agent(props_blk)
    cost_vars = _cost_vars_for_agent(props_blk)

    for r in rules:
        w = r.get("write")
        expr = (r.get("expr") or "").strip()
        if not isinstance(w, str) or not expr:
            continue

        rs = 0.0
        w_type = prop_type_map.get(w)  # Semantic type of the output variable

        # basic: has a non-empty expression
        rs += 0.5

        vars_used = set(_safe_vars(expr))

        # depends on its own previous value
        if w in vars_used:
            rs += 0.2

        # ---------------- NEW: Semantic Type Enforcement and Reward ----------------

        # DiscreteVolume/Stock Rules: check for non-negativity and flow-dependency
        if w_type == "DiscreteVolume" or _is_stock_like(w):
            if "ArrivingOrders" in vars_used and "DownstreamDemand" in vars_used:
                rs += 0.4  # Reward for using both flow variables
            if "max" in expr and "0" in expr:
                rs += 0.2  # Non-negativity constraint awareness
            if "min" in expr and any(cv in vars_used for cv in cap_vars):
                rs += 0.2  # Capacity constraint awareness

        # MonetaryValue Rules: check for revenue/cost logic
        if w_type == "MonetaryValue":
            uses_price = any(pv in vars_used for pv in price_vars)
            uses_cost = any(cv in vars_used for cv in cost_vars)
            uses_flow = ("ArrivingOrders" in vars_used) or ("DownstreamDemand" in vars_used)

            if uses_price and uses_flow:
                rs += 0.4  # Price * Quantity (Revenue/Cost calculation)

            # Penalize simple addition/subtraction without flow context
            if not uses_flow and not uses_price and not uses_cost:
                rs -= 0.3

            # Penalize operations that break dimensional consistency (e.g., multiplying two MonetaryValue)
            # This is complex to check fully, but we can do a simple check:
            if re.search(r"(\*|\/)\s*(" + "|".join(price_vars + cost_vars) + ")", expr):
                # Simple heuristic: If Price or Cost is involved in multiplication, assume its with quantity (Good)
                rs += 0.1
            elif re.search(r"(\*|\/)\s*([A-Z][a-z_]+)", expr) and not uses_flow:
                # If multiplication/division involves non-flow/non-price/cost terms (Ambiguous/Bad)
                # We skip penalizing here, as it's too complex without full type resolution.
                pass

        # Price-like Rules: check link to cost (Markup logic)
        if _is_price_like(w) or (w_type == "MonetaryValue" and "Price" in w):
            if uses_cost:
                rs += 0.5  # Price often depends on Cost (markup)

        # ---------------- END NEW ----------------

        dbg[w] = rs
        s += rs

    return s, dbg


# ----------------- Public API -----------------


def score_joint_sample(
        prop_packet: Dict[str, Any],
        rules_packet: Dict[str, Any],
) -> Tuple[float, Dict[str, Any]]:
    """
    Score a joint (properties + relations + rules) proposal
    for a small set of agents (typically 3 in a subgraph).

    Returns:
        total_score (float)
        debug_info  (dict)  # per-agent breakdown
    """
    neighbors_all = _neighbors_from_topology()

    total = 0.0
    debug: Dict[str, Any] = {}

    # We iterate over agents present in prop_packet; rules_packet may have same keys.
    for agent, pblk in prop_packet.items():
        pblk = pblk or {}
        rblk = rules_packet.get(agent, {}) or {}

        # property coverage + category alignment + type alignment
        sp, dbg_p = _score_properties_for_agent(agent, pblk)

        # relation direction vs topology + flow type alignment
        neigh_meta = neighbors_all.get(agent, {"upstream": [], "downstream": []})
        sr, dbg_rel_topo = _score_relations_topology(agent, pblk, neigh_meta)

        # rule patterns + semantic type adherence
        srule, dbg_rules = _score_rules_for_agent(agent, pblk, rblk)

        s_agent = sp + sr + srule
        total += s_agent

        debug[agent] = {
            "props": dbg_p,
            "relations_topology": dbg_rel_topo,
            "rules": dbg_rules,
            "agent_total": s_agent,
        }

    return total, debug

