# -*- coding: utf-8 -*-
"""
Scoring and convergence checks for State/Action proposals.

- Frequency bags: accumulate how often each item appears (per agent)
- Scores: consistency-only soft frequency (like properties step)
- Window convergence: Jaccard on Self sets; Jaccard on action-type sequences
- Aggregation: majority over the last tail window (configurable by caller)
- NEW: Action aggregation now performs per-action canonical param normalization
       so required/explicit params are never lost by frequency voting.
"""
from collections import Counter, defaultdict
from typing import Dict, Any, List

# ----------------- Helpers -----------------
def jaccard_set(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    return len(A & B) / max(1, len(A | B))

# ----------------- Frequency bags -----------------
def update_state_bag(bag: Dict[str, Dict[str, Counter]], spec: Dict[str, Any]) -> None:
    """
    bag[agent]['Self'][prop] += 1
    bag[agent]['Neighbors'][prop] += 1
    bag[agent]['Relations'][name] += 1
    """
    for ag, s in spec.items():
        b = bag.setdefault(ag, {"Self": Counter(), "Neighbors": Counter(), "Relations": Counter()})
        for k in ("Self", "Neighbors", "Relations"):
            b[k].update(s.get(k, []))

def update_action_bag(bag: Dict[str, Counter], acts: Dict[str, Any]) -> None:
    """
    bag[agent][action_type] += 1
    """
    for ag, lst in acts.items():
        B = bag.setdefault(ag, Counter())
        for a in lst:
            t = a.get("type")
            if t:
                B[t] += 1

def update_action_param_bag(param_bag: Dict[str, Dict[str, Counter]], acts: Dict[str, Any]) -> None:
    """
    param_bag[agent][action_type][param_name] += 1
    """
    for ag, lst in acts.items():
        P = param_bag.setdefault(ag, {})
        for a in lst:
            t = a.get("type")
            PB = P.setdefault(t, Counter())
            PB.update(a.get("params", []))

# ----------------- Scores (consistency-only) -----------------
def score_state_spec(spec: Dict[str, Any], bag: Dict[str, Dict[str, Counter]], sample_count: int) -> float:
    score = 0.0
    for ag, s in spec.items():
        B = bag.get(ag, {"Self": Counter(), "Neighbors": Counter(), "Relations": Counter()})
        for k in ("Self", "Neighbors", "Relations"):
            for x in s.get(k, []):
                score += B[k][x] / max(1, sample_count)
        if s.get("Self"):
            score += 0.05  # mild validity nudge
        if s.get("Relations"):
            score += 0.02
    return score

def score_action_spec(acts: Dict[str, Any],
                      bag: Dict[str, Counter],
                      param_bag: Dict[str, Dict[str, Counter]],
                      sample_count: int) -> float:
    score = 0.0
    for ag, lst in acts.items():
        B = bag.get(ag, Counter())
        PB = param_bag.get(ag, {})
        for a in lst:
            t = a.get("type")
            score += B[t] / max(1, sample_count)
            for p in a.get("params", []):
                score += PB.get(t, Counter())[p] / max(1, sample_count)
        if lst:
            score += 0.05
    return score

# ----------------- Window convergence -----------------
def window_converged_states(history: List[Dict[str, Any]], window: int = 12, thresh: float = 0.9) -> bool:
    if len(history) < window:
        return False
    recent = history[-window:]
    agents = list(recent[-1].keys())
    for ag in agents:
        sims = []
        for i in range(1, len(recent)):
            a, b = recent[i - 1][ag], recent[i][ag]
            sims.append(jaccard_set(a.get("Self", []), b.get("Self", [])))
        if (sum(sims) / len(sims)) < thresh:
            return False
    return True

def window_converged_actions(history: List[Dict[str, Any]], window: int = 12, thresh: float = 0.9) -> bool:
    if len(history) < window:
        return False
    recent = history[-window:]
    agents = list(recent[-1].keys())
    for ag in agents:
        seqs = [[a["type"] for a in r[ag]] for r in recent]
        sims = []
        for i in range(1, len(seqs)):
            sims.append(jaccard_set(seqs[i - 1], seqs[i]))
        if (sum(sims) / len(sims)) < thresh:
            return False
    return True

# ----------------- Aggregation (action-level canonicalization) -----------------

# Canonical required/fixed params per action (used to normalize after voting).
REQUIRED_BY_ACTION = {
    "AllocateInventory": ["To", "Quantity", "Priority"],           # explicit-only
    "SupplierSelection": ["AddSupplierId", "RemoveSupplierId"],    # explicit-only
    "FulfillOrder":      ["To", "Quantity"],                       # DueDate optional
                                               # e.g., consider adding "DueDate" if heavily voted
    "OrderPlacement": ["To", "Quantity"],                       # Item optional
                                               # e.g., consider adding "Item" if heavily voted
    "CapacityAllocation":["Period", "Quantity"],
    "SetPrice":          ["Item", "UnitCost"],
    "SetServiceLevel":   ["Priority"],
    "Produce":           ["Quantity"],
    "Procure":           ["From", "Quantity"]                      # adjust if you prefer Item+LeadTime
}


def _normalize_params_post_agg(action_type: str, counted_params: dict, min_count: int) -> list:
    """
    Normalize after voting so explicit/required params are kept.
    counted_params: Counter-like dict of param->count within tail window.
    """
    action = action_type or ""
    req = REQUIRED_BY_ACTION.get(action, [])
    out = list(req)

    # Optional additions (thresholded) with max length <= 3
    if action == "FulfillOrder":
        if counted_params.get("DueDate", 0) >= min_count and "DueDate" not in out:
            out = (out + ["DueDate"])[:3]

    if action == "OrderPlacement":
        if counted_params.get("Item", 0) >= min_count and "Item" not in out:
            out = (out + ["Item"])[:3]

    return out[:3]

def aggregate_states(tail: List[Dict[str, Any]], min_count: int = 4) -> Dict[str, Any]:
    out = {}
    agents = list(tail[-1].keys())
    for ag in agents:
        bag_self, bag_nei, bag_rel = Counter(), Counter(), Counter()
        for s in tail:
            bag_self.update(s[ag].get("Self", []))
            bag_nei.update(s[ag].get("Neighbors", []))
            bag_rel.update(s[ag].get("Relations", []))
        out[ag] = {
            "Self": sorted([k for k, c in bag_self.items() if c >= min_count]),
            "Neighbors": sorted([k for k, c in bag_nei.items() if c >= min_count]),
            "Relations": sorted([k for k, c in bag_rel.items() if c >= min_count]),
        }
    return out

def aggregate_actions(tail: List[Dict[str, Any]], min_count: int = 4) -> Dict[str, Any]:
    """
    Majority voting by type + canonical param normalization.
    Ensures explicit schema in final_action_space.json regardless of proposal drift.
    """
    out = {}
    agents = list(tail[-1].keys())
    for ag in agents:
        type_bag = Counter()
        param_bag = defaultdict(Counter)

        # collect votes
        for r in tail:
            for a in r[ag]:
                t = a.get("type")
                if not t:
                    continue
                type_bag[t] += 1
                param_bag[t].update(a.get("params", []))

        final = []
        for t, c in type_bag.items():
            if c >= min_count:
                counted = dict(param_bag[t])
                keep_params = _normalize_params_post_agg(t, counted, min_count)
                final.append({"type": t, "params": keep_params})

        if not final:
            final = [{"type": "OrderPlacement", "params": ["To", "Quantity"]}]

        out[ag] = sorted(final, key=lambda x: x["type"])
    return out

