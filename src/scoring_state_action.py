# -*- coding: utf-8 -*-
"""
Scoring and convergence checks for State/Action proposals.

- Frequency bags: accumulate how often each item appears (per agent)
- Scores: consistency-only soft frequency (like properties step)
- Window convergence: Jaccard on Self sets; Jaccard on action-type sequences
- Aggregation (STATE): majority over the last tail window
- Aggregation (ACTION): composite-confidence Top-K with canonical param normalization
    * confidence = w_freq * frequency + w_acc * acceptance + w_stab * stability
    * keep only top-K (default K=2) per agent
"""

import os
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

# ----------------- Aggregation (action-level canonicalization + Top-K) -----------------

# Canonical required/fixed params per action (explicit-only lists keep exact shapes).
REQUIRED_BY_ACTION = {
    "AllocateInventory": ["To", "Quantity", "Priority"],         # explicit-only
    "SupplierSelection": ["AddSupplierId", "RemoveSupplierId"],  # explicit-only
    "FulfillOrder":      ["To", "Quantity"],                     # DueDate optional
    "OrderPlacement":    ["To", "Quantity"],                     # Item optional
    "CapacityAllocation":["Period", "Quantity"],
    "SetPrice":          ["Item", "UnitCost"],
    "SetServiceLevel":   ["Priority"],
    "Produce":           ["Quantity"],
    "Procure":           ["From", "Quantity"]                    # adjust if you prefer Item+LeadTime
}

def _normalize_params_post_agg(action_type: str, counted_params: dict, min_count: int) -> list:
    """
    Normalize AFTER voting so explicit/required params are kept.
    For explicit-only actions we return the exact param list.
    For others we allow at most one optional param if heavily voted.
    """
    action = action_type or ""
    # explicit-only actions keep exact shapes
    if action == "AllocateInventory":
        return ["To", "Quantity", "Priority"]
    if action == "SupplierSelection":
        return ["AddSupplierId", "RemoveSupplierId"]

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

def _presence_vector_for_action(tail: List[Dict[str, Any]], agent: str, action_type: str) -> List[bool]:
    """
    Build a boolean presence vector across tail samples for (agent, action_type).
    """
    pres = []
    for r in tail:
        pres.append(any(a.get("type") == action_type for a in r[agent]))
    return pres

def _presence_stability(pres_vec: List[bool]) -> float:
    """
    Stability in [0,1]: fraction of adjacent pairs where presence stays unchanged.
    If tail length < 2, return 1.0 by definition (no variance).
    """
    if len(pres_vec) < 2:
        return 1.0
    same = 0
    for i in range(1, len(pres_vec)):
        if pres_vec[i] == pres_vec[i - 1]:
            same += 1
    return same / max(1, len(pres_vec) - 1)

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

def aggregate_actions(
    tail: List[Dict[str, Any]],
    min_count: int = 4,
    top_k: int = None,
    accepted_flags_tail: List[bool] = None,
) -> Dict[str, Any]:
    """
    Aggregate actions with a composite confidence and keep only top-K per agent.

    For each agent and action type t in the tail window:
        freq = count_t / tail_len
        acc  = accepted_occurrences_t / max(1, count_t)
        stab = presence stability across tail

        confidence = w_freq * freq + w_acc * acc + w_stab * stab

    Weights can be tuned via env vars:
        CONF_W_FREQ   (default 0.5)
        CONF_W_ACCEPT (default 0.3)
        CONF_W_STAB   (default 0.2)
        ACTION_TOP_K  (default 2 if top_k is None)

    Canonical params are normalized AFTER voting (explicit/required preserved).
    Fallback: if nothing qualifies, keep one conservative default.
    """
    # read weights from env with sensible defaults
    w_freq = float(os.getenv("CONF_W_FREQ", "0.5"))
    w_acc  = float(os.getenv("CONF_W_ACCEPT", "0.3"))
    w_stab = float(os.getenv("CONF_W_STAB", "0.2"))
    if top_k is None:
        top_k = int(os.getenv("ACTION_TOP_K", "2"))

    out = {}
    agents = list(tail[-1].keys())
    tail_len = max(1, len(tail))

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

        candidates = []
        for t, c in type_bag.items():
            if c < min_count:
                continue

            counted = dict(param_bag[t])
            canon_params = _normalize_params_post_agg(t, counted, min_count)

            # frequency (coverage over tail)
            freq = c / tail_len

            # acceptance rate: among tail samples where this action is present,
            # how many come from "accepted" iterations
            if accepted_flags_tail is not None and len(accepted_flags_tail) == tail_len:
                accepted_present = 0
                for i, r in enumerate(tail):
                    if any(a.get("type") == t for a in r[ag]) and accepted_flags_tail[i]:
                        accepted_present += 1
                acc = accepted_present / max(1, c)
            else:
                acc = 0.0  # if not provided, be conservative

            # stability across tail window
            pres_vec = _presence_vector_for_action(tail, ag, t)
            stab = _presence_stability(pres_vec)

            conf = w_freq * freq + w_acc * acc + w_stab * stab

            candidates.append({
                "type": t,
                "params": canon_params,
                "confidence": conf
            })

        # sort by confidence desc, tie-break by type name
        candidates.sort(key=lambda x: (-x["confidence"], x["type"]))

        # keep only top-K
        k = max(1, top_k)
        final = [{"type": x["type"], "params": x["params"]} for x in candidates[:k]]

        # fallback if empty
        if not final:
            final = [{"type": "OrderPlacement", "params": ["To", "Quantity"]}]

        out[ag] = final
    return out

