# -*- coding: utf-8 -*-
"""
Confidence-based pruning for property space (same tri-metric style as action pruning).

- Inputs:
    outputs/final_structure.json                  # current vocabulary per agent
    outputs/trace_YYYYmmdd_HHMMSS/iter_*.json     # structural MCMC trace (preferred)
  (fallback) outputs/trace_structure.jsonl        # optional JSONL trace with current/proposal structs

- Outputs:
    outputs/final_structure.pruned.json
    outputs/prune_properties_debug.json

- Policy:
    * Classify properties into minimal-commonality buckets by conservative stems (no renaming).
    * Drop the "Debt" bucket entirely (as per mentor direction).
    * For each agent and each bucket, compute a confidence score per property:
        conf = w_freq * freq_norm + w_acc * accept_rate + w_stab * window_stability
      where:
        - freq_norm    = (number of iterations property present in CURRENT struct) / N
        - accept_rate  = accepted_proposals_containing_prop / proposals_containing_prop (smoothed)
        - stability    = fraction of adjacent windows where presence does NOT flip (tail window)
    * Keep top-K per bucket (K configurable; default 2).
    * Clamp per-agent total <= 6 (configurable), and keep deterministic order (by score desc, then name).

Notes:
    - This script NEVER invents new tokens and NEVER renames; it uses only the tokens that
      already appear in final_structure.json.
    - If no structural trace is found, falls back to neutral scores with a warning (keeps <=K per bucket
      by original order). You should really run structural MCMC to get meaningful confidence.

Run:
    python -m src.prune_properties_conf

Optional ENV:
    PRUNE_TRACE   # set to a specific trace dir or "jsonl:<path-to-jsonl>"
"""

import json
import os
import re
from glob import glob
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

# ---------------- Config ----------------
# (1) Confidence weights (same flavor as action pruning)
W_FREQ  = 0.5
W_ACC   = 0.3
W_STAB  = 0.2

# (2) Keep at most K props per bucket; and per agent overall
TOP_K_PER_BUCKET = 2
MAX_PROPS_PER_AGENT = 6

# (3) Sliding window for stability (last W CURRENT snapshots)
STABILITY_WINDOW = 12

# (4) Minimal-commonality stems (lowercase). Conservative, no renaming; just bucket tags.
#     Debt/Backlog is intentionally omitted per mentor direction.
STEMS = {
    # Added "onhand" and "stored" to better catch OnHandInventory / StoredGoods
    "Stock":    ["inventory", "stock", "shelf", "rawmaterial", "finishedgood",
                 "warehouse", "onhand", "stored"],
    "Capacity": ["capacity", "throughput", "servicecapacity", "productioncapacity",
                 "delivery", "storage", "sales"],
    "Cost":     ["cost", "holdingcost", "productioncost", "logistic", "operatingcost",
                 "handlingcost", "manufacturingcost", "transportationcost"],
    "Assets":   ["cash", "asset", "equipment", "machinery", "fleet", "vehicle",
                 "warehousecount", "storecount", "machinerycount"],
    "Price":    ["price", "unitprice", "unitcost", "wholesale", "retail", "selling", "list"],
}
DROP_BUCKETS = {"Debt"}  # if a property matches "debt"/"payable"/"receivable", we skip it
DEBT_STEMS = ["debt", "payable", "receivable", "credit", "loan", "overdraft"]

# If you want to enforce floors for other buckets later, extend this set.
FLOOR_BUCKETS = {"Stock"}

# ---------------- IO helpers ----------------
HERE = os.path.dirname(__file__)
OUTDIR = os.path.join(HERE, "..", "outputs")

def _load_final_structure() -> Dict[str, List[str]]:
    path = os.path.join(OUTDIR, "final_structure.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _find_best_trace_dir() -> Tuple[str, str]:
    """
    Returns (mode, path):
      - ("dir", /abs/path/to/trace_dir) if we found a structural trace folder
      - ("jsonl", /abs/path/to/jsonl)   if a JSONL is configured
      - ("", "") if nothing
    Priority:
      1) ENV PRUNE_TRACE
      2) latest outputs/trace_* directory that looks like 'structure' (current_struct is dict->list[str])
      3) outputs/trace_structure.jsonl (optional)
    """
    env = os.getenv("PRUNE_TRACE", "").strip()
    if env:
        if env.lower().startswith("jsonl:"):
            return ("jsonl", env.split(":", 1)[1])
        if os.path.isdir(env):
            return ("dir", env)

    # scan dirs
    cand_dirs = sorted(glob(os.path.join(OUTDIR, "trace_*")), key=os.path.getmtime, reverse=True)
    for d in cand_dirs:
        iters = sorted(glob(os.path.join(d, "iter_*.json")))
        if not iters:
            continue
        try:
            with open(iters[0], "r", encoding="utf-8") as f:
                js = json.load(f)
            cur = js.get("current_struct", {})
            # Heuristic: structure trace -> value is list[str]; state trace -> value is dict with "Self"
            if cur and isinstance(next(iter(cur.values())), list):
                return ("dir", d)
        except Exception:
            pass

    jpath = os.path.join(OUTDIR, "trace_structure.jsonl")
    if os.path.isfile(jpath):
        return ("jsonl", jpath)

    return ("", "")

# ---------------- Trace reader ----------------
def _read_trace_dir(path: str):
    """
    Yield tuples per-iteration:
        (accepted: bool, proposal_struct: Dict[agent, List[str]], current_struct: Dict[agent, List[str]])
    """
    files = sorted(glob(os.path.join(path, "iter_*.json")))
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            js = json.load(f)
        yield (
            bool(js.get("accepted", False)),
            js.get("proposal_struct", {}) or {},
            js.get("current_struct", {}) or {}
        )

def _read_trace_jsonl(path: str):
    """
    Fallback JSONL format; each line should contain keys 'accepted', 'proposal_struct', 'current_struct'.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            js = json.loads(line)
            yield (
                bool(js.get("accepted", False)),
                js.get("proposal_struct", {}) or {},
                js.get("current_struct", {}) or {}
            )

# ---------------- Confidence metrics ----------------
def _presence_stability(history_sets: List[set], prop: str) -> float:
    """
    Fraction of adjacent pairs where presence(prop) does not flip, measured on the tail window.
    """
    if len(history_sets) < 2:
        return 1.0
    tail = history_sets[-STABILITY_WINDOW:] if len(history_sets) > STABILITY_WINDOW else history_sets
    same = 0
    total = 0
    prev = prop in tail[0]
    for s in tail[1:]:
        cur = prop in s
        same += 1 if (cur == prev) else 0
        total += 1
        prev = cur
    return same / max(1, total)

def _bucket_of(prop: str) -> str:
    """Map a property name into a minimal-commonality bucket by conservative stem matching."""
    p = prop.lower()
    # Drop debt-like items
    if any(stem in p for stem in DEBT_STEMS):
        return "Debt"
    for bkt, stems in STEMS.items():
        if any(stem in p for stem in stems):
            return bkt
    return ""  # unknown -> unbucketed (we won't keep it)

# ---------------- Main prune ----------------
def prune_properties():
    base = _load_final_structure()

    # Locate trace
    mode, tpath = _find_best_trace_dir()

    # Accumulators
    # freq[agent][prop] = #iterations where prop in CURRENT
    freq: Dict[str, Counter] = defaultdict(Counter)
    # prop_accept/prop_proposed track acceptance when proposal contained the prop
    prop_accept: Dict[str, Counter] = defaultdict(Counter)
    prop_proposed: Dict[str, Counter] = defaultdict(Counter)
    # history sets for stability
    history_sets: Dict[str, List[set]] = defaultdict(list)

    N = 0
    if mode == "dir":
        reader = _read_trace_dir(tpath)
    elif mode == "jsonl":
        reader = _read_trace_jsonl(tpath)
    else:
        reader = None

    if reader:
        for accepted, proposal, current in reader:
            N += 1
            # Current presence
            for ag, props in current.items():
                present = set(props)
                history_sets[ag].append(present)
                for p in props:
                    freq[ag][p] += 1
            # Acceptance signal conditioned on proposal
            for ag, props in proposal.items():
                for p in props:
                    prop_proposed[ag][p] += 1
                    if accepted:
                        prop_accept[ag][p] += 1

    # If no trace found, degrade gracefully
    no_trace = (N == 0)
    if no_trace:
        # Build one snapshot from final_structure (neutral scores)
        for ag, props in base.items():
            present = set(props)
            history_sets[ag].append(present)
            for p in props:
                freq[ag][p] += 1
                prop_proposed[ag][p] += 1
                prop_accept[ag][p] += 1
        N = 1

    # Compute scores and bucket
    debug = {}
    pruned = {}
    for ag, props in base.items():
        # Prepare presence list
        hist = history_sets.get(ag, [set()])
        # Compute metrics per prop
        scored_by_bucket: Dict[str, List[Tuple[str, float, Dict]]] = defaultdict(list)
        for p in props:
            bkt = _bucket_of(p)
            if (not bkt) or (bkt in DROP_BUCKETS):
                continue
            f = freq[ag][p] / max(1, N)
            proposed = prop_proposed[ag][p]
            acc_rate = (prop_accept[ag][p] + 1) / (proposed + 2)  # Laplace smoothing
            stab = _presence_stability(hist, p)
            conf = W_FREQ * f + W_ACC * acc_rate + W_STAB * stab
            scored_by_bucket[bkt].append((p, conf, {"freq": f, "acc": acc_rate, "stab": stab}))

        # Per-bucket top-K
        kept = []
        dbg_agent = {}
        for bkt, items in scored_by_bucket.items():
            items.sort(key=lambda x: (-x[1], x[0]))
            top = items[:TOP_K_PER_BUCKET]
            kept.extend([name for (name, _, _) in top])
            dbg_agent[bkt] = [{"name": name, "score": round(score, 4), **meta}
                              for (name, score, meta) in top]

        # -------- Floor #1 (post-topK): ensure at least one Stock if candidates exist --------
        if "Stock" in scored_by_bucket and "Stock" in FLOOR_BUCKETS:
            has_stock_after_topk = any((_bucket_of(x) == "Stock") for x in kept)
            if not has_stock_after_topk:
                top_stock_items = sorted(scored_by_bucket["Stock"], key=lambda x: (-x[1], x[0]))
                if top_stock_items:
                    kept.append(top_stock_items[0][0])
                    dbg_agent.setdefault("Stock_fallback", []).append(
                        {"name": top_stock_items[0][0],
                         "score": round(top_stock_items[0][1], 4),
                         "reason": "post-topk-floor"}
                    )
        # ------------------------------------------------------------------------------------

        # Clamp per-agent total using a deterministic merge of kept items
        flat = []
        for bkt, items in scored_by_bucket.items():
            flat.extend(items)
        flat.sort(key=lambda x: (-x[1], x[0]))

        dedup_seen, final_list = set(), []
        for name, score, _meta in flat:
            if name in dedup_seen:
                continue
            if name in kept:
                dedup_seen.add(name)
                final_list.append(name)

        pruned[ag] = final_list[:MAX_PROPS_PER_AGENT]
        debug[ag] = dbg_agent

        # -------- Floor #2 (post-clamp): still ensure at least one Stock in final list --------
        if "Stock" in scored_by_bucket and "Stock" in FLOOR_BUCKETS:
            has_stock_final = any(_bucket_of(p) == "Stock" for p in pruned[ag])
            if (not has_stock_final) and scored_by_bucket["Stock"]:
                top_stock_items = sorted(scored_by_bucket["Stock"], key=lambda x: (-x[1], x[0]))
                top_stock = top_stock_items[0][0]
                if pruned[ag]:
                    if top_stock not in pruned[ag]:
                        # Replace the last slot to keep MAX_PROPS_PER_AGENT cap
                        pruned[ag][-1] = top_stock
                else:
                    pruned[ag] = [top_stock]
                debug[ag].setdefault("Stock_fallback", []).append(
                    {"name": top_stock, "reason": "post-clamp-floor"}
                )
        # -------------------------------------------------------------------------------------

    # Write results
    os.makedirs(OUTDIR, exist_ok=True)
    out_main = os.path.join(OUTDIR, "final_structure.pruned.json")
    out_dbg  = os.path.join(OUTDIR, "prune_properties_debug.json")
    with open(out_main, "w", encoding="utf-8") as f:
        json.dump(pruned, f, ensure_ascii=False, indent=2)
    with open(out_dbg, "w", encoding="utf-8") as f:
        json.dump({"no_trace": no_trace, "trace": tpath, "debug": debug}, f, ensure_ascii=False, indent=2)

    print(f"Wrote {out_main} and {out_dbg}")
    if no_trace:
        print("WARNING: no structural trace found; scores are neutral. Consider running structural MCMC first.")
    else:
        print(f"Used trace source: {mode}:{tpath}")

if __name__ == "__main__":
    prune_properties()

