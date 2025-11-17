# -*- coding: utf-8 -*-
"""
MCMC-like loops for State Space and Action Space.

State loop:
- Propose (via LLM) -> score by historical frequency consistency
- Metropolis-Hastings accept/reject
- Update frequency bags after warmup
- Stop when sliding-window stability is met
- Aggregate by majority on the tail window (default last 20% of samples)

Action loop:
- Same MH loop as State
- During aggregation, keep only Top-K (default K=2) actions per agent
  ranked by a composite confidence:
      confidence = w_freq * frequency + w_acc * acceptance + w_stab * stability
  (weights are read by scoring module from env vars; defaults 0.5/0.3/0.2)
- Parameters are normalized canonically so required/explicit params are preserved
  (e.g., AllocateInventory -> ["To","Quantity","Priority"], SupplierSelection -> ["AddSupplierId","RemoveSupplierId"])

Adjust tail_pct to 0.10 if you want 10% tail voting.
"""
import os
import json
import math
import random
from collections import defaultdict, Counter
from tqdm import tqdm

from src.proposals_state_action import (
    llm_propose_state_space,
    llm_propose_action_space,
)
from src.scoring_state_action import (
    update_state_bag, update_action_bag, update_action_param_bag,
    score_state_spec, score_action_spec,
    window_converged_states, window_converged_actions,
    aggregate_states, aggregate_actions
)

# ----------------- State loop -----------------
def mh_loop_states(n_rounds=60, beta=1.2, warmup=8, tail_pct: float = 0.20, trace_path=None):
    """
    Run an MH-like loop over STATE specs.
    tail_pct: fraction of the most recent samples used for aggregation (e.g., 0.20 = last 20%)
    """
    bag = {}
    samples = []

    # init
    current = llm_propose_state_space()
    samples.append(current)
    update_state_bag(bag, current)
    curr_score = score_state_spec(current, bag, len(samples))

    # MH iterations
    for it in tqdm(range(1, n_rounds + 1), desc="State MCMC"):
        prop = llm_propose_state_space()
        prop_score = score_state_spec(prop, bag, len(samples))

        acc_ratio = min(1.0, math.exp(beta * (prop_score - curr_score)))
        accepted = random.random() < acc_ratio
        if accepted:
            current, curr_score = prop, prop_score

        samples.append(current)
        if len(samples) > warmup:
            update_state_bag(bag, current)

        if trace_path:
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            with open(trace_path, "a", encoding="utf-8") as f:
                rec = {
                    "iter": it,
                    "accepted": accepted,
                    "acc_ratio": acc_ratio,
                    "proposal_score": prop_score,
                    "current_score": curr_score,
                    "current_struct": current
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if window_converged_states(samples, window=12, thresh=0.92):
            break

    # Tail aggregation (majority)
    tail_len = max(1, math.ceil(tail_pct * len(samples)))
    tail = samples[-tail_len:]
    min_count = max(2, math.ceil(0.5 * tail_len))
    final = aggregate_states(tail, min_count=min_count)
    return final, samples

# ----------------- Action loop -----------------
def mh_loop_actions(n_rounds=60, beta=1.2, warmup=8, tail_pct: float = 0.20, trace_path=None):
    """
    Run an MH-like loop over ACTION specs.
    tail_pct: fraction of the most recent samples used for aggregation (e.g., 0.20 = last 20%)

    During aggregation, we keep only Top-K actions per agent by composite confidence:
        confidence = w_freq * frequency + w_acc * acceptance + w_stab * stability
    The scoring_state_action.aggregate_actions() reads weights from env vars:
        CONF_W_FREQ (default 0.5), CONF_W_ACCEPT (0.3), CONF_W_STAB (0.2), ACTION_TOP_K (2)
    """
    bag = defaultdict(Counter)   # action-type frequency
    param_bag = defaultdict(dict)  # per-action-type param frequency
    samples = []
    accepted_history = [True]   # mark init sample as accepted for alignment

    # init
    current = llm_propose_action_space()
    samples.append(current)
    update_action_bag(bag, current)
    update_action_param_bag(param_bag, current)
    curr_score = score_action_spec(current, bag, param_bag, len(samples))

    # MH iterations
    for it in tqdm(range(1, n_rounds + 1), desc="Action MCMC"):
        prop = llm_propose_action_space()
        prop_score = score_action_spec(prop, bag, param_bag, len(samples))

        acc_ratio = min(1.0, math.exp(beta * (prop_score - curr_score)))
        accepted = random.random() < acc_ratio
        if accepted:
            current, curr_score = prop, prop_score

        samples.append(current)
        accepted_history.append(accepted)

        if len(samples) > warmup:
            update_action_bag(bag, current)
            update_action_param_bag(param_bag, current)

        if trace_path:
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            with open(trace_path, "a", encoding="utf-8") as f:
                rec = {
                    "iter": it,
                    "accepted": accepted,
                    "acc_ratio": acc_ratio,
                    "proposal_score": prop_score,
                    "current_score": curr_score,
                    "current_struct": current
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if window_converged_actions(samples, window=12, thresh=0.92):
            break

    # Tail aggregation (Top-K by composite confidence)
    tail_len = max(1, math.ceil(tail_pct * len(samples)))
    tail = samples[-tail_len:]
    min_count = max(2, math.ceil(0.5 * tail_len))
    acc_tail = accepted_history[-tail_len:]  # align acceptance flags with tail

    final = aggregate_actions(
        tail,
        min_count=min_count,
        top_k=2,                        # keep only the top-2 actions per agent
        accepted_flags_tail=acc_tail,   # provide acceptance info for confidence
    )
    return final, samples

# ----------------- Entrypoint -----------------
if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # 1) State space MCMC
    state_final, _ = mh_loop_states(
        n_rounds=60, beta=1.2, warmup=8, tail_pct=0.20,
        trace_path=os.path.join(out_dir, "trace_states.jsonl")
    )
    with open(os.path.join(out_dir, "final_state_space.json"), "w", encoding="utf-8") as f:
        json.dump(state_final, f, ensure_ascii=False, indent=2)
    print("Wrote final_state_space.json")

    # 2) Action space MCMC
    act_final, _ = mh_loop_actions(
        n_rounds=60, beta=1.2, warmup=8, tail_pct=0.20,
        trace_path=os.path.join(out_dir, "trace_actions.jsonl")
    )
    with open(os.path.join(out_dir, "final_action_space.json"), "w", encoding="utf-8") as f:
        json.dump(act_final, f, ensure_ascii=False, indent=2)
    print("Wrote final_action_space.json")



