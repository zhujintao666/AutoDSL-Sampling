import os
import json
import math
import random
from collections import defaultdict
from tqdm import tqdm

from src.proposals_priority import llm_rank_priorities_with_direction
from src.scoring_priority import (
    update_pairwise_bag,
    score_by_pairwise_consistency,
    window_converged_rank,
    aggregate_rankings_with_direction
)


def run_priority_mcmc(
    n_rounds: int = 60,
    beta: float = 1.2,
    warmup: int = 8,
    tail_pct: float = 0.2
):
    """
    LLM proposes a ranking+direction for each agent ->
    SCORE by pairwise ORDER consistency ONLY (direction is ignored here) ->
    Metropolis-Hastings accept/reject ->
    update pairwise bag after warmup ->
    stop on window stability (order only) ->
    aggregate the LAST tail_pct of samples:
      - ORDER: Copeland-like majority
      - DIRECTION: simple per-item majority
    """
    bag_pairs = defaultdict(dict)   # order-only history
    samples = []

    current = llm_rank_priorities_with_direction()
    samples.append(current)
    update_pairwise_bag(bag_pairs, current)
    curr_score = score_by_pairwise_consistency(current, bag_pairs, len(samples))

    for _ in tqdm(range(1, n_rounds + 1), desc="Priority MCMC"):
        proposal = llm_rank_priorities_with_direction()
        prop_score = score_by_pairwise_consistency(proposal, bag_pairs, len(samples))

        acc_ratio = min(1.0, math.exp(beta * (prop_score - curr_score)))
        if random.random() < acc_ratio:
            current, curr_score = proposal, prop_score

        samples.append(current)

        if len(samples) > warmup:
            update_pairwise_bag(bag_pairs, current)

        if window_converged_rank(samples, window=14, thresh=0.92):
            break

    # Aggregate last X% of samples (default 20%)
    tail_len = max(1, math.ceil(tail_pct * len(samples)))
    tail = samples[-tail_len:]
    final = aggregate_rankings_with_direction(tail)
    return final, samples


if __name__ == "__main__":
    outputs_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    final, history = run_priority_mcmc(tail_pct=0.2)  # change to 0.1 for last 10%
    out_path = os.path.join(outputs_dir, "final_priority.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print(f"Wrote: {out_path}")
