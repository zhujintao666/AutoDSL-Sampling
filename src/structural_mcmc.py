import json
import math
import os
import random
from collections import defaultdict, Counter
from datetime import datetime
from tqdm import tqdm

from src.proposals import llm_generate_structure
from src.scoring import structure_score, window_converged

def aggregate(last_structs, min_count: int = 4) -> dict:
    """
    Aggregate the last few accepted structures:
    pick properties that appear at least `min_count` times (majority vote).
    """
    agents = list(last_structs[-1].keys())
    out = {}
    for a in agents:
        bag = Counter()
        for s in last_structs:
            bag.update(s[a])
        out[a] = sorted([p for p, c in bag.items() if c >= min_count])
    return out

def run_structural_mcmc(n_rounds: int = 40, beta: float = 1.4, warmup: int = 10):
    """
    Structural 'MCMC-like' loop (unchanged logic):
      - Propose a structure via LLM
      - Score against frequency bag (consistency only)
      - Metropolis-Hastings accept/reject
      - Update frequency bag after warmup
      - Stop when sliding-window Jaccard converges

    NEW: trace every iteration into outputs/trace_YYYYmmdd_HHMMSS/*.json
    """
    samples = []
    freq_bag = defaultdict(Counter)

    # Create a dedicated trace folder per run
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", f"trace_{ts}")
    os.makedirs(trace_dir, exist_ok=True)

    # ----- init -----
    current = llm_generate_structure()
    samples.append(current)
    for a, props in current.items():
        freq_bag[a].update(props)
    curr_score = structure_score(current, freq_bag, len(samples))

    # write iteration 0 snapshot
    with open(os.path.join(trace_dir, "iter_0000.json"), "w", encoding="utf-8") as f:
        json.dump({
            "iter": 0,
            "accepted": True,
            "acc_ratio": 1.0,
            "proposal_score": curr_score,
            "current_score": curr_score,
            "proposal_struct": current,   # same as current on init
            "current_struct": current
        }, f, ensure_ascii=False, indent=2)

    # ----- main loop -----
    for t in tqdm(range(1, n_rounds + 1), desc="Structural MCMC"):
        proposal = llm_generate_structure()
        prop_score = structure_score(proposal, freq_bag, len(samples))

        acc_ratio = min(1.0, math.exp(beta * (prop_score - curr_score)))
        accepted = random.random() < acc_ratio
        if accepted:
            current, curr_score = proposal, prop_score

        samples.append(current)

        # update frequency bag only after warmup (same as before)
        if len(samples) > warmup:
            for a, props in current.items():
                freq_bag[a].update(props)

        # trace this iteration (proposal + accept/reject + resulting current)
        trace_path = os.path.join(trace_dir, f"iter_{t:04d}.json")
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump({
                "iter": t,
                "accepted": accepted,
                "acc_ratio": acc_ratio,
                "proposal_score": prop_score,
                "current_score": curr_score,
                "proposal_struct": proposal,
                "current_struct": current
            }, f, ensure_ascii=False, indent=2)

        # stop on window convergence (unchanged)
        if window_converged(samples, window=16, thresh=0.995):
            break

    # aggregate tail (unchanged)
    agg_window = 12
    tail = samples[-agg_window:] if len(samples) >= agg_window else samples
    final = aggregate(tail, min_count=max(4, len(tail) // 2))

    # write final
    outputs_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    out_path = os.path.join(outputs_dir, "final_structure.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    # write a frequency snapshot (optional but handy)
    freq_snapshot = {a: dict(freq_bag[a]) for a in freq_bag}
    with open(os.path.join(trace_dir, "freq_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(freq_snapshot, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {out_path}")
    print(f"Trace directory: {trace_dir}")
    return final, samples

if __name__ == "__main__":
    # Ensure outputs/ exists regardless of run location
    outputs_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    final, history = run_structural_mcmc()

