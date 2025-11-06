# scoring.py

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))

def structure_score(struct: dict, freq_bag: dict, sample_count: int) -> float:
    """
    Score = frequency consistency only.
    Remove any fixed-size preference so each agent can keep as many/few props
    as the data/process wants.
    """
    freq_score = 0.0
    for agent, props in struct.items():
        for p in props:
            freq_score += freq_bag[agent][p] / max(1, sample_count)
    return freq_score

def window_converged(history, window: int = 8, thresh: float = 0.9) -> bool:
    if len(history) < window:
        return False
    recent = history[-window:]
    agents = list(recent[-1].keys())
    for a in agents:
        sims = []
        for i in range(1, len(recent)):
            sims.append(jaccard(set(recent[i-1][a]), set(recent[i][a])))
        if (sum(sims) / len(sims)) < thresh:
            return False
    return True

