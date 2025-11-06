from typing import Dict, List, Tuple

# ---------- Ranking → pairwise wins (order only) ----------

def ranking_to_pairs(order_names: List[str]) -> Dict[Tuple[str, str], int]:
    """
    From total order [a,b,c] produce pairwise wins: (a>b)=1, (a>c)=1, (b>c)=1.
    """
    wins = {}
    for i in range(len(order_names)):
        for j in range(i + 1, len(order_names)):
            wins[(order_names[i], order_names[j])] = 1
    return wins


def update_pairwise_bag(bag_pairs: Dict[str, dict], ranking: Dict[str, List[Tuple[str, str]]]) -> None:
    """
    Accumulate pairwise wins per-agent using only the order of property NAMES.
    bag_pairs[agent][(x,y)] = times x>y
    """
    for agent, ordered in ranking.items():
        order_names = [name for name, _dir in ordered]
        wins = ranking_to_pairs(order_names)
        store = bag_pairs.setdefault(agent, {})
        for k, v in wins.items():
            store[k] = store.get(k, 0) + v


def score_by_pairwise_consistency(
    ranking: Dict[str, List[Tuple[str, str]]],
    bag_pairs: Dict[str, dict],
    sample_count: int
) -> float:
    """
    Score by pairwise ORDER consistency only (ignore directions).
    Each agreed pair contributes wins/(sample_count).
    """
    score = 0.0
    for agent, ordered in ranking.items():
        order_names = [name for name, _dir in ordered]
        wins = ranking_to_pairs(order_names)
        hist = bag_pairs.get(agent, {})
        for pair in wins.keys():
            score += hist.get(pair, 0) / max(1, sample_count)
    return score


# ---------- Sliding-window stability via Kendall-like agreement (order only) ----------

def kendall_like_tau(order_a: List[str], order_b: List[str]) -> float:
    """
    Simple Kendall-like agreement in [-1,1] using ORDER only.
    """
    pos_a = {k: i for i, k in enumerate(order_a)}
    pairs = 0
    agree = 0
    for i in range(len(order_b)):
        for j in range(i + 1, len(order_b)):
            x, y = order_b[i], order_b[j]
            if x not in pos_a or y not in pos_a:
                continue
            pairs += 1
            agree += 1 if pos_a[x] < pos_a[y] else -1
    if pairs == 0:
        return 1.0
    return agree / pairs


def window_converged_rank(history: List[Dict[str, List[Tuple[str, str]]]], window: int = 12, thresh: float = 0.9) -> bool:
    """
    Converged if average Kendall-like agreement across the last `window` samples >= thresh,
    computed per-agent on ORDER only.
    """
    if len(history) < window:
        return False
    recent = history[-window:]
    agents = list(recent[-1].keys())
    for a in agents:
        sims = []
        for i in range(1, len(recent)):
            oa = [n for n, _d in recent[i - 1][a]]
            ob = [n for n, _d in recent[i][a]]
            sims.append(kendall_like_tau(oa, ob))
        if (sum(sims) / len(sims)) < thresh:
            return False
    return True


# ---------- Aggregation (order by majority; direction by majority but NOT used in scoring) ----------

def aggregate_rankings_with_direction(tail: List[Dict[str, List[Tuple[str, str]]]]) -> Dict[str, List[dict]]:
    """
    Final ranking by Copeland-like majority for ORDER, plus separate majority vote on DIRECTION.
    Output:
      { "Agent": [ {"name": "...", "direction": "Max|Min"}, ... ] }
    Direction is aggregated but was NOT part of scoring or acceptance.
    """
    out: Dict[str, List[dict]] = {}
    if not tail:
        return out
    agents = list(tail[-1].keys())

    for a in agents:
        # universe of items (property names)
        items = [n for n, _d in tail[-1][a]]

        # Copeland-like order aggregation
        wins = {x: 0 for x in items}
        losses = {x: 0 for x in items}
        for r in tail:
            order = [n for n, _d in r[a]]
            pos = {k: i for i, k in enumerate(order)}
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    x, y = items[i], items[j]
                    if pos[x] < pos[y]:
                        wins[x] += 1
                        losses[y] += 1
                    else:
                        wins[y] += 1
                        losses[x] += 1
        score = {x: wins[x] - losses[x] for x in items}
        ordered_items = sorted(items, key=lambda z: (-score[z], z))

        # Direction majority (not used in scoring)
        dir_votes = {x: {"Max": 0, "Min": 0} for x in items}
        for r in tail:
            for name, direction in r[a]:
                if name in dir_votes:
                    if direction == "Max":
                        dir_votes[name]["Max"] += 1
                    else:
                        dir_votes[name]["Min"] += 1
        final = [{"name": name, "direction": ("Max" if dir_votes[name]["Max"] >= dir_votes[name]["Min"] else "Min")}
                 for name in ordered_items]
        out[a] = final

    return out


