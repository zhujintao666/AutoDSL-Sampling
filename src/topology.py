import os, json
from typing import Dict, List, Set

def _read(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_network():
    """
    Load supply-chain topology. Fallback to a default 5-stage chain if config missing.
    Expected config format (example):
    {
      "agents": ["Supplier","Manufacturer","Distributor","Wholesaler","Retailer"],
      "edges": [{"from":"Supplier","to":"Manufacturer"}, ...]
    }
    """
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "configs", "network.json")
    if os.path.exists(path):
        return _read(path)
    # fallback default chain
    return {
        "agents": ["Supplier","Manufacturer","Distributor","Wholesaler","Retailer"],
        "edges": [
            {"from":"Supplier","to":"Manufacturer"},
            {"from":"Manufacturer","to":"Distributor"},
            {"from":"Distributor","to":"Wholesaler"},
            {"from":"Wholesaler","to":"Retailer"}
        ]
    }

def load_visibility():
    """
    Load visibility (public/private). Fallback: treat ALL properties as public to simplify sampling.
    Expected format:
    {
      "Supplier": {"public": ["Price","ProductionCapacity"], "private": ["Inventory","Cash"]},
      ...
    }
    """
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "configs", "visibility.json")
    if os.path.exists(path):
        return _read(path)
    return {}  # fallback: empty => treat all as public

def one_hop_neighbors(agent: str, edges: List[Dict[str, str]]) -> Set[str]:
    ups, downs = set(), set()
    for e in edges:
        if e["from"] == agent: ups.add(e["to"])
        if e["to"] == agent:   downs.add(e["from"])
    return ups | downs

def neighbor_public_whitelist(agent: str, props_by_agent: Dict[str, List[str]]) -> List[str]:
    """
    Allowed neighbor props for `agent` = union of PUBLIC props of its 1-hop neighbors,
    intersected with neighbors' own property lists. If visibility is missing, default to all.
    """
    net = load_network()
    vis = load_visibility()
    neighbors = one_hop_neighbors(agent, net["edges"])
    bag = []
    for nb in neighbors:
        own_list = props_by_agent.get(nb, [])
        if nb in vis and isinstance(vis.get(nb, {}).get("public"), list):
            public = set(vis[nb]["public"])
            cand = [p for p in own_list if p in public]
        else:
            cand = list(own_list)  # fallback: all visible
        bag.extend(cand)
    seen, out = set(), []
    for x in bag:
        if x not in seen:
            seen.add(x); out.append(x)
    return out
