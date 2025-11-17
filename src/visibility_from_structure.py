# -*- coding: utf-8 -*-
"""
Build a conservative visibility SoT (public/private) directly from final_structure.json.

Heuristics (can be tuned via CLI):
  ALWAYS PUBLIC:
    - Tokens containing "Price" or "Capacity" (e.g., RetailPrice, ServiceCapacity)
    - Tokens containing "LeadTime"
  ALWAYS PRIVATE:
    - Tokens containing "Cost"
    - Inventory family: Inventory/OnHand/FinishedGoods/RawMaterials/ShelfInventory
    - Asset family: Cash/Receivables/Payables/WorkInProgress
    - Debt-like: Backlog, Stockout (policy can override)
  UNKNOWN -> PRIVATE (conservative fallback)

CLI options:
  --structure <path>  (default: outputs/final_structure.json)
  --out <path>        (default: configs/visibility.json)
  --stockout {public,private,keep}  (default: keep)
  --backlog  {public,private,keep}  (default: keep)
"""

import os
import json
import argparse
from typing import Dict, List


# ---------------------------- Load / Save ----------------------------

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------------------- Heuristics ----------------------------

INVENTORY_TOKENS = {
    "Inventory", "OnHand", "FinishedGoods", "RawMaterials", "ShelfInventory"
}
ASSET_TOKENS = {
    "Cash", "Receivables", "Payables", "WorkInProgress"
}

def is_public_token(tok: str) -> bool:
    """Public by business convention: pricing/offer + capacity + lead time."""
    return ("Price" in tok) or ("Capacity" in tok) or ("LeadTime" in tok)

def is_private_token(tok: str) -> bool:
    """Private by convention: costs + inventories + assets (finance-like)."""
    return (
        ("Cost" in tok)
        or (tok in INVENTORY_TOKENS)
        or (tok in ASSET_TOKENS)
    )


def apply_policy(name: str, policy: str, pub: set, prv: set) -> None:
    """
    Enforce a unification policy for specific tokens (Stockout/Backlog).
    policy: 'public' | 'private' | 'keep'
    """
    if policy == "public":
        pub.add(name); prv.discard(name)
    elif policy == "private":
        prv.add(name); pub.discard(name)
    # keep -> do nothing


# ---------------------------- Core ----------------------------

def build_visibility_from_structure(
    struct: Dict[str, List[str]],
    stockout_policy: str = "keep",
    backlog_policy: str = "keep",
) -> Dict[str, Dict[str, List[str]]]:
    """
    Given final_structure.json (authoritative property names per agent),
    produce a conservative visibility split for each agent.
    """
    out: Dict[str, Dict[str, List[str]]] = {}

    for agent, props in struct.items():
        pub, prv = set(), set()

        for p in props:
            if is_public_token(p):
                pub.add(p)
            elif is_private_token(p):
                prv.add(p)
            else:
                # unknown -> conservative private
                prv.add(p)

        # Optional unify policies for ambiguous debt-like tokens
        if "Stockout" in props:
            apply_policy("Stockout", stockout_policy, pub, prv)
        if "Backlog" in props:
            apply_policy("Backlog", backlog_policy, pub, prv)

        out[agent] = {
            "public": sorted(pub),
            "private": sorted(prv),
        }

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure", default="outputs/final_structure.json")
    parser.add_argument("--out", default="configs/visibility.json")
    parser.add_argument("--stockout", choices=["public", "private", "keep"], default="keep")
    parser.add_argument("--backlog", choices=["public", "private", "keep"], default="keep")
    args = parser.parse_args()

    if not os.path.exists(args.structure):
        raise FileNotFoundError(args.structure)

    struct = load_json(args.structure)
    vis = build_visibility_from_structure(
        struct,
        stockout_policy=args.stockout,
        backlog_policy=args.backlog,
    )
    save_json(args.out, vis)
    print(f"Wrote SoT visibility: {args.out}")


if __name__ == "__main__":
    main()
