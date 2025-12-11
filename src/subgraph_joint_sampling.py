# -*- coding: utf-8 -*-
"""
Subgraph-level joint sampling for properties + relations + deterministic rules.

Idea:
- Real supply-chain interactions are mostly local: each agent interacts with its direct
  upstream and downstream neighbors.
- Instead of sampling rules for each agent in isolation, we sample on 3-node subgraphs
  (Upstream -> Mid -> Downstream), then project the best templates back to concrete agents.
- Mid agents appear in multiple triples, making their properties/rules more robust.

This script:
- Loads final_property_definitions.json and final_transition_rules.json as a baseline.
- Loads the supply-chain topology from configs/network.json (or uses the default chain).
- Enumerates all length-2 paths (A -> B -> C) as subgraphs.
- For EACH subgraph, runs a small MH-like loop where an LLM jointly proposes:
    * property categories + English definitions for A/B/C (using existing names);
    * relation from/to definitions for A/B/C;
    * deterministic next-step update rules for all properties of A/B/C.
- Scores each triple-level candidate using score_joint_sample (topology-aware, category-aware).
- Keeps the best candidate per triple and merges them back into global
  property definitions and rules.

Outputs:
    outputs/final_property_definitions.subgraph.json
    outputs/final_transition_rules.subgraph.json
    outputs/subgraph_sampling_debug.json

Run:
    python -m src.subgraph_joint_sampling
"""

from __future__ import annotations

import os
import re
import json
import math
import random
from typing import Dict, Any, List, Tuple

from openai import OpenAI

from src.topology import load_network
from src.joint_props_rules_sampling import _process_candidate_for_agent
from src.joint_rules_scoring import score_joint_sample

# -------------------------- Paths --------------------------

HERE = os.path.dirname(__file__)
OUTDIR = os.path.join(HERE, "..", "outputs")

PATH_PROP_DEFS = os.path.join(OUTDIR, "final_property_definitions.json")
PATH_RULES = os.path.join(OUTDIR, "final_transition_rules.json")

PATH_OUT_PROP = os.path.join(OUTDIR, "final_property_definitions.subgraph.json")
PATH_OUT_RULES = os.path.join(OUTDIR, "final_transition_rules.subgraph.json")
PATH_OUT_DEBUG = os.path.join(OUTDIR, "subgraph_sampling_debug.json")

# --------------------- LLM / API helpers -------------------

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"
SYSTEM = (
    "You are a precise supply chain expert. You output ONLY valid JSON. "
    "No prose. No code fences. Deterministic arithmetic only."
)


def _get_api_key() -> str:
    key = os.getenv("DASHSCOPE_API_KEY")
    if key:
        return key
    # optional: configs/keys.json fallback
    try:
        with open(os.path.join(HERE, "..", "configs", "keys.json"), "r", encoding="utf-8") as f:
            j = json.load(f)
            return j.get("DASHSCOPE_API_KEY", "")
    except Exception:
        return ""


def _strip_code_fences(s: str) -> str:
    s = re.sub(r"```(?:json)?\s*", "", s, flags=re.I)
    s = s.replace("```", "")
    return s.strip()


def _first_complete_json_block(text: str) -> str:
    """Extract the first balanced {...} block as JSON."""
    s = text
    start = s.find("{")
    if start < 0:
        return ""
    stack: List[str] = []
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                stack.append("{")
            elif ch == "}":
                if stack:
                    stack.pop()
                    if not stack:
                        return s[start : i + 1]
    return ""


def _json_like_cleanup(s: str) -> str:
    """Best-effort cleanup before feeding into json.loads."""
    s = _strip_code_fences(s)
    # normalize smart quotes
    s = (
        s.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )
    # remove trailing commas before ] or }
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s.strip()


def _try_parse_json(text: str) -> Any:
    blob = _first_complete_json_block(text) or text
    blob = _json_like_cleanup(blob)
    try:
        return json.loads(blob)
    except Exception:
        # one more pass of trailing-comma cleanup
        blob2 = re.sub(r",\s*([}\]])", r"\1", blob)
        return json.loads(blob2)


def _call_llm(prompt: str) -> Dict[str, Any]:
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY (env or configs/keys.json)")

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=2200,
    )
    raw = resp.choices[0].message.content
    return _try_parse_json(raw)


# --------------------- Data + topology helpers ---------------------


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_baseline_defs_and_rules() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not os.path.exists(PATH_PROP_DEFS):
        raise FileNotFoundError(f"Baseline property definitions not found: {PATH_PROP_DEFS}")
    if not os.path.exists(PATH_RULES):
        raise FileNotFoundError(f"Baseline transition rules not found: {PATH_RULES}")
    prop_defs = _load_json(PATH_PROP_DEFS)
    rules = _load_json(PATH_RULES)
    return prop_defs, rules


def _extract_triples(net: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """Enumerate all length-2 paths A -> B -> C from the network topology."""
    agents = net.get("agents", [])
    edges = net.get("edges", [])

    succ = {a: [] for a in agents}
    for e in edges:
        src = e.get("from")
        dst = e.get("to")
        if src in succ and dst in agents:
            succ[src].append(dst)

    triples: List[Tuple[str, str, str]] = []
    for a in agents:
        for b in succ.get(a, []):
            for c in succ.get(b, []):
                triples.append((a, b, c))
    return triples


def _props_list_for_agent(prop_defs: Dict[str, Any], agent: str) -> List[str]:
    blk = prop_defs.get(agent, {})
    props = blk.get("properties", {}) or {}
    return list(props.keys())


def _relations_names_for_agent(prop_defs: Dict[str, Any], agent: str) -> List[str]:
    blk = prop_defs.get(agent, {})
    rels = blk.get("relations", []) or []
    out: List[str] = []
    for r in rels:
        name = (r or {}).get("name")
        if isinstance(name, str):
            out.append(name)
    # keep unique & stable
    seen = set()
    ordered: List[str] = []
    for n in out:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def _summarize_agent_for_prompt(prop_defs: Dict[str, Any], rules: Dict[str, Any], agent: str) -> str:
    """Build a compact summary of baseline properties + rules for one agent."""
    blk = prop_defs.get(agent, {})
    p_dict = blk.get("properties", {}) or {}
    r_blk = rules.get(agent, {}) or {}
    r_list = r_blk.get("rules", []) or []

    lines: List[str] = []
    lines.append(f'Agent "{agent}" baseline properties:')
    for name, meta in p_dict.items():
        cat = (meta or {}).get("category", "Other")
        dfn = (meta or {}).get("definition", "")
        lines.append(f'  - {name} (category: {cat}): {dfn}')

    lines.append(f'Agent "{agent}" baseline rules:')
    for r in r_list:
        w = r.get("write")
        e = r.get("expr")
        if not w or not e:
            continue
        lines.append(f'  - {w}_next = {e}')

    return "\n".join(lines)


# ------------------- Prompt builder for triple -------------------


def _build_triple_prompt(
    upstream: str,
    mid: str,
    downstream: str,
    prop_defs: Dict[str, Any],
    rules: Dict[str, Any],
) -> str:
    """Build a joint prompt for one triple (Upstream -> Mid -> Downstream)."""
    up_summary = _summarize_agent_for_prompt(prop_defs, rules, upstream)
    mid_summary = _summarize_agent_for_prompt(prop_defs, rules, mid)
    down_summary = _summarize_agent_for_prompt(prop_defs, rules, downstream)

    up_props = _props_list_for_agent(prop_defs, upstream)
    mid_props = _props_list_for_agent(prop_defs, mid)
    down_props = _props_list_for_agent(prop_defs, downstream)

    lines: List[str] = []
    lines.append("You are designing state properties, relations, and deterministic update rules")
    lines.append("for a 3-agent SUPPLY CHAIN SUBGRAPH with fixed agent names and property names.")
    lines.append("")
    lines.append("Subgraph structure:")
    lines.append(f'  Upstream agent:   "{upstream}"')
    lines.append(f'  Mid agent:        "{mid}"')
    lines.append(f'  Downstream agent: "{downstream}"')
    lines.append("")
    lines.append("Each agent already has a list of Self properties (state) and baseline rules.")
    lines.append("You MUST:")
    lines.append("  - Use ONLY these existing property names for each agent; do not invent new names.")
    lines.append("  - For each agent, for every property in its list, output:")
    lines.append('      * a category in ["Stock","Capacity","Cost","Asset","Price","Other"];')
    lines.append("      * a clear one-sentence English definition (you may refine the baseline text);")
    lines.append("      * exactly ONE deterministic update rule for that property.")
    lines.append("  - For each agent, if it uses relational properties like ArrivingOrders or DownstreamDemand,")
    lines.append("    define their FROM/TO direction consistent with the subgraph structure:")
    lines.append(f'      * Shipments from "{upstream}" to "{mid}" then from "{mid}" to "{downstream}".')
    lines.append("      * Typically: ArrivingOrders at an agent comes from its direct upstream neighbor(s);")
    lines.append("        DownstreamDemand at an agent comes from its direct downstream neighbor(s).")
    lines.append("")
    lines.append("Rules MUST:")
    lines.append("  - Be deterministic and use only properties from the same agent + relational properties visible there;")
    lines.append("  - Use only +, -, *, /, numeric constants, min(), max(), and parentheses;")
    lines.append("  - For STOCK-like properties (inventory, stock, on-hand, etc.), follow a conservation pattern like:")
    lines.append("      Inventory_next = min(CapacityVar, max(0, Inventory + ArrivingOrders - DownstreamDemand))")
    lines.append("    If there is no capacity property, you may omit the min(capacity, ...) clamp.")
    lines.append("")
    lines.append("Baseline (for context, you may refine but not break semantics):")
    lines.append("")
    lines.append(up_summary)
    lines.append("")
    lines.append(mid_summary)
    lines.append("")
    lines.append(down_summary)
    lines.append("")
    lines.append("Now return ONLY ONE JSON object with EXACTLY this schema:")
    lines.append("{")
    lines.append('  "Upstream": {')
    lines.append('    "agent": "' + upstream + '",')
    lines.append('    "properties": [')
    lines.append('      {"name": "...", "category": "Stock", "definition": "..."}')
    lines.append("    ],")
    lines.append('    "relations": [')
    lines.append('      {"name": "ArrivingOrders", "from": ["..."], "to": "' + upstream + '", "definition": "..."}')
    lines.append("    ],")
    lines.append('    "rules": [')
    lines.append('      {"write": "...", "expr": "..."}')
    lines.append("    ]")
    lines.append("  },")
    lines.append('  "Mid": {')
    lines.append('    "agent": "' + mid + '",')
    lines.append('    "properties": [ {"name": "...", "category": "Stock", "definition": "..."} ],')
    lines.append('    "relations": [ {"name": "DownstreamDemand", "from": ["..."], "to": "' + mid + '", "definition": "..."} ],')
    lines.append('    "rules": [ {"write": "...", "expr": "..."} ]')
    lines.append("  },")
    lines.append('  "Downstream": {')
    lines.append('    "agent": "' + downstream + '",')
    lines.append('    "properties": [ {"name": "...", "category": "Stock", "definition": "..."} ],')
    lines.append('    "relations": [ {"name": "ArrivingOrders", "from": ["..."], "to": "' + downstream + '", "definition": "..."} ],')
    lines.append('    "rules": [ {"write": "...", "expr": "..."} ]')
    lines.append("  }")
    lines.append("}")
    lines.append("")
    lines.append("IMPORTANT HARD CONSTRAINTS:")
    lines.append("  - For each role (Upstream/Mid/Downstream), for each property listed below,")
    lines.append("    you MUST output exactly one entry in the corresponding 'properties' array")
    lines.append("    and exactly one rule in the 'rules' array:")
    lines.append(f"    Upstream properties:   {up_props}")
    lines.append(f"    Mid properties:        {mid_props}")
    lines.append(f"    Downstream properties: {down_props}")
    lines.append("  - Do NOT invent new property names.")
    lines.append("  - Do NOT change agent names.")
    lines.append("")
    lines.append("Return ONLY this JSON object. No comments, no code fences.")
    return "\n".join(lines)


# ------------------- MCMC over one triple -------------------


def _run_triple_mcmc(
    upstream: str,
    mid: str,
    downstream: str,
    prop_defs: Dict[str, Any],
    rules: Dict[str, Any],
    n_rounds: int = 6,
    beta: float = 1.0,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    MH-like loop for a single triple (upstream -> mid -> downstream).

    Returns:
        best_props: {agent_name: {"properties": {...}, "relations": [...]}}
        best_rules: {agent_name: {"rules": [...]}}
        debug:      per-iteration scores and accept flags
    """
    debug: Dict[str, Any] = {
        "triple": [upstream, mid, downstream],
        "mcmc": [],
    }

    up_props = _props_list_for_agent(prop_defs, upstream)
    mid_props = _props_list_for_agent(prop_defs, mid)
    down_props = _props_list_for_agent(prop_defs, downstream)

    up_rels_allowed = _relations_names_for_agent(prop_defs, upstream)
    mid_rels_allowed = _relations_names_for_agent(prop_defs, mid)
    down_rels_allowed = _relations_names_for_agent(prop_defs, downstream)

    prompt = _build_triple_prompt(upstream, mid, downstream, prop_defs, rules)

    def _process_role(cand_raw: Dict[str, Any], role_key: str, agent_name: str,
                      allowed_props: List[str], allowed_rel_names: List[str]):
        block = cand_raw.get(role_key, {}) or {}
        block["agent"] = agent_name  # force correct name
        return _process_candidate_for_agent(
            agent_name,
            allowed_props,
            allowed_rel_names,
            block,
        )

    # ----- initial proposal -----
    cand_raw = _call_llm(prompt)

    up_defs, up_rels, up_rules, up_patch = _process_role(cand_raw, "Upstream", upstream, up_props, up_rels_allowed)
    mid_defs, mid_rels, mid_rules, mid_patch = _process_role(cand_raw, "Mid", mid, mid_props, mid_rels_allowed)
    down_defs, down_rels, down_rules, down_patch = _process_role(cand_raw, "Downstream", downstream, down_props, down_rels_allowed)

    props_packet = {
        upstream: {"properties": up_defs, "relations": up_rels},
        mid: {"properties": mid_defs, "relations": mid_rels},
        downstream: {"properties": down_defs, "relations": down_rels},
    }
    rules_packet = {
        upstream: {"rules": up_rules},
        mid: {"rules": mid_rules},
        downstream: {"rules": down_rules},
    }

    curr_score, _dbg0 = score_joint_sample(props_packet, rules_packet)

    best_props = props_packet
    best_rules = rules_packet
    best_score = curr_score

    debug["mcmc"].append(
        {"iter": 0, "score": curr_score, "accepted": True}
    )

    # ----- MH iterations -----
    for it in range(1, n_rounds):
        cand_raw = _call_llm(prompt)

        up_defs, up_rels, up_rules, up_patch = _process_role(cand_raw, "Upstream", upstream, up_props, up_rels_allowed)
        mid_defs, mid_rels, mid_rules, mid_patch = _process_role(cand_raw, "Mid", mid, mid_props, mid_rels_allowed)
        down_defs, down_rels, down_rules, down_patch = _process_role(cand_raw, "Downstream", downstream, down_props, down_rels_allowed)

        props_packet_new = {
            upstream: {"properties": up_defs, "relations": up_rels},
            mid: {"properties": mid_defs, "relations": mid_rels},
            downstream: {"properties": down_defs, "relations": down_rels},
        }
        rules_packet_new = {
            upstream: {"rules": up_rules},
            mid: {"rules": mid_rules},
            downstream: {"rules": down_rules},
        }

        cand_score, _dbg_new = score_joint_sample(props_packet_new, rules_packet_new)

        acc_ratio = min(1.0, math.exp(beta * (cand_score - curr_score)))
        accepted = random.random() < acc_ratio

        if accepted:
            curr_score = cand_score

        if cand_score > best_score:
            best_score = cand_score
            best_props = props_packet_new
            best_rules = rules_packet_new

        debug["mcmc"].append(
            {
                "iter": it,
                "score": cand_score,
                "accepted": accepted,
                "acc_ratio": acc_ratio,
            }
        )

    debug["best_score"] = best_score
    return best_props, best_rules, debug


# ------------------- Global merge and orchestration -------------------


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    prop_defs_base, rules_base = _load_baseline_defs_and_rules()
    net = load_network()
    triples = _extract_triples(net)

    # start from baseline
    merged_prop_defs = json.loads(json.dumps(prop_defs_base))  # deep copy
    merged_rules = json.loads(json.dumps(rules_base))          # deep copy

    # track best score per (agent, property) / (agent, rule) for conflict resolution
    best_prop_score: Dict[Tuple[str, str], float] = {}
    best_rule_score: Dict[Tuple[str, str], float] = {}

    debug_all: Dict[str, Any] = {
        "triples": triples,
        "runs": [],
    }

    for (up, mid, down) in triples:
        best_props, best_rules_for_triple, dbg = _run_triple_mcmc(
            up, mid, down, prop_defs_base, rules_base, n_rounds=6, beta=1.0
        )
        debug_all["runs"].append(dbg)
        triple_score = dbg.get("best_score", 0.0)

        # merge properties and relations
        for agent in (up, mid, down):
            agent_best = best_props.get(agent, {})
            p_def = agent_best.get("properties", {}) or {}
            r_def = agent_best.get("relations", []) or []

            # ensure blocks exist
            merged_agent_blk = merged_prop_defs.setdefault(agent, {"properties": {}, "relations": []})
            merged_p = merged_agent_blk.setdefault("properties", {})
            merged_r = merged_agent_blk.setdefault("relations", [])

            # properties: update if this triple has higher score
            for pname, meta in p_def.items():
                key = (agent, pname)
                prev_best = best_prop_score.get(key, -1e9)
                if triple_score >= prev_best:
                    merged_p[pname] = meta
                    best_prop_score[key] = triple_score

            # relations: we merge by name; higher-scoring triple wins
            for r in r_def:
                rname = r.get("name")
                if not isinstance(rname, str):
                    continue
                key = (agent, rname)
                prev_best = best_prop_score.get(key, -1e9)
                if triple_score >= prev_best:
                    # replace or append
                    replaced = False
                    for i, old in enumerate(merged_r):
                        if old.get("name") == rname:
                            merged_r[i] = r
                            replaced = True
                            break
                    if not replaced:
                        merged_r.append(r)
                    best_prop_score[key] = triple_score

        # merge rules
        for agent in (up, mid, down):
            rules_best_blk = best_rules_for_triple.get(agent, {}) or {}
            rules_list = rules_best_blk.get("rules", []) or []
            merged_agent_rules_blk = merged_rules.setdefault(agent, {"rules": []})
            merged_rules_list = merged_agent_rules_blk.setdefault("rules", [])

            # index existing by 'write'
            idx = {r.get("write"): i for i, r in enumerate(merged_rules_list) if isinstance(r, dict)}

            for r in rules_list:
                w = r.get("write")
                if not isinstance(w, str):
                    continue
                key = (agent, w)
                prev_best = best_rule_score.get(key, -1e9)
                if triple_score >= prev_best:
                    if w in idx:
                        merged_rules_list[idx[w]] = r
                    else:
                        merged_rules_list.append(r)
                    best_rule_score[key] = triple_score

    # write outputs
    with open(PATH_OUT_PROP, "w", encoding="utf-8") as f:
        json.dump(merged_prop_defs, f, ensure_ascii=False, indent=2)

    with open(PATH_OUT_RULES, "w", encoding="utf-8") as f:
        json.dump(merged_rules, f, ensure_ascii=False, indent=2)

    with open(PATH_OUT_DEBUG, "w", encoding="utf-8") as f:
        json.dump(debug_all, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {PATH_OUT_PROP}")
    print(f"Wrote: {PATH_OUT_RULES}")
    print(f"Wrote: {PATH_OUT_DEBUG}")


if __name__ == "__main__":
    main()
