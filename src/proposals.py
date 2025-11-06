# -*- coding: utf-8 -*-
"""
Free-form property proposal for supply-chain agents.

This version intentionally removes ANY synonym mapping or keyword-based bucketing.
It treats names returned by the LLM as the single source of truth (after
a light PascalCase normalization), so downstream modules (STATE/ACTION MCMC, renderers)
can rely on one consistent vocabulary coming from outputs/final_structure.json.

Behavior kept:
- Ask LLM for 2–6 concise PascalCase properties per agent
- Light sanitize: PascalCase + per-agent de-dup + clamp to [min_k, max_k]
- NO renaming, NO collapsing, NO dimension checking
"""

import os
import json
import re
from openai import OpenAI

# DashScope (Aliyun) OpenAI-compatible endpoint
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"

SYSTEM = "You are a supply chain expert. Output ONLY valid JSON. No extra text."

# ====== Free-form prompt (no schema whitelist) ======
FREEFORM_PROMPT = """You will propose agent-specific properties for a supply chain simulation.

Goal:
For each agent, list 2–6 short properties that reflect its decision-relevant state under SIX conceptual dimensions:
- Stock level
- Debt
- Cost
- Capacity
- Assets
- Price

Rules:
- Create concise PascalCase names (e.g., Inventory, Backlog, HoldingCost, ServiceCapacity, Cash, Price).
- Each property must logically belong to ONE of the six dimensions above.
- Avoid units/values; names only. No explanations.
- Output a SINGLE JSON object mapping each agent name to a flat array of property names.
- Agents: [Supplier, Manufacturer, Distributor, Wholesaler, Retailer]
"""


# ---------- Utilities ----------
def _extract_json(text: str) -> str:
    """
    Extract the first JSON-looking block to make parsing robust to minor pre/post text.
    """
    m = re.search(r"\{.*\}", text, flags=re.S)
    return m.group(0) if m else text

def _get_api_key() -> str:
    """
    Read DASHSCOPE_API_KEY from env, otherwise fallback to configs/keys.json.
    """
    key = os.getenv("DASHSCOPE_API_KEY")
    if key:
        return key
    try:
        here = os.path.dirname(__file__)
        with open(os.path.join(here, "..", "configs", "keys.json"), "r", encoding="utf-8") as f:
            j = json.load(f)
            return j.get("DASHSCOPE_API_KEY", "")
    except Exception:
        return ""

# ---------- Light sanitize (NO synonyms, NO keyword bucketing) ----------
def _pascal(s: str) -> str:
    """
    Convert to PascalCase while preserving internal capitals.
    Example: 'raw_material inventory' -> 'Raw_materialInventory' -> then split/normalize:
    Here we simply split on whitespace/dash/underscore and .title() each token.
    """
    tokens = re.split(r"[\s\-_]+", s)
    return "".join(t[:1].upper() + t[1:] for t in tokens if t)

def _light_sanitize(js: dict, min_k: int = 2, max_k: int = 6) -> dict:
    """
    Minimal postprocess:
      - PascalCase normalization
      - Per-agent de-duplication (stable order)
      - Clamp count into [min_k, max_k]
    IMPORTANT: No renaming beyond PascalCase. No synonym collapsing.
    """
    fixed = {}
    for agent, props in js.items():
        seen, kept = set(), []
        for p in props:
            n = _pascal(str(p))
            if not n:
                continue
            if n not in seen:
                kept.append(n)
                seen.add(n)
        # Clamp to the requested range (we do not force a minimum here — that is handled later if needed)
        if len(kept) > max_k:
            kept = kept[:max_k]
        fixed[agent] = kept
    return fixed

# ---------- LLM call ----------
def llm_generate_structure(min_k: int = 2, max_k: int = 6) -> dict:
    """
    Free-form property generation from commonsense (no schema whitelist).
    Postprocess with light normalization (PascalCase + dedupe + clamp).
    This returns the authoritative vocabulary for subsequent pipelines.
    """
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY (env var) or configs/keys.json")

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": FREEFORM_PROMPT}],
        temperature=0.7,
        max_tokens=800,
    )
    txt = resp.choices[0].message.content
    js = json.loads(_extract_json(txt))

    # Minimal, authority-preserving sanitize
    js = _light_sanitize(js, min_k=min_k, max_k=max_k)
    return js



