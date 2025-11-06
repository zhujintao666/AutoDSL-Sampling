import os
import json
import re
from typing import Dict, List, Tuple
from openai import OpenAI

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"
SYSTEM = "You are a supply chain expert. Output ONLY valid JSON. No extra text."


def _get_api_key() -> str:
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


def load_final_structure() -> Dict[str, List[str]]:
    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "outputs", "final_structure.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_json(text: str) -> str:
    m = re.search(r"\{.*\}", text, flags=re.S)
    return m.group(0) if m else text


def build_priority_with_dir_prompt(struct: Dict[str, List[str]]) -> str:
    lines = []
    lines.append("For each agent, rank its properties from HIGHEST to LOWEST priority AND assign a control direction.")
    lines.append("Direction rules: choose exactly one of 'Max' or 'Min' for each property.")
    lines.append('Output ONLY ONE JSON object, no commentary, no code fences.')
    lines.append('Format: {"Agent":[["PropName","Max|Min"], ...], ...}')
    lines.append("")
    for agent, props in struct.items():
        lines.append(f'{agent}: [{", ".join(props)}]')
    lines.append("")
    lines.append('Return only JSON like: {"Retailer":[["Stockout","Min"],["Inventory","Max"],["HoldingCost","Min"]]}')
    return "\n".join(lines)


def llm_rank_priorities_with_direction() -> Dict[str, List[Tuple[str, str]]]:
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("Missing DASHSCOPE_API_KEY (env var) or configs/keys.json")

    struct = load_final_structure()
    prompt = build_priority_with_dir_prompt(struct)

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000,
    )
    txt = resp.choices[0].message.content
    js = json.loads(_extract_json(txt))

    cleaned: Dict[str, List[Tuple[str, str]]] = {}
    for agent, props in struct.items():
        seen = set()
        lst = []
        proposed = js.get(agent, [])
        for item in proposed:
            try:
                name, direction = item[0], item[1]
            except Exception:
                continue
            if name not in props or name in seen:
                continue
            dnorm = "Max" if str(direction).strip().lower().startswith("max") else "Min"
            lst.append((name, dnorm))
            seen.add(name)
        for p in props:
            if p not in seen:
                lst.append((p, "Min"))
        cleaned[agent] = lst

    return cleaned
