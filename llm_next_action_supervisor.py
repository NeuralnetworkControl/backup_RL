# # llm_next_action_supervisor.py
# from __future__ import annotations
# import json
# import os
# import time
# from pathlib import Path
# from typing import List, Set
#
# import pandas as pd
# import requests
#
# INFO_SLOTS = ["edu", "role", "exec", "industry", "depth"]
#
# # ---------- merged.csv cache ----------
# _MERGED_CACHE = None
#
#
# def _load_merged(merged_csv_path="merged.csv"):
#     global _MERGED_CACHE
#     if _MERGED_CACHE is None:
#         _MERGED_CACHE = pd.read_csv(merged_csv_path)
#     return _MERGED_CACHE
#
#
# def _append_llm_log(prompt: str, raw: str | None, parsed: dict | None, log_path: str):
#     ts = time.strftime("%Y-%m-%d %H:%M:%S")
#     p = Path(log_path)
#
#     entry = []
#     entry.append("\n" + "=" * 80)
#     entry.append(f"[{ts}] LLM NEXT ACTION")
#     entry.append("-" * 80)
#     entry.append("[PROMPT]")
#     entry.append(prompt)
#     entry.append("-" * 80)
#     entry.append("[RAW]")
#     entry.append("" if raw is None else raw)
#     entry.append("-" * 80)
#     entry.append("[PARSED]")
#     entry.append("null" if parsed is None else json.dumps(parsed, ensure_ascii=False))
#     entry.append("=" * 80 + "\n")
#
#     with p.open("a", encoding="utf-8") as f:
#         f.write("\n".join(entry))
#         f.flush()
#
#
# # ---------- slot-aware profile summary ----------
#
# def _summarise_profile(profile: dict, observed_slots: Set[str]) -> str:
#     """
#     只根据「已经 query 过的 slot」生成摘要
#     确保 LLM 视角 == agent 视角
#     """
#     lines: List[str] = []
#
#     def add(line: str):
#         if line and len(lines) < 10:
#             lines.append(line)
#
#     # industry
#     if "industry" in observed_slots:
#         add(f"Industry: {profile.get('industry')}")
#
#     # role
#     if "role" in observed_slots:
#         add(f"Current role: {profile.get('role')}")
#
#     # exec: 从 jobs_json / prose 中提取
#     if "exec" in observed_slots:
#         jobs = profile.get("jobs_json")
#         if isinstance(jobs, str):
#             add("Executive signals:")
#             for j in jobs.split("\\n"):
#                 if any(k in j.lower() for k in ["founder", "ceo", "head", "director"]):
#                     add(f"- {j.strip()}")
#                     break
#
#     # edu
#     if "edu" in observed_slots:
#         edu = profile.get("educations_json")
#         if isinstance(edu, str):
#             add("Education:")
#             for e in edu.split("\\n"):
#                 if "degree" in e.lower():
#                     add(f"- {e.strip()}")
#                     break
#
#     # depth: 年限 / 资历
#     if "depth" in observed_slots:
#         prose = profile.get("anonymised_prose")
#         if isinstance(prose, str) and prose.strip():
#             add(f"Experience summary: {prose.split('.')[0].strip()}.")
#
#     if not lines:
#         return "(No concrete information observed yet)"
#
#     return "\n".join(lines)
#
#
# def _build_prompt_from_merged(fid: str, actions_taken, merged_csv_path="merged.csv") -> str:
#     """
#     next-action supervisor prompt：
#     - 只暴露已 query 的信息
#     - 不泄露 merged.csv 的未观测字段
#     """
#     df = _load_merged(merged_csv_path)
#     row = df[df["founder_uuid"] == fid]
#     profile = row.iloc[0].to_dict() if len(row) > 0 else {}
#
#     observed_slots: Set[str] = {
#         a.get("action")
#         for a in actions_taken
#         if isinstance(a, dict) and a.get("action") in INFO_SLOTS
#     }
#
#     profile_summary = _summarise_profile(profile, observed_slots)
#
#     return (
#         "You are helping an information-gathering agent choose the NEXT query.\n"
#         "This is NOT about stopping.\n"
#         "This is ONLY about which remaining slot is MOST likely to change the final decision.\n\n"
#
#         f"Already queried slots: {sorted(observed_slots)}\n\n"
#
#         "Available slots (choose ONLY from these):\n"
#         "- edu: education background, including degrees (Bachelor/Master/PhD), "
#         "fields of study, and institution quality signals such as QS ranking. "
#         "This reflects formal training and academic strength.\n"
#         "- role: professional roles and job titles across the career, combined with "
#         "industry context and company environment. "
#         "This reflects what the person has actually been doing.\n"
#         "- exec: executive or leadership signals from career history, such as "
#         "founder, CEO, CTO, head, director, or other decision-making positions. "
#         "This reflects authority and leadership.\n"
#         "- industry: industry background and domain exposure aggregated from declared "
#         "industry fields and job history. "
#         "This provides contextual alignment and disambiguation.\n"
#         "- depth: experience depth and seniority, including total years of experience, "
#         "number of roles held, and presence of long-tenure positions. "
#         "This reflects how seasoned the person is.\n\n"
#
#
#         "Observed information (agent-visible only):\n"
#         f"{profile_summary}\n\n"
#
#         "Return ONLY valid JSON.\n"
#         "Allowed values: edu, role, exec, industry, depth\n"
#         "Format:\n"
#         '{"prefer": ["exec", "role"]}\n'
#         "If nothing stands out:\n"
#         '{"prefer": []}\n'
#     )
#
#
# def llm_prefer_next_actions_from_merged(
#         fid: str,
#         actions_taken,
#         *,
#         merged_csv_path="merged.csv",
#         log_path: str = "llm_next_action.log",
# ) -> List[str]:
#     """
#     next-action LLM supervisor（slot-consistent 版本）
#     """
#     api_key = os.getenv("OPENAI_API_KEY", "").strip()
#     if not api_key:
#         return []
#
#     model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
#     base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
#     url = f"{base_url}/chat/completions"
#
#     prompt = _build_prompt_from_merged(fid, actions_taken, merged_csv_path)
#
#     payload = {
#         "model": model,
#         "messages": [
#             {"role": "system", "content": "You must reply with ONLY valid JSON. No extra text."},
#             {"role": "user", "content": prompt},
#         ],
#         "temperature": 0.0,
#         "max_tokens": 200,
#     }
#
#     raw = None
#     parsed = None
#
#     try:
#         r = requests.post(
#             url,
#             headers={
#                 "Authorization": f"Bearer {api_key}",
#                 "Content-Type": "application/json",
#             },
#             data=json.dumps(payload),
#             timeout=20,
#         )
#         if r.status_code == 200:
#             raw = r.json()["choices"][0]["message"]["content"]
#             if raw and raw.strip():
#                 parsed = json.loads(raw)
#             else:
#                 raw = "[EMPTY_COMPLETION]"
#     except Exception as e:
#         raw = f"[EXCEPTION] {repr(e)}"
#
#     _append_llm_log(prompt, raw, parsed if isinstance(parsed, dict) else None, log_path)
#
#     if isinstance(parsed, dict):
#         prefer = parsed.get("prefer", [])
#         if isinstance(prefer, list):
#             return [p for p in prefer if p in INFO_SLOTS]
#
#     return []
# llm_next_action_supervisor.py
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Set, Dict, Any

import pandas as pd
import requests

INFO_SLOTS = ["edu", "role", "exec", "industry", "depth"]

# ================= merged.csv cache =================
_MERGED_CACHE: pd.DataFrame | None = None


def _load_merged(path: str) -> pd.DataFrame:
    global _MERGED_CACHE
    if _MERGED_CACHE is None:
        _MERGED_CACHE = pd.read_csv(path)
    return _MERGED_CACHE


# ================= logging =================

def _append_llm_log(prompt: str, raw: Any, parsed: Any, log_path: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    p = Path(log_path)

    entry = [
        "\n" + "=" * 80,
        f"[{ts}] GPT-5.2 NEXT ACTION",
        "-" * 80,
        "[PROMPT]",
        prompt,
        "-" * 80,
        "[RAW]",
        json.dumps(raw, ensure_ascii=False, indent=2) if raw is not None else "null",
        "-" * 80,
        "[PARSED]",
        json.dumps(parsed, ensure_ascii=False, indent=2) if parsed is not None else "null",
        "=" * 80 + "\n",
    ]

    with p.open("a", encoding="utf-8") as f:
        f.write("\n".join(entry))
        f.flush()

def _extract_parsed_from_response(raw: dict):
    """
    GPT-5 Responses API robust parser:
    1) Prefer output_parsed if present
    2) Fallback to parsing output_text JSON
    """
    # 1️⃣ 官方结构化字段（不保证一定有）
    parsed = raw.get("output_parsed")
    if isinstance(parsed, dict):
        return parsed

    # 2️⃣ fallback：从 output[].content[].text 里 parse
    try:
        outputs = raw.get("output", [])
        for msg in outputs:
            if msg.get("type") != "message":
                continue
            for c in msg.get("content", []):
                if c.get("type") == "output_text":
                    txt = c.get("text", "").strip()
                    if txt.startswith("{") and txt.endswith("}"):
                        return json.loads(txt)
    except Exception:
        pass

    return None
# ================= profile summarisation =================

def _summarise_profile(profile: Dict[str, Any], observed: Set[str]) -> str:
    """
    只暴露 agent 已经 query 过的 slot
    """
    lines: List[str] = []

    def add(x: str):
        if x and len(lines) < 10:
            lines.append(x)

    if "industry" in observed:
        add(f"Industry: {profile.get('industry')}")

    if "role" in observed:
        add(f"Role: {profile.get('role')}")

    if "exec" in observed:
        jobs = profile.get("jobs_json")
        if isinstance(jobs, str):
            for j in jobs.split("\\n"):
                if any(k in j.lower() for k in ["founder", "ceo", "cto", "head", "director"]):
                    add(f"Exec signal: {j.strip()}")
                    break

    if "edu" in observed:
        edu = profile.get("educations_json")
        if isinstance(edu, str):
            for e in edu.split("\\n"):
                if "degree" in e.lower():
                    add(f"Education: {e.strip()}")
                    break

    if "depth" in observed:
        prose = profile.get("anonymised_prose")
        if isinstance(prose, str) and prose.strip():
            add(f"Experience: {prose.split('.')[0].strip()}.")

    return "\n".join(lines) if lines else "(No observed information yet)"


# ================= prompt =================

def _build_prompt(fid: str, actions_taken, merged_csv_path: str) -> str:
    df = _load_merged(merged_csv_path)
    row = df[df["founder_uuid"] == fid]
    profile = row.iloc[0].to_dict() if len(row) else {}

    observed: Set[str] = {
        a.get("action")
        for a in actions_taken
        if isinstance(a, dict) and a.get("action") in INFO_SLOTS
    }

    summary = _summarise_profile(profile, observed)

    return f"""
You are a decision support module for an information-gathering agent.

Your task:
Choose which ONE OR TWO remaining information slots are MOST likely
to change the final success/failure decision.

Already observed slots:
{sorted(observed)}

Observed information (agent-visible only):
{summary}

Available slots (choose ONLY from these):

- edu: education background, including degrees (Bachelor/Master/PhD),
  fields of study, and institution quality signals such as QS ranking.
  This reflects formal training and academic strength.

- role: professional roles and job titles across the career, combined with
  industry context and company environment.
  This reflects what the person has actually been doing.

- exec: executive or leadership signals from career history, such as
  founder, CEO, CTO, head, director, or other decision-making positions.
  This reflects authority and leadership.

- industry: industry background and domain exposure aggregated from declared
  industry fields and job history.
  This provides contextual alignment and disambiguation.

- depth: experience depth and seniority, including total years of experience,
  number of roles held, and presence of long-tenure positions.
  This reflects how seasoned the person is.

Rules:
- Do NOT suggest slots already observed.
- Prefer slots with the highest marginal information value.
- If nothing stands out, return an empty list.
- Return structured output only (no explanations).

""".strip()



# ================= GPT-5.2 call =================

def llm_prefer_next_actions_from_merged(
    fid: str,
    actions_taken,
    *,
    merged_csv_path: str = "merged.csv",
    log_path: str = "llm_next_action.log",
) -> List[str]:

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return []

    model = os.getenv("OPENAI_MODEL", "gpt-5.2")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = f"{base_url}/responses"

    prompt = _build_prompt(fid, actions_taken, merged_csv_path)

    # ===== JSON schema: GPT-5.2 真正吃这一套 =====
    schema = {
        "name": "next_action",
        "schema": {
            "type": "object",
            "properties": {
                "prefer": {
                    "type": "array",
                    "items": {"type": "string", "enum": INFO_SLOTS},
                    "maxItems": 2,
                }
            },
            "required": ["prefer"],
            "additionalProperties": False,
        },
    }

    payload = {
        "model": model,
        "input": prompt,
        "temperature": 0.0,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "next_action",  # ✅ 必须有
                "schema": {
                    "type": "object",
                    "properties": {
                        "prefer": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": INFO_SLOTS
                            },
                            "maxItems": 2
                        }
                    },
                    "required": ["prefer"],
                    "additionalProperties": False
                }
            }
        }
    }

    raw = None
    parsed = None

    try:
        r = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=30,
        )
        raw = r.json()

        parsed = _extract_parsed_from_response(raw)


    except Exception as e:
        raw = {"exception": repr(e)}

    _append_llm_log(prompt, raw, parsed, log_path)

    if isinstance(parsed, dict):
        prefer = parsed.get("prefer")
        if isinstance(prefer, list):
            return [p for p in prefer if p in INFO_SLOTS]

    return []
