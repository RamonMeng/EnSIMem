#!/usr/bin/env python3
"""Convert one LongMemEval item into the input shape used by EnSI.

The original EnSI pipeline expects LoCoMo-style sessions with ``speaker``,
``text`` and unique ``dia_id`` fields.  LongMemEval uses ``role`` and
``content`` instead, and every benchmark item owns an isolated haystack.  This
adapter bridges only that representation gap.  Gold answers and ``has_answer``
labels are retained for audit metadata but are never used by retrieval or
generation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DATASET_FILENAME = "longmemeval_s_cleaned.json"
SCHEMA_VERSION = "longmemeval-s-cleaned-ensi-adapter-v1"


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(destination)


def load_instances(path: str | Path) -> list[dict[str, Any]]:
    value = read_json(path)
    if not isinstance(value, list):
        raise ValueError(f"LongMemEval dataset must be a JSON list: {path}")
    for position, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"dataset item {position} is not an object")
        for field in (
            "question_id",
            "question",
            "question_date",
            "haystack_session_ids",
            "haystack_dates",
            "haystack_sessions",
        ):
            if field not in item:
                raise ValueError(f"dataset item {position} is missing {field!r}")
    return value


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned.strip("._") or "unknown_query"


def _turn_content(raw_turn: dict[str, Any]) -> str:
    content = raw_turn.get("content", raw_turn.get("text", ""))
    if isinstance(content, list):
        # The released S dataset uses strings.  This fallback keeps the
        # adapter readable if a future copy contains structured content parts.
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", part.get("content", ""))))
            else:
                parts.append(str(part))
        return " ".join(part for part in parts if part).strip()
    return str(content or "").strip()


def _raw_session_turns(raw_session: Any) -> list[dict[str, Any]]:
    if isinstance(raw_session, list):
        turns = raw_session
    elif isinstance(raw_session, dict):
        turns = raw_session.get("turns") or raw_session.get("messages") or []
    else:
        turns = []
    if not isinstance(turns, list) or not turns:
        raise ValueError("each LongMemEval haystack session must contain turns")
    if not all(isinstance(turn, dict) for turn in turns):
        raise ValueError("LongMemEval session turns must be objects")
    return turns


def adapt_instance(item: dict[str, Any]) -> dict[str, Any]:
    """Return one isolated EnSI-compatible conversation plus benchmark metadata."""
    question_id = str(item["question_id"])
    session_ids = item["haystack_session_ids"]
    dates = item["haystack_dates"]
    raw_sessions = item["haystack_sessions"]
    if not isinstance(session_ids, list) or not isinstance(dates, list):
        raise ValueError(f"{question_id}: session ids/dates must be lists")
    if not isinstance(raw_sessions, list):
        raise ValueError(f"{question_id}: haystack_sessions must be a list")
    if not (len(session_ids) == len(dates) == len(raw_sessions)):
        raise ValueError(
            f"{question_id}: session ids, dates and sessions have different lengths"
        )

    sessions: list[dict[str, Any]] = []
    source_turn_count = 0
    for session_position, (raw_session_id, raw_date, raw_session) in enumerate(
        zip(session_ids, dates, raw_sessions)
    ):
        session_id = str(raw_session_id)
        observed_at = str(raw_date)
        turns: list[dict[str, Any]] = []
        for turn_position, raw_turn in enumerate(_raw_session_turns(raw_session)):
            role = str(raw_turn.get("role", "")).strip().lower()
            if role == "user":
                speaker = "User"
            elif role == "assistant":
                speaker = "Assistant"
            else:
                speaker = role.title() if role else "Unknown"
            # DIA IDs are synthetic EnSI identifiers.  The source session ID,
            # source role and source position remain attached for evaluation.
            dia_id = f"{session_id}::turn_{turn_position:04d}"
            turns.append(
                {
                    "dia_id": dia_id,
                    "speaker": speaker,
                    "text": _turn_content(raw_turn),
                    "source_session_id": session_id,
                    "source_turn_index": turn_position,
                    "source_role": role,
                    # This is deliberately not rendered by format_turns and is
                    # never passed as a retrieval signal.  It is audit-only.
                    "has_answer": bool(raw_turn.get("has_answer", False)),
                }
            )
        source_turn_count += len(turns)
        sessions.append(
            {
                "conversation_id": question_id,
                "session_id": session_id,
                "source_session_id": session_id,
                "source_session_position": session_position,
                "observed_at": observed_at,
                "include_image_captions": False,
                "turns": turns,
                "dia_ids": [turn["dia_id"] for turn in turns],
            }
        )

    # Keep the answer in this in-memory object only so the optional evaluation
    # helper can join predictions with gold later.  The pipeline never sends
    # it to any LLM and does not write it into the per-query prompt artifacts.
    return {
        "schema_version": SCHEMA_VERSION,
        "question_id": question_id,
        "question_type": str(item.get("question_type", "")),
        "question": str(item["question"]),
        "question_date": str(item["question_date"]),
        "sessions": sessions,
        "source_session_count": len(sessions),
        "source_turn_count": source_turn_count,
        "answer_session_ids": [str(value) for value in item.get("answer_session_ids", [])],
        "gold_answer": str(item.get("answer", "")),
        "is_abstention": question_id.endswith("_abs"),
    }


def public_instance_metadata(instance: dict[str, Any]) -> dict[str, Any]:
    """Return safe metadata for manifests without exposing the gold answer."""
    return {
        "schema_version": instance["schema_version"],
        "question_id": instance["question_id"],
        "question_type": instance["question_type"],
        "question": instance["question"],
        "question_date": instance["question_date"],
        "source_session_count": instance["source_session_count"],
        "source_turn_count": instance["source_turn_count"],
        "is_abstention": instance["is_abstention"],
    }

