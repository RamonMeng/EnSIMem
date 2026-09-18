#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from _common import read_json, select_questions, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare raw LoCoMo sessions for theme partitioning")
    parser.add_argument("--locomo", default="data/locomo10.json")
    parser.add_argument("--out-dir", default="runs/entity_condition_hop_v3")
    parser.add_argument("--conversation-index", type=int, default=0)
    parser.add_argument("--questions-per-category", type=int, default=3, help="0 selects all questions")
    parser.add_argument(
        "--selection-file",
        default=None,
        help="Optional balanced manifest; preserves its query order/IDs and excludes unselected categories",
    )
    parser.add_argument("--query-start", type=int, default=1, help="1-based question position after selection")
    parser.add_argument("--query-count", type=int, default=0, help="Number of questions; 0 selects through the end")
    parser.add_argument("--extraction-context-radius", type=int, default=2)
    parser.add_argument("--include-image-captions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--oracle-relevant-sessions-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Diagnostic only: preprocess sessions named by selected questions' gold evidence IDs",
    )
    args = parser.parse_args()
    if args.extraction_context_radius < 0:
        raise ValueError("extraction-context-radius must be non-negative")
    if args.query_start <= 0 or args.query_count < 0:
        raise ValueError("query-start must be positive and query-count cannot be negative")
    out = Path(args.out_dir)
    existing = sorted(str(path) for path in out.glob("0*.json"))
    if existing:
        raise ValueError(f"Use a fresh --out-dir; checkpoints already exist: {existing}")
    samples = read_json(args.locomo)
    if not isinstance(samples, list) or not samples:
        raise ValueError("LoCoMo input must be a non-empty JSON list")
    if not 0 <= args.conversation_index < len(samples):
        raise ValueError("conversation-index is outside the LoCoMo list")
    sample = samples[args.conversation_index]
    conversation_id = str(sample["sample_id"])
    conversation = sample["conversation"]
    session_ids = sorted(
        (key for key in conversation if key.startswith("session_") and key.count("_") == 1),
        key=lambda value: int(value.split("_")[1]),
    )
    sessions = []
    for session_id in session_ids:
        turns = [dict(turn) for turn in conversation[session_id]]
        dia_ids = [str(turn.get("dia_id", "")) for turn in turns]
        if not turns or not all(dia_ids) or len(set(dia_ids)) != len(dia_ids):
            raise ValueError(f"{session_id} requires unique non-empty dia_ids")
        sessions.append(
            {
                "conversation_id": conversation_id,
                "session_id": session_id,
                "observed_at": str(conversation.get(f"{session_id}_date_time", "")),
                "speaker_a": conversation.get("speaker_a"),
                "speaker_b": conversation.get("speaker_b"),
                "include_image_captions": args.include_image_captions,
                "extraction_context_radius": args.extraction_context_radius,
                "dia_ids": dia_ids,
                "turns": turns,
            }
        )
    if args.selection_file:
        selection_path = Path(args.selection_file)
        with selection_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest_queries = manifest.get("queries") if isinstance(manifest, dict) else None
        if not isinstance(manifest_queries, list):
            raise ValueError("selection-file must contain a JSON object with a queries list")
        selected_manifest = [
            item
            for item in manifest_queries
            if (
                str(item.get("conversation_id", "")) == conversation_id
                or int(item.get("conversation_index", -1)) == args.conversation_index
            )
        ]
        if not selected_manifest:
            raise ValueError(
                f"selection-file contains no queries for conversation {conversation_id} "
                f"(index {args.conversation_index})"
            )
        chosen = selected_manifest
    else:
        chosen = select_questions(sample.get("qa") or [], args.questions_per_category)
    start = args.query_start - 1
    end = start + args.query_count if args.query_count else None
    chosen = chosen[start:end]
    if not chosen:
        raise ValueError("query range selected no questions")
    questions, gold = [], []
    for position, item in enumerate(chosen, 1):
        # Balanced manifests already carry stable benchmark IDs (for example b0001).
        # For raw LoCoMo QA, retain the local q0001-style IDs used by the old pipeline.
        query_id = str(item.get("query_id") or f"q{position:04d}")
        category = int(item.get("category", 0))
        questions.append({"query_id": query_id, "conversation_id": conversation_id, "question": str(item["question"])})
        gold.append(
            {
                "query_id": query_id,
                "category": category,
                "answer": item.get("answer", "Unknown" if category == 5 else ""),
                "has_explicit_answer": "answer" in item,
                "evidence": item.get("evidence") or [],
            }
        )
    selected_session_ids = [session["session_id"] for session in sessions]
    if args.oracle_relevant_sessions_only:
        evidence_session_numbers = set()
        for item in gold:
            for raw_evidence in item["evidence"]:
                for evidence in str(raw_evidence).split(";"):
                    match = re.fullmatch(r"\s*D(\d+):\d+\s*", evidence)
                    if match:
                        evidence_session_numbers.add(int(match.group(1)))
        selected_session_ids = [f"session_{number}" for number in sorted(evidence_session_numbers)]
        selected = set(selected_session_ids)
        sessions = [session for session in sessions if session["session_id"] in selected]
        if not sessions:
            raise ValueError("oracle session selection found no sessions")
    write_json(
        out / "01_sessions.json",
        {
            "schema_version": "entity-condition-hop-v3-raw-sessions",
            "conversation_id": conversation_id,
            "image_evidence_policy": "caption_proxy" if args.include_image_captions else "text_only",
            "selection_file": str(args.selection_file or ""),
            "episode_policy": "partition each session into contiguous theme-coherent episodes in Step 2",
            "extraction_context_radius": args.extraction_context_radius,
            "corpus_subset_policy": (
                "oracle_gold_evidence_sessions_for_diagnostic_only"
                if args.oracle_relevant_sessions_only
                else "all_conversation_sessions"
            ),
            "selected_session_ids": selected_session_ids,
            "warning": (
                "Gold evidence IDs were used only to define this diagnostic corpus subset; "
                "do not report its retrieval score as a full-corpus benchmark."
                if args.oracle_relevant_sessions_only
                else ""
            ),
            "sessions": sessions,
        },
    )
    write_json(out / "01_questions.json", {"conversation_id": conversation_id, "questions": questions})
    write_json(out / "01_gold_DO_NOT_USE_BEFORE_STEP_7.json", {"conversation_id": conversation_id, "gold": gold})
    print(f"Prepared {conversation_id}: {len(sessions)} raw sessions and {len(questions)} questions")
    print(f"Selected sessions: {selected_session_ids}")
    print("Episodes are created by theme partitioning in Step 2; no session is an episode yet.")


if __name__ == "__main__":
    main()
