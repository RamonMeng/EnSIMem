#!/usr/bin/env python3
"""Adapt one LongMemEval item into the exact file shapes consumed by the LoCoMo EnSI pipeline.

Gold answers are intentionally not written into any file read by Steps 2-6.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from longmemeval_adapter import adapt_instance, load_instances, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare one LongMemEval item for EnSI-Memory")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--question-id", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    dataset = {str(x["question_id"]): x for x in load_instances(args.dataset)}
    if args.question_id not in dataset:
        raise ValueError(f"question_id not found in dataset: {args.question_id}")
    raw = dataset[args.question_id]
    adapted = adapt_instance(raw)
    out = Path(args.run_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Shape intentionally mirrors LoCoMo 01_prepare_sessions.py.
    write_json(
        out / "01_sessions.json",
        {
            "schema_version": "longmemeval-ensi-locomo-method-step1-v1",
            "conversation_id": adapted["question_id"],
            "question_id": adapted["question_id"],
            "question_type": adapted["question_type"],
            "question_date": adapted["question_date"],
            "image_evidence_policy": "text_only",
            "episode_policy": "partition each session into contiguous theme-coherent episodes in Step 2",
            "extraction_context_radius": 2,
            "corpus_subset_policy": "complete_query_private_haystack",
            "sessions": adapted["sessions"],
        },
    )
    write_json(
        out / "01_questions.json",
        {
            "conversation_id": adapted["question_id"],
            "questions": [
                {
                    "query_id": adapted["question_id"],
                    "conversation_id": adapted["question_id"],
                    "question": adapted["question"],
                }
            ],
        },
    )
    # Public metadata needed by the LongMemEval-specific generator. No gold answer.
    write_json(
        out / "01_longmemeval_metadata.json",
        {
            "question_id": adapted["question_id"],
            "question": adapted["question"],
            "question_type": adapted["question_type"],
            "question_date": adapted["question_date"],
            "source_session_count": adapted["source_session_count"],
            "source_turn_count": adapted["source_turn_count"],
        },
    )
    print(
        f"Prepared {adapted['question_id']}: {adapted['source_session_count']} sessions, "
        f"{adapted['source_turn_count']} turns"
    )


if __name__ == "__main__":
    main()
