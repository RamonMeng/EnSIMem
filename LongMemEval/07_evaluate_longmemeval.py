#!/usr/bin/env python3
"""Run the Memora-style LongMemEval judge over saved predictions.

This can be run independently after generation, or used to finish a run that
already has ``06_prediction.json`` files but stopped before evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from _common import make_client
from longmemeval_adapter import DATASET_FILENAME, load_instances, read_json, write_json
from memora_evaluation import (
    build_evaluation_metadata,
    judge_prediction,
    summarize_judges,
)


HERE = Path(__file__).resolve().parent
DATASET_CANDIDATES = (
    HERE.parent / "data" / DATASET_FILENAME,
    HERE.parent / DATASET_FILENAME,
)
DEFAULT_DATASET = next(
    (path for path in DATASET_CANDIDATES if path.exists()), DATASET_CANDIDATES[0]
)


def _query_id_from_prediction(path: Path) -> str:
    value = read_json(path)
    question_id = str(value.get("question_id", "")).strip()
    if not question_id:
        raise ValueError(f"prediction has no question_id: {path}")
    return question_id


def _find_query_dir(run_dir: Path, question_id: str, prediction_path: Path) -> Path:
    if prediction_path.parent.name == question_id:
        return prediction_path.parent
    # The main runner sanitizes IDs.  The fallback scan handles future IDs
    # whose filesystem-safe name differs from the source ID.
    for candidate in (run_dir / "queries").glob("*/06_prediction.json"):
        if _query_id_from_prediction(candidate) == question_id:
            return candidate.parent
    return prediction_path.parent


def _prediction_files(run_dir: Path, question_ids: set[str] | None) -> list[Path]:
    files = sorted((run_dir / "queries").glob("*/06_prediction.json"))
    if not files:
        raise FileNotFoundError(f"no 06_prediction.json files found under {run_dir / 'queries'}")
    if question_ids is None:
        return files
    selected = []
    for path in files:
        if _query_id_from_prediction(path) in question_ids:
            selected.append(path)
    missing = question_ids - {_query_id_from_prediction(path) for path in selected}
    if missing:
        raise FileNotFoundError(f"predictions not found for question_id(s): {sorted(missing)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Memora-style LongMemEval Step 7 evaluator")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--question-id", action="append", default=None)
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--api-key", default=None, help="Defaults to OPENAI_API_KEY")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dataset = {
        str(item["question_id"]): item
        for item in load_instances(args.dataset)
    }
    run_dir = Path(args.run_dir).expanduser().resolve()
    requested = set(args.question_id) if args.question_id else None
    prediction_files = _prediction_files(run_dir, requested)
    client = make_client(
        SimpleNamespace(
            provider="openai",
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            timeout=args.timeout,
        )
    )
    client.check()

    evaluated_rows: list[dict[str, Any]] = []
    for position, prediction_path in enumerate(prediction_files, 1):
        prediction = read_json(prediction_path)
        question_id = str(prediction["question_id"])
        sample = dataset.get(question_id)
        if sample is None:
            raise ValueError(f"prediction question_id is absent from dataset: {question_id}")
        query_dir = _find_query_dir(run_dir, question_id, prediction_path)
        judge_path = query_dir / "07_judge.json"
        existing: dict[str, Any] | None = None
        if judge_path.exists() and not args.force:
            candidate = read_json(judge_path)
            if isinstance(candidate, dict) and isinstance(candidate.get("judge"), dict):
                existing = candidate

        if existing is not None:
            judged = existing
            judge = judged["judge"]
            usage = judged.get("judge_usage", {})
            judge_status = "cached"
        else:
            client.begin_usage_scope(question_id)
            try:
                judge = judge_prediction(client, sample, str(prediction.get("hypothesis", prediction.get("prediction", ""))))
            finally:
                usage = client.end_usage_scope()
            judged = {
                "schema_version": "longmemeval-memora-style-judge-v1",
                "question_id": question_id,
                "judge_model": args.model,
                "judge": judge,
                "judge_usage": usage,
            }
            judge_status = "judged"

        retrieval_path = query_dir / "05_hop_retrievals.json"
        retrieval_wrapper = read_json(retrieval_path) if retrieval_path.exists() else {}
        retrieval_rows = retrieval_wrapper.get("retrievals") if isinstance(retrieval_wrapper, dict) else None
        retrieval = retrieval_rows[0] if isinstance(retrieval_rows, list) and retrieval_rows else {}
        evaluation = judged.get("evaluation")
        if not isinstance(evaluation, dict) or "gold_answer" not in evaluation:
            evaluation = build_evaluation_metadata(
                sample,
                retrieval,
                prediction.get("selected_reasoning_episode_ids", []),
            )
            judged = {
                **judged,
                "schema_version": "longmemeval-memora-style-judge-v1",
                "question_id": question_id,
                "judge_model": args.model,
                "gold_answer": evaluation["gold_answer"],
                "gold_answer_session_ids": evaluation["gold_answer_session_ids"],
                "gold_dialog_retrieval": {
                    "retrieved_gold_session_ids": evaluation[
                        "retrieved_gold_session_ids"
                    ],
                    "missing_gold_session_ids_after_step5": evaluation[
                        "missing_gold_session_ids_after_step5"
                    ],
                    "all_gold_sessions_retrieved": evaluation[
                        "all_gold_sessions_retrieved"
                    ],
                },
                "evaluation": evaluation,
                "judge_usage": usage,
            }
            write_json(judge_path, judged)
            judge_status = "metadata-updated" if judge_status == "cached" else judge_status

        if (
            prediction.get("judge") != judged["judge"]
            or prediction.get("evaluation") != evaluation
        ):
            prediction["judge"] = judged["judge"]
            prediction["judge_model"] = judged.get("judge_model", args.model)
            prediction["judge_usage"] = judged.get("judge_usage", usage)
            prediction["evaluation"] = evaluation
            write_json(prediction_path, prediction)
        print(
            f"[{position}/{len(prediction_files)}] {question_id}: {judge_status}; "
            f"judge={judged['judge']['correct']}; "
            f"gold_dialogs_retrieved={evaluation['all_gold_sessions_retrieved']}"
        )
        evaluated_rows.append(
            {
                "question_id": question_id,
                "question_type": str(sample.get("question_type", "")),
                "judge": judged["judge"],
                "evaluation": evaluation,
            }
        )

    summary = summarize_judges(evaluated_rows, args.model)
    summary["prediction_count"] = len(prediction_files)
    write_json(run_dir / "07_results_and_context_summary.json", summary)
    print(f"Saved evaluation summary to {run_dir / '07_results_and_context_summary.json'}")
    print(json.dumps(summary["overall"], ensure_ascii=False))


if __name__ == "__main__":
    main()
