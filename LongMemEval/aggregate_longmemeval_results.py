#!/usr/bin/env python3
"""Aggregate saved LongMemEval judge results across multiple run folders.

This script is read-only with respect to query results.  It reads
``06_prediction.json`` and ``07_judge.json`` files, de-duplicates by the
benchmark ``question_id``, and reports accuracy for all six categories plus
the overall accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from longmemeval_adapter import load_instances, read_json, write_json


QUESTION_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "knowledge-update",
    "temporal-reasoning",
)


def _bool_correct(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "yes", "1"}


def _run_dirs(roots: list[Path], explicit: list[Path] | None) -> list[Path]:
    if explicit:
        return [path.expanduser().resolve() for path in explicit]

    found: set[Path] = set()
    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"run root does not exist: {root}")
        for child in root.iterdir():
            if child.is_dir() and (child / "queries").is_dir():
                found.add(child)
    return sorted(found)


def _saved_rows(run_dir: Path, dataset: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    queries_dir = run_dir / "queries"
    rows: list[dict[str, Any]] = []
    if not queries_dir.is_dir():
        return rows

    for prediction_path in sorted(queries_dir.glob("*/06_prediction.json")):
        prediction = read_json(prediction_path)
        question_id = str(prediction.get("question_id", "")).strip()
        if not question_id or question_id not in dataset:
            continue
        query_dir = prediction_path.parent
        judge_path = query_dir / "07_judge.json"
        judge_file = read_json(judge_path) if judge_path.exists() else {}
        judge = judge_file.get("judge") if isinstance(judge_file, dict) else None
        if not isinstance(judge, dict):
            judge = prediction.get("judge")
        if not isinstance(judge, dict) or "correct" not in judge:
            continue

        judge_model = str(
            (judge_file.get("judge_model") if isinstance(judge_file, dict) else "")
            or prediction.get("judge_model", "")
        )
        rows.append(
            {
                "question_id": question_id,
                "question_type": str(dataset[question_id].get("question_type", "unknown")),
                "correct": _bool_correct(judge.get("correct")),
                "judge_model": judge_model,
                "run_dir": str(run_dir),
                "judge_path": str(judge_path),
                "judge_mtime": judge_path.stat().st_mtime if judge_path.exists() else 0.0,
            }
        )
    return rows


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(bool(row["correct"]) for row in rows)
    return {
        "count": len(rows),
        "correct": correct,
        "accuracy": round(correct / len(rows), 4) if rows else None,
    }


def _run_preference(run_dir: str) -> tuple[int, float]:
    """Prefer v2 result folders, then the most recently written judge file."""
    name = Path(run_dir).name.lower()
    version = 2 if "v2" in name else (1 if "v1" in name else 0)
    return version, 0.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate saved LongMemEval judge results across run folders"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--run-root",
        action="append",
        default=[],
        help="Root containing run directories; may be repeated",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=None,
        help="Exact run directory; may be repeated and overrides --run-root discovery",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Only include results whose judge_model matches this value",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    dataset_items = load_instances(args.dataset)
    dataset = {str(item["question_id"]): item for item in dataset_items}
    roots = [Path(path) for path in args.run_root]
    explicit = [Path(path) for path in args.run_dir] if args.run_dir else None
    if not roots and not explicit:
        parser.error("provide at least one --run-root or --run-dir")

    all_rows: list[dict[str, Any]] = []
    for run_dir in _run_dirs(roots, explicit):
        all_rows.extend(_saved_rows(run_dir, dataset))

    if args.judge_model:
        all_rows = [row for row in all_rows if row["judge_model"] == args.judge_model]

    # The same query can exist in a v1 and v2 run.  Prefer v2 result folders,
    # then the most recently written judge file.  This keeps a v2 rerun from
    # being silently replaced by an older exploratory run in another folder.
    selected: dict[str, dict[str, Any]] = {}
    for row in all_rows:
        previous = selected.get(row["question_id"])
        row_key = (_run_preference(row["run_dir"])[0], row["judge_mtime"])
        previous_key = (
            _run_preference(previous["run_dir"])[0],
            previous["judge_mtime"],
        ) if previous is not None else None
        if previous is None or row_key >= previous_key:
            selected[row["question_id"]] = row

    rows = list(selected.values())
    by_type = {
        question_type: _metrics(
            [row for row in rows if row["question_type"] == question_type]
        )
        for question_type in QUESTION_TYPES
    }
    unknown_types = sorted(
        {row["question_type"] for row in rows} - set(QUESTION_TYPES)
    )
    for question_type in unknown_types:
        by_type[question_type] = _metrics(
            [row for row in rows if row["question_type"] == question_type]
        )

    expected_by_type = {
        question_type: sum(
            str(item.get("question_type", "")) == question_type
            for item in dataset_items
        )
        for question_type in QUESTION_TYPES
    }
    missing_by_type = {
        question_type: expected_by_type[question_type] - by_type[question_type]["count"]
        for question_type in QUESTION_TYPES
    }
    summary = {
        "schema_version": "longmemeval-cross-run-summary-v1",
        "judge_model_filter": args.judge_model,
        "run_dirs": [str(path) for path in _run_dirs(roots, explicit)],
        "unique_judged_queries": len(rows),
        "expected_dataset_queries": len(dataset_items),
        "overall": _metrics(rows),
        "by_question_type": by_type,
        "expected_by_question_type": expected_by_type,
        "missing_by_question_type": missing_by_type,
        "selected_queries": sorted(selected.values(), key=lambda row: row["question_id"]),
    }

    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (roots[0].expanduser().resolve() / "all_categories_summary.json")
    )
    write_json(output, summary)

    print("category                               correct / total    accuracy")
    print("-------------------------------------------------------------------")
    for question_type in QUESTION_TYPES:
        metric = by_type[question_type]
        print(
            f"{question_type:36} {metric['correct']:7} / {metric['count']:<5}    "
            f"{metric['accuracy']}"
        )
    print("-------------------------------------------------------------------")
    print(
        f"{'overall':36} {summary['overall']['correct']:7} / "
        f"{summary['overall']['count']:<5}    {summary['overall']['accuracy']}"
    )
    print(f"Saved summary to {output}")
    incomplete = {key: value for key, value in missing_by_type.items() if value > 0}
    if incomplete:
        print(f"WARNING: unjudged dataset queries remain: {incomplete}")


if __name__ == "__main__":
    main()
