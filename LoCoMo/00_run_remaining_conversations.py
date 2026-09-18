#!/usr/bin/env python3
"""Run the complete current LoCoMo entity-structured pipeline for several conversations.

This is an orchestration script only.  It does not change any retrieval or answer
logic.  Each conversation gets an independent run directory, so a failed
conversation can be resumed without mixing checkpoints with another one.

The default exclusion list is the five conversations already evaluated in the
current experiment (conv-26, conv-30, conv-41, conv-42, and conv-49).  The
remaining conversations are discovered from ``locomo10.json`` rather than being
hard-coded by positional index.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_EXCLUDED = "conv-26,conv-30,conv-41,conv-42,conv-49"
DEFAULT_SUFFIX = "all_gpt41_chunked_v3_fix1"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_step(command: list[str], env: dict[str, str], *, dry_run: bool) -> None:
    print("\n$ " + " ".join(subprocess.list2cmdline([part]) for part in command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True, env=env)


def conversation_catalog(locomo_path: Path) -> list[tuple[int, str]]:
    data = read_json(locomo_path)
    if not isinstance(data, list) or not data:
        raise ValueError(f"LoCoMo input must be a non-empty JSON list: {locomo_path}")
    catalog: list[tuple[int, str]] = []
    for index, sample in enumerate(data):
        if not isinstance(sample, dict) or not sample.get("sample_id"):
            raise ValueError(f"LoCoMo sample {index} has no sample_id")
        catalog.append((index, str(sample["sample_id"])))
    return catalog


def add_common_openai(command: list[str], args: argparse.Namespace) -> None:
    command.extend(
        [
            "--provider",
            "openai",
            "--base-url",
            args.openai_base_url,
            "--model",
            args.openai_model,
            "--timeout",
            str(args.timeout),
        ]
    )


def judge_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    summary = payload.get("summary") if isinstance(payload, dict) else None
    if isinstance(summary, dict):
        return summary
    if isinstance(payload, dict) and {"count", "correct"} <= payload.keys():
        return {
            key: payload[key]
            for key in ("count", "correct", "llm_judge_accuracy", "by_category")
            if key in payload
        }
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Steps 1-7 plus the official Memora-compatible judge for all "
            "remaining LoCoMo conversations"
        )
    )
    parser.add_argument("--steps", default=None, help="entity_structured_steps directory")
    parser.add_argument("--locomo", default=None, help="path to data/locomo10.json")
    parser.add_argument("--output-root", default=None, help="parent directory for per-conversation runs")
    parser.add_argument(
        "--conversation-ids",
        default="",
        help="comma-separated IDs to run; empty means all IDs not in --exclude-conversation-ids",
    )
    parser.add_argument("--exclude-conversation-ids", default=DEFAULT_EXCLUDED)
    parser.add_argument("--run-suffix", default=DEFAULT_SUFFIX)
    parser.add_argument("--python", default=sys.executable, help="Python executable used for every step")
    parser.add_argument("--judge-script", default=None, help="Memora-compatible 08_memora_llm_judge.py")
    parser.add_argument("--qwen-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--qwen-model", default="qwen3-32B")
    parser.add_argument("--openai-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--openai-model", default="gpt-4.1-mini-2025-04-14")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--step2-batch-size", type=int, default=4)
    parser.add_argument("--step3-batch-size", type=int, default=8)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-device", default="cuda:0")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-max-length", type=int, default=256)
    parser.add_argument("--cuda-visible-devices", default="2")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop at the first failed conversation; default continues with the next one",
    )
    args = parser.parse_args()

    steps = Path(args.steps).expanduser().resolve() if args.steps else Path(__file__).resolve().parent
    code_root = steps.parent
    locomo_path = Path(args.locomo).expanduser().resolve() if args.locomo else code_root / "data" / "locomo10.json"
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else code_root / "runs" / "entity_structured_v2_remaining5_gpt41"
    )
    judge_script = (
        Path(args.judge_script).expanduser().resolve()
        if args.judge_script
        else code_root / "property_coverage_steps" / "08_memora_llm_judge.py"
    )
    embedding_model = args.embedding_model or str(code_root.parent / "litsearch" / "qwen3-embedding-8B")

    required_steps = [
        "01_prepare_sessions.py",
        "02_partition_theme_episodes.py",
        "03_build_entity_index.py",
        "04_plan_query_hops.py",
        "05_retrieve_hops.py",
        "06_answer_from_hops.py",
        "07_score.py",
    ]
    missing_scripts = [name for name in required_steps if not (steps / name).exists()]
    if missing_scripts:
        raise FileNotFoundError(f"Missing entity-structured step scripts under {steps}: {missing_scripts}")
    if not judge_script.exists():
        raise FileNotFoundError(f"Missing Memora-compatible judge script: {judge_script}")

    catalog = conversation_catalog(locomo_path)
    by_id = {conversation_id: index for index, conversation_id in catalog}
    requested = parse_csv(args.conversation_ids)
    excluded = set(parse_csv(args.exclude_conversation_ids))
    selected_ids = requested or [conversation_id for _, conversation_id in catalog if conversation_id not in excluded]
    unknown = [conversation_id for conversation_id in selected_ids if conversation_id not in by_id]
    if unknown:
        raise ValueError(f"Unknown conversation IDs: {unknown}; available={sorted(by_id)}")
    if not selected_ids:
        raise ValueError("No conversations selected")
    output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    print(f"LoCoMo conversations available: {len(catalog)}")
    print(f"Selected for this batch ({len(selected_ids)}): {', '.join(selected_ids)}")
    print(f"Excluded as already tested: {', '.join(sorted(excluded)) or '(none)'}")
    print(f"Per-conversation output root: {output_root}")

    statuses: list[dict[str, Any]] = []
    for ordinal, conversation_id in enumerate(selected_ids, 1):
        conversation_index = by_id[conversation_id]
        run_dir = output_root / f"entity_structured_v2_{conversation_id}_{args.run_suffix}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 88}\n[{ordinal}/{len(selected_ids)}] {conversation_id} (LoCoMo index {conversation_index})\nrun={run_dir}\n{'=' * 88}")
        status: dict[str, Any] = {
            "conversation_id": conversation_id,
            "conversation_index": conversation_index,
            "run_dir": str(run_dir),
            "status": "pending",
        }
        try:
            # Step 1 intentionally refuses partial directories.  A completed
            # Step 1 is reused; all later steps have their own checkpoints.
            prepare_outputs = [
                run_dir / "01_sessions.json",
                run_dir / "01_questions.json",
                run_dir / "01_gold_DO_NOT_USE_BEFORE_STEP_7.json",
            ]
            any_prepare = any(path.exists() for path in prepare_outputs)
            if not all(path.exists() for path in prepare_outputs):
                if any_prepare:
                    raise RuntimeError(
                        "Step 1 directory is partial. Move it aside or choose a fresh per-conversation run directory."
                    )
                run_step(
                    [
                        args.python,
                        str(steps / "01_prepare_sessions.py"),
                        "--locomo",
                        str(locomo_path),
                        "--out-dir",
                        str(run_dir),
                        "--conversation-index",
                        str(conversation_index),
                        "--questions-per-category",
                        "0",
                        "--query-start",
                        "1",
                        "--query-count",
                        "0",
                        "--include-image-captions",
                        "--extraction-context-radius",
                        "2",
                    ],
                    env,
                    dry_run=args.dry_run,
                )
            else:
                print("Step 1 complete; reusing 01_sessions/01_questions/01_gold checkpoints.")

            run_step(
                [
                    args.python,
                    str(steps / "02_partition_theme_episodes.py"),
                    "--run-dir",
                    str(run_dir),
                    "--batch-size",
                    str(args.step2_batch_size),
                    "--provider",
                    "vllm",
                    "--base-url",
                    args.qwen_base_url,
                    "--model",
                    args.qwen_model,
                    "--timeout",
                    str(args.timeout),
                ],
                env,
                dry_run=args.dry_run,
            )
            run_step(
                [
                    args.python,
                    str(steps / "03_build_entity_index.py"),
                    "--run-dir",
                    str(run_dir),
                    "--batch-size",
                    str(args.step3_batch_size),
                    "--provider",
                    "openai",
                    "--base-url",
                    args.openai_base_url,
                    "--model",
                    args.openai_model,
                    "--timeout",
                    str(args.timeout),
                ],
                env,
                dry_run=args.dry_run,
            )

            plan_file = "04_query_hop_plans_gpt41_all_v1.json"
            retrieval_file = "05_entity_structured_retrievals_gpt41_all_v1.json"
            prediction_file = "06_entity_structured_predictions_gpt41_all_v1.json"
            score_file = "07_entity_structured_scores_gpt41_all_v1.json"
            judge_file = f"08_memora_judge_{args.run_suffix}.json"

            run_step(
                [
                    args.python,
                    str(steps / "04_plan_query_hops.py"),
                    "--run-dir",
                    str(run_dir),
                    "--questions-file",
                    "01_questions.json",
                    "--index-file",
                    "03_entity_index.json",
                    "--output-file",
                    plan_file,
                    "--provider",
                    "openai",
                    "--base-url",
                    args.openai_base_url,
                    "--model",
                    args.openai_model,
                    "--timeout",
                    str(args.timeout),
                ],
                env,
                dry_run=args.dry_run,
            )
            run_step(
                [
                    args.python,
                    str(steps / "05_retrieve_hops.py"),
                    "--run-dir",
                    str(run_dir),
                    "--plan-file",
                    plan_file,
                    "--output-file",
                    retrieval_file,
                    "--top-k-per-hop",
                    "5",
                    "--lexical-fallback-top-k",
                    "12",
                    "--dense-fallback-top-k",
                    "8",
                    "--minimum-score",
                    "0.35",
                    "--property-semantic-threshold",
                    "0.78",
                    "--value-semantic-threshold",
                    "0.86",
                    "--embedding-model",
                    embedding_model,
                    "--embedding-device",
                    args.embedding_device,
                    "--embedding-batch-size",
                    str(args.embedding_batch_size),
                    "--embedding-max-length",
                    str(args.embedding_max_length),
                    "--provider",
                    "openai",
                    "--base-url",
                    args.openai_base_url,
                    "--model",
                    args.openai_model,
                    "--timeout",
                    str(args.timeout),
                ],
                env,
                dry_run=args.dry_run,
            )
            run_step(
                [
                    args.python,
                    str(steps / "06_answer_from_hops.py"),
                    "--run-dir",
                    str(run_dir),
                    "--retrieval-file",
                    retrieval_file,
                    "--output-file",
                    prediction_file,
                    "--gold-file",
                    "01_gold_DO_NOT_USE_BEFORE_STEP_7.json",
                    "--provider",
                    "openai",
                    "--base-url",
                    args.openai_base_url,
                    "--model",
                    args.openai_model,
                    "--timeout",
                    str(args.timeout),
                ],
                env,
                dry_run=args.dry_run,
            )
            run_step(
                [
                    args.python,
                    str(steps / "07_score.py"),
                    "--run-dir",
                    str(run_dir),
                    "--gold-file",
                    "01_gold_DO_NOT_USE_BEFORE_STEP_7.json",
                    "--retrieval-file",
                    retrieval_file,
                    "--predictions-file",
                    prediction_file,
                    "--output-file",
                    score_file,
                ],
                env,
                dry_run=args.dry_run,
            )
            run_step(
                [
                    args.python,
                    str(judge_script),
                    "--run-dir",
                    str(run_dir),
                    "--input-file",
                    prediction_file,
                    "--output-file",
                    judge_file,
                    "--base-url",
                    args.openai_base_url,
                    "--model",
                    args.judge_model,
                    "--timeout",
                    str(args.timeout),
                    "--retries",
                    str(args.retries),
                ],
                env,
                dry_run=args.dry_run,
            )
            status["status"] = "completed"
            status["judge_file"] = str(run_dir / judge_file)
            status["judge_summary"] = judge_summary(run_dir / judge_file)
        except subprocess.CalledProcessError as error:
            status["status"] = "failed"
            status["error"] = f"step exited with code {error.returncode}: {error.cmd}"
            print(f"FAILED {conversation_id}: {status['error']}", file=sys.stderr)
            if args.fail_fast:
                statuses.append(status)
                break
        except Exception as error:
            status["status"] = "failed"
            status["error"] = f"{type(error).__name__}: {error}"
            print(f"FAILED {conversation_id}: {status['error']}", file=sys.stderr)
            if args.fail_fast:
                statuses.append(status)
                break
        statuses.append(status)

    summary_path = output_root / "00_remaining_conversations_summary.json"
    summary = {
        "schema_version": "ensi-locomo-entity-structured-batch-run-v1",
        "locomo_file": str(locomo_path),
        "selected_conversation_ids": selected_ids,
        "excluded_conversation_ids": sorted(excluded),
        "statuses": statuses,
        "completed": sum(item["status"] == "completed" for item in statuses),
        "failed": sum(item["status"] == "failed" for item in statuses),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved batch summary to {summary_path}")
    for item in statuses:
        judge = item.get("judge_summary") or {}
        if judge:
            print(
                f"{item['conversation_id']}: {judge.get('correct')}/{judge.get('count')} "
                f"({judge.get('llm_judge_accuracy')})"
            )
        else:
            print(f"{item['conversation_id']}: {item['status']}")
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
