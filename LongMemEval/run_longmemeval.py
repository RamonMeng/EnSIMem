#!/usr/bin/env python3
"""Sequential EnSI-Memory runner for LongMemEval-S-cleaned.

Each dataset item is processed as one isolated query:

    adapt sessions -> partition -> entity index -> query plan -> retrieval
    -> answer -> persist all artifacts

The next item starts only after the current item has completed.  The required
EnSI modules are bundled in this directory, so the directory can be copied to
a server as a self-contained program folder.  ``--original-dir`` remains as an
optional override for comparing against another read-only EnSI source tree.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

from longmemeval_adapter import (
    DATASET_FILENAME,
    adapt_instance,
    load_instances,
    public_instance_metadata,
    read_json,
    write_json,
)
from longmemeval_generation import generate_answer
from memora_evaluation import (
    build_evaluation_metadata,
    judge_prediction,
    summarize_judges,
)


HERE = Path(__file__).resolve().parent
_DATASET_CANDIDATES = (
    HERE.parent / "data" / DATASET_FILENAME,
    HERE.parent / DATASET_FILENAME,
)
DEFAULT_DATASET = next(
    (path for path in _DATASET_CANDIDATES if path.exists()),
    _DATASET_CANDIDATES[0],
)
_EMBEDDING_CANDIDATES = (
    HERE.parent.parent / "litsearch" / "qwen3-embedding-8B",
    HERE.parent / "litsearch" / "qwen3-embedding-8B",
)
DEFAULT_EMBEDDING_MODEL = next(
    (str(path) for path in _EMBEDDING_CANDIDATES if path.exists()),
    "../../litsearch/qwen3-embedding-8B",
)
DEFAULT_ORIGINAL_DIR = HERE
DEFAULT_OUTPUT_DIR = HERE / "runs" / "longmemeval_s_cleaned"
SESSION_CACHE_SCHEMA = "longmemeval-ensi-session-cache-v1"
BASE_QUESTION_TYPES = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "knowledge-update",
    "temporal-reasoning",
)


def _load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load EnSI module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_original_stages(original_dir: Path) -> dict[str, ModuleType]:
    """Load reusable EnSI functions without modifying the source directory."""
    original_dir = original_dir.resolve()
    required = [
        "_common.py",
        "llm.py",
        "online_efficiency.py",
        "prompts.py",
        "02_partition_theme_episodes.py",
        "03_build_entity_index.py",
        "04_plan_query_hops.py",
        "05_retrieve_hops.py",
    ]
    missing = [name for name in required if not (original_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"original EnSI directory is missing: {', '.join(missing)}"
        )
    # The original numbered scripts use direct imports such as ``from _common``.
    # Put only the read-only source directory on sys.path for those imports.
    if str(original_dir) not in sys.path:
        sys.path.insert(0, str(original_dir))
    modules = {
        "common": _load_module("ensi_original_common", original_dir / "_common.py"),
        "partition": _load_module(
            "ensi_original_partition", original_dir / "02_partition_theme_episodes.py"
        ),
        "index": _load_module(
            "ensi_original_index", original_dir / "03_build_entity_index.py"
        ),
        "planner": _load_module(
            "ensi_original_planner", original_dir / "04_plan_query_hops.py"
        ),
        "retrieval": _load_module(
            "ensi_original_retrieval", original_dir / "05_retrieve_hops.py"
        ),
    }
    return modules


def _add_arguments(parser: argparse.ArgumentParser, common: ModuleType) -> None:
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument(
        "--original-dir",
        default=str(DEFAULT_ORIGINAL_DIR),
        help="Bundled EnSI source directory; may be overridden with a read-only source tree",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of dataset items to process; omit to process through the end",
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=None,
        help=(
            "Select this many items from each of the six base LongMemEval "
            "question types; abstention items are excluded by default"
        ),
    )
    parser.add_argument(
        "--question-type",
        choices=BASE_QUESTION_TYPES,
        default=None,
        help="Process only one LongMemEval question type (use with --limit/--start)",
    )
    parser.add_argument(
        "--exclude-question-id",
        action="append",
        default=[],
        help="Exclude a question ID from selection; repeat for multiple IDs",
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=42,
        help="Random seed used by --per-category (default: 42)",
    )
    parser.add_argument(
        "--include-abstention",
        action="store_true",
        help="Allow *_abs abstention items in --per-category selection",
    )
    parser.add_argument(
        "--stop-after",
        choices=("index", "judge"),
        default="judge",
        help="Stop after Step 3 entity indexing, or continue through the full pipeline (default: judge)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess query artifacts that already contain a final prediction",
    )
    parser.add_argument(
        "--answer-only",
        action="store_true",
        help=(
            "Reuse existing Steps 1-5 artifacts and regenerate only Step 6 plus "
            "the Step-7 judge; useful for comparing answer-generation methods"
        ),
    )
    parser.add_argument("--top-k-per-hop", type=int, default=5)
    parser.add_argument("--lexical-fallback-top-k", type=int, default=12)
    parser.add_argument("--dense-fallback-top-k", type=int, default=8)
    parser.add_argument("--minimum-score", type=float, default=0.35)
    parser.add_argument("--property-semantic-threshold", type=float, default=0.78)
    parser.add_argument("--value-semantic-threshold", type=float, default=0.86)
    parser.add_argument("--diagnostic-limit", type=int, default=25)
    parser.add_argument(
        "--max-reasoning-episodes",
        type=int,
        default=20,
        help=(
            "Maximum episodes used by the Step-6 evidence-first reranker for "
            "non-exhaustive questions (default: 20)"
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Same embedding model argument used by the original Step 5",
    )
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-max-length", type=int, default=256)
    parser.add_argument("--rebuild-embeddings", action="store_true")
    parser.add_argument(
        "--partition-workers",
        type=int,
        default=4,
        help="Concurrent Step-2 local vLLM requests per query (default: 4)",
    )
    parser.add_argument(
        "--partition-batch-size",
        type=int,
        default=4,
        help=(
            "Independent sessions per Step-2 local vLLM request; the original "
            "partition prompt is preserved inside each batch (default: 4)"
        ),
    )
    parser.add_argument(
        "--index-workers",
        type=int,
        default=4,
        help="Concurrent Step-3 GPT API requests per query (default: 4)",
    )
    parser.add_argument(
        "--index-batch-size",
        type=int,
        default=4,
        help=(
            "Independent episodes per Step-3 GPT request; the single-episode "
            "LoCoMo extraction prompt is preserved inside each batch (default: 4)"
        ),
    )
    parser.add_argument(
        "--gpt-base-url",
        default="https://api.openai.com/v1",
        help="OpenAI-compatible API used from Step 3 through Step 6",
    )
    parser.add_argument(
        "--gpt-model",
        default="gpt-4.1-mini",
        help="GPT model used from Step 3 through Step 6",
    )
    parser.add_argument(
        "--gpt-api-key",
        default=None,
        help="API key for Step 3-6; defaults to OPENAI_API_KEY",
    )
    # Batched extraction requests contain several complete prompts and need a
    # larger socket timeout than the original single-episode call.
    parser.add_argument("--gpt-timeout", type=int, default=600)
    parser.add_argument(
        "--judge-base-url",
        default="https://api.openai.com/v1",
        help="OpenAI-compatible API used by the Memora-style Step 7 judge",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-mini-2024-07-18",
        help="Judge model used by the existing Memora-style evaluator",
    )
    parser.add_argument(
        "--judge-api-key",
        default=None,
        help="API key for Step 7; defaults to OPENAI_API_KEY",
    )
    parser.add_argument("--judge-timeout", type=int, default=900)
    # These existing arguments configure Step 2 only.  Keeping them separate
    # lets partitioning remain on the local vLLM endpoint while all later LLM
    # stages use the requested GPT API.
    common.add_vllm_arguments(parser)
    # Keep the original --timeout override, but use a batch-safe default for
    # this runner. The standalone original scripts retain their old default.
    parser.set_defaults(timeout=600)


def _make_gpt_client(common: ModuleType, args: argparse.Namespace) -> Any:
    gpt_args = argparse.Namespace(
        provider="openai",
        base_url=args.gpt_base_url,
        model=args.gpt_model,
        api_key=args.gpt_api_key,
        timeout=args.gpt_timeout,
    )
    return common.make_client(gpt_args)


def _make_judge_client(common: ModuleType, args: argparse.Namespace) -> Any:
    judge_args = argparse.Namespace(
        provider="openai",
        base_url=args.judge_base_url,
        model=args.judge_model,
        api_key=args.judge_api_key,
        timeout=args.judge_timeout,
    )
    return common.make_client(judge_args)


def _validate_args(args: argparse.Namespace) -> None:
    if args.start < 0:
        raise ValueError("--start must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided")
    if args.per_category is not None and args.per_category <= 0:
        raise ValueError("--per-category must be positive when provided")
    if args.per_category is not None and (args.start != 0 or args.limit is not None):
        raise ValueError("--per-category cannot be combined with --start or --limit")
    if args.per_category is not None and args.question_type is not None:
        raise ValueError("--per-category cannot be combined with --question-type")
    if args.include_abstention and args.per_category is None:
        raise ValueError("--include-abstention requires --per-category")
    for name in (
        "top_k_per_hop",
        "lexical_fallback_top_k",
        "dense_fallback_top_k",
        "embedding_batch_size",
        "embedding_max_length",
        "max_reasoning_episodes",
        "partition_workers",
        "partition_batch_size",
        "index_workers",
        "index_batch_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "minimum_score",
        "property_semantic_threshold",
        "value_semantic_threshold",
    ):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")


def _query_dir(output_dir: Path, question_id: str) -> Path:
    # The adapter's IDs are hexadecimal in the released dataset, but this
    # keeps artifacts safe if a future copy contains punctuation or slashes.
    safe = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in question_id
    ).strip("._") or "unknown_query"
    return output_dir / "queries" / safe


def _completed_question_ids(output_dir: Path) -> set[str]:
    """Return IDs with a final answer already present in this run directory."""
    completed: set[str] = set()
    queries_dir = output_dir / "queries"
    if not queries_dir.exists():
        return completed
    for prediction_path in queries_dir.glob("*/06_prediction.json"):
        try:
            prediction = read_json(prediction_path)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        question_id = str(prediction.get("question_id", "")).strip()
        if question_id:
            completed.add(question_id)
    return completed


def _session_fingerprint(session: dict[str, Any], args: argparse.Namespace) -> str:
    """Hash only source/session data that affects partitioning or extraction.

    ``conversation_id`` is intentionally excluded: the same LongMemEval
    source session can appear in more than one isolated query, while the
    query-specific episode IDs are rebound when a cached index is reused.
    """
    rendered_turns = []
    for turn in session.get("turns", []):
        rendered_turns.append(
            {
                "dia_id": str(turn.get("dia_id", "")),
                "speaker": str(turn.get("speaker", "")),
                "text": str(turn.get("text", "")),
                "blip_caption": turn.get("blip_caption", ""),
                "image_caption": turn.get("image_caption", ""),
                "caption": turn.get("caption", ""),
                "query": turn.get("query", ""),
                "img_url": turn.get("img_url", ""),
            }
        )
    fingerprint_input = {
        "session_id": str(session.get("session_id", "")),
        "observed_at": str(session.get("observed_at", "")),
        "include_image_captions": bool(session.get("include_image_captions")),
        "turns": rendered_turns,
        "partition_model": str(args.model),
        "index_model": str(args.gpt_model),
        # Change this when the model-facing extraction policy changes.
        "index_prompt_revision": "locomo-index-user-v1",
    }
    encoded = json.dumps(
        fingerprint_input, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _session_cache_path(cache_dir: Path, fingerprint: str) -> Path:
    return cache_dir / f"{fingerprint}.json"


def _read_cache_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != SESSION_CACHE_SCHEMA:
        return None
    if not str(payload.get("session_fingerprint", "")):
        return None
    return payload


def _rebind_cached_record(record: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
    """Move a query-independent record onto this query's episode identity."""
    rebound = dict(record)
    old_episode_id = str(record.get("episode_id", ""))
    new_episode_id = str(episode["episode_id"])
    rebound["episode_id"] = new_episode_id
    rebound["conversation_id"] = episode["conversation_id"]
    old_index_id = str(record.get("index_id", ""))
    if old_index_id and old_episode_id:
        rebound["index_id"] = old_index_id.replace(old_episode_id, new_episode_id, 1)
    provenance = dict(record.get("provenance") or {})
    provenance.update(
        {
            "session_id": episode["session_id"],
            "episode_id": new_episode_id,
            "episode_theme": episode["theme"],
            "extraction_unit": "complete_theme_episode",
        }
    )
    rebound["provenance"] = provenance
    return rebound


def _materialize_cached_partition(
    session: dict[str, Any],
    payload: dict[str, Any],
    common: ModuleType,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]], bool] | None:
    """Validate and rebuild immutable episodes from a cached partition."""
    if payload.get("partition_model") != args.model or payload.get("index_model") != args.gpt_model:
        return None
    partition = payload.get("partition")
    if not isinstance(partition, dict) or not isinstance(partition.get("segments"), list):
        return None
    try:
        episodes = common.construct_theme_episodes(session, partition["segments"])
    except (TypeError, ValueError, KeyError):
        return None
    raw_partition = {
        "segments": partition["segments"],
        "fallback_used": bool(partition.get("fallback_used", False)),
    }
    if partition.get("fallback_reason"):
        raw_partition["fallback_reason"] = str(partition["fallback_reason"])
    return raw_partition, episodes, bool(payload.get("repaired", False))


def _make_session_cache_payload(
    session: dict[str, Any],
    args: argparse.Namespace,
    raw_partition: dict[str, Any],
    repaired: bool,
    records_by_episode_number: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "schema_version": SESSION_CACHE_SCHEMA,
        "session_fingerprint": _session_fingerprint(session, args),
        "session_id": session["session_id"],
        "observed_at": session["observed_at"],
        "partition_model": args.model,
        "index_model": args.gpt_model,
        "repaired": bool(repaired),
        "partition": {
            "segments": raw_partition["segments"],
            "fallback_used": bool(raw_partition.get("fallback_used", False)),
            "fallback_reason": str(raw_partition.get("fallback_reason", "")),
        },
        "records_by_episode_number": records_by_episode_number,
    }


def _cache_records_for_episode(
    payload: dict[str, Any], episode: dict[str, Any]
) -> list[dict[str, Any]] | None:
    mapping = payload.get("records_by_episode_number")
    if not isinstance(mapping, dict):
        return None
    key = str(episode["episode_number"])
    if key not in mapping or not isinstance(mapping[key], list):
        return None
    return [
        _rebind_cached_record(record, episode)
        for record in mapping[key]
        if isinstance(record, dict)
    ]


def _build_session_cache_index(
    output_dir: Path,
    args: argparse.Namespace,
    common: ModuleType,
) -> tuple[Path, dict[str, dict[str, Any]]]:
    """Load cache files and seed them from completed query artifacts.

    Seeding is deliberately limited to completed queries, so an interrupted
    Step-3 checkpoint can never be mistaken for an empty finished episode.
    """
    cache_dir = output_dir / "session_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_index: dict[str, dict[str, Any]] = {}
    for path in cache_dir.glob("*.json"):
        payload = _read_cache_payload(path)
        if payload:
            cache_index[str(payload["session_fingerprint"])] = payload

    manifest_model = args.model
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        try:
            manifest = read_json(manifest_path)
            manifest_model = str(manifest.get("options", {}).get("model", args.model))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    queries_dir = output_dir / "queries"
    if not queries_dir.exists():
        return cache_dir, cache_index
    for query_dir in sorted(path for path in queries_dir.iterdir() if path.is_dir()):
        if not (query_dir / "06_prediction.json").exists():
            continue
        sessions_path = query_dir / "01_sessions.json"
        partition_path = query_dir / "02_theme_episodes.json"
        index_path = query_dir / "03_entity_index.json"
        if not (sessions_path.exists() and partition_path.exists() and index_path.exists()):
            continue
        try:
            source = read_json(sessions_path)
            partition_source = read_json(partition_path)
            index_source = read_json(index_path)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if not isinstance(source, dict) or not isinstance(index_source, dict):
            continue
        if str(index_source.get("model", "")) != str(args.gpt_model):
            continue
        sessions = source.get("sessions")
        partitions = partition_source.get("partitions") if isinstance(partition_source, dict) else None
        all_records = index_source.get("records")
        if not isinstance(sessions, list) or not isinstance(partitions, list) or not isinstance(all_records, list):
            continue
        partition_by_session = {
            str(item.get("session_id", "")): item
            for item in partitions
            if isinstance(item, dict)
        }
        record_by_episode_id: dict[str, list[dict[str, Any]]] = {}
        for record in all_records:
            if not isinstance(record, dict):
                continue
            episode_id = str(record.get("episode_id", ""))
            if episode_id:
                record_by_episode_id.setdefault(episode_id, []).append(record)
        for session in sessions:
            if not isinstance(session, dict):
                continue
            session_id = str(session.get("session_id", ""))
            partition = partition_by_session.get(session_id)
            if not partition or str(manifest_model) != str(args.model):
                continue
            try:
                episodes = common.construct_theme_episodes(session, partition.get("segments"))
            except (TypeError, ValueError, KeyError):
                continue
            records_by_number = {
                str(episode["episode_number"]): record_by_episode_id.get(
                    str(episode["episode_id"]), []
                )
                for episode in episodes
            }
            cache_session = _make_session_cache_payload(
                session,
                args,
                partition,
                bool(partition.get("repaired", False)),
                records_by_number,
            )
            fingerprint = str(cache_session["session_fingerprint"])
            if fingerprint not in cache_index:
                cache_path = _session_cache_path(cache_dir, fingerprint)
                write_json(cache_path, cache_session)
                cache_index[fingerprint] = cache_session
    return cache_dir, cache_index


def _parallel_stage(
    *,
    items: list[dict[str, Any]],
    client: Any,
    worker: Any,
    max_workers: int,
    scope_prefix: str,
    on_success: Any | None = None,
) -> tuple[list[tuple[dict[str, Any], Any]], list[dict[str, Any]]]:
    """Run independent per-session/per-episode calls concurrently.

    Query processing remains sequential in ``main``.  This helper only
    overlaps independent calls inside the current query.  Each worker owns a
    thread-local usage scope so retry/call accounting remains correct when the
    shared HTTP client is used concurrently.
    """
    if not items:
        return [], []

    def invoke(item: dict[str, Any]) -> tuple[dict[str, Any], Any, dict[str, Any]]:
        item_label = str(
            item.get("session_id")
            or item.get("episode_id")
            or item.get("batch_id")
            or item.get("conversation_id")
            or "item"
        )
        client.begin_usage_scope(f"{scope_prefix}:{item_label}")
        try:
            value = worker(client, item)
        finally:
            usage = client.end_usage_scope()
        return item, value, usage

    successes: list[tuple[dict[str, Any], Any]] = []
    usages: list[dict[str, Any]] = []
    failures: list[tuple[dict[str, Any], Exception]] = []
    worker_count = min(max_workers, len(items))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(invoke, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                completed_item, value, usage = future.result()
                successes.append((completed_item, value))
                usages.append(usage)
                if on_success is not None:
                    on_success(completed_item, value)
            except Exception as error:
                failures.append((item, error))

    if failures:
        details = "; ".join(
            f"{str(item.get('session_id') or item.get('episode_id') or item.get('batch_id') or 'item')}: {error}"
            for item, error in failures
        )
        raise RuntimeError(
            f"{scope_prefix} failures; successful calls were not persisted: {details}"
        )
    return successes, usages


def _combine_usage(label: str, scopes: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-worker and main-thread usage summaries."""
    return {
        "scope_label": label,
        "llm_call_count": sum(int(scope.get("llm_call_count", 0)) for scope in scopes),
        "retry_count": sum(int(scope.get("retry_count", 0)) for scope in scopes),
    }


def _select_per_category(
    instances: list[dict[str, Any]],
    per_category: int,
    seed: int,
    completed_ids: set[str],
    include_abstention: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select a reproducible balanced sample, retaining completed queries first."""
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for question_type in BASE_QUESTION_TYPES:
        candidates = [
            (dataset_index, item)
            for dataset_index, item in enumerate(instances)
            if str(item.get("question_type", "")) == question_type
            and (
                include_abstention
                or not str(item.get("question_id", "")).endswith("_abs")
            )
        ]
        if len(candidates) < per_category:
            raise ValueError(
                f"question type {question_type!r} has only {len(candidates)} "
                f"eligible items; cannot select {per_category}"
            )

        completed = [
            pair
            for pair in candidates
            if str(pair[1].get("question_id", "")) in completed_ids
        ]
        chosen = completed[:per_category]
        chosen_ids = {str(item.get("question_id", "")) for _, item in chosen}
        remaining = [
            pair
            for pair in candidates
            if str(pair[1].get("question_id", "")) not in chosen_ids
        ]
        needed = per_category - len(chosen)
        if needed:
            chosen.extend(rng.sample(remaining, needed))
        chosen.sort(key=lambda pair: pair[0])

        for dataset_index, item in chosen:
            question_id = str(item["question_id"])
            selected.append(item)
            manifest.append(
                {
                    "selection_position": len(manifest) + 1,
                    "dataset_index": dataset_index,
                    "question_id": question_id,
                    "question_type": question_type,
                    "is_abstention": question_id.endswith("_abs"),
                    "already_completed": question_id in completed_ids,
                }
            )

    return selected, manifest


def _write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    dataset_path: Path,
    instance_count: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    options = vars(args).copy()
    options.pop("api_key", None)
    options.pop("gpt_api_key", None)
    write_json(
        output_dir / "run_manifest.json",
        {
            "schema_version": "longmemeval-ensi-sequential-run-v2",
            "dataset_filename": DATASET_FILENAME,
            "dataset_path": str(dataset_path.resolve()),
            "original_program_dir": str(Path(args.original_dir).resolve()),
            "dataset_item_count": instance_count,
            "workflow": [
                "longmemeval_adapter",
                "original_ensi_step_2_partition",
                "original_ensi_step_3_entity_index",
                "original_ensi_step_4_query_plan",
                "original_ensi_step_5_retrieval",
                "longmemeval_answer_generation",
                "memora_style_longmemeval_judge",
            ],
            "query_isolation": "one complete query pipeline before the next query",
            "gold_isolation": (
                "gold answer is sent only to the Step-7 judge after generation; "
                "has_answer labels are never sent to an LLM"
            ),
            "options": options,
        },
    )


def _write_prediction_indexes(output_dir: Path, dataset_order: list[str]) -> None:
    """Write official-style JSONL plus a richer human-readable index."""
    rows_by_id: dict[str, dict[str, Any]] = {}
    for question_id in dataset_order:
        prediction_path = _query_dir(output_dir, question_id) / "06_prediction.json"
        if not prediction_path.exists():
            continue
        row = read_json(prediction_path)
        rows_by_id[question_id] = row
    rows = [rows_by_id[question_id] for question_id in dataset_order if question_id in rows_by_id]
    official_rows = [
        {"question_id": row["question_id"], "hypothesis": row["hypothesis"]}
        for row in rows
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "predictions.jsonl"
    temporary = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in official_rows),
        encoding="utf-8",
    )
    temporary.replace(jsonl_path)
    write_json(
        output_dir / "predictions_with_metadata.json",
        {
            "schema_version": "longmemeval-ensi-predictions-v1",
            "prediction_count": len(rows),
            "predictions": rows,
        },
    )


def _write_evaluation_summary(
    output_dir: Path, dataset_order: list[str], judge_model: str
) -> None:
    rows: list[dict[str, Any]] = []
    for question_id in dataset_order:
        prediction_path = _query_dir(output_dir, question_id) / "06_prediction.json"
        if not prediction_path.exists():
            continue
        prediction = read_json(prediction_path)
        judge = prediction.get("judge")
        if not isinstance(judge, dict):
            continue
        rows.append(
            {
                "question_id": question_id,
                "question_type": str(prediction.get("question_type", "")),
                "judge": judge,
                "evaluation": prediction.get("evaluation", {}),
            }
        )
    summary = summarize_judges(rows, judge_model)
    summary["prediction_count"] = len(rows)
    write_json(output_dir / "07_results_and_context_summary.json", summary)


def _judge_and_attach(
    *,
    judge_client: Any,
    sample: dict[str, Any],
    result: dict[str, Any],
    query_dir: Path,
    judge_model: str,
    existing_judge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate/attach Step 7 and gold/retrieval diagnostics.

    An old ``07_judge.json`` can be reused without another API call; this is
    useful when upgrading a run that already has a judge result but predates
    the gold-dialog diagnostics.
    """
    question_id = str(sample["question_id"])
    if isinstance(existing_judge, dict) and isinstance(existing_judge.get("judge"), dict):
        judge = existing_judge["judge"]
        usage = existing_judge.get("judge_usage", {})
    else:
        judge_client.begin_usage_scope(question_id)
        try:
            judge = judge_prediction(
                judge_client,
                sample,
                str(result.get("hypothesis", result.get("prediction", ""))),
            )
        finally:
            usage = judge_client.end_usage_scope()

    retrieval_path = query_dir / "05_retrieval.json"
    if retrieval_path.exists():
        retrieval = read_json(retrieval_path)
    else:
        # Preserve gold-dialog diagnostics when the reused Step 5 artifact
        # came from the legacy runner under its wrapper filename.
        legacy_path = query_dir / "05_hop_retrievals.json"
        legacy_payload = read_json(legacy_path) if legacy_path.exists() else {}
        retrieval_rows = (
            legacy_payload.get("retrievals", [])
            if isinstance(legacy_payload, dict)
            else []
        )
        retrieval = (
            retrieval_rows[0]
            if isinstance(retrieval_rows, list)
            and len(retrieval_rows) == 1
            and isinstance(retrieval_rows[0], dict)
            else {}
        )
    evaluation = build_evaluation_metadata(
        sample,
        retrieval,
        result.get("answer_context_episode_order", []),
    )
    judged = {
        "schema_version": "longmemeval-memora-style-judge-v1",
        "question_id": question_id,
        "judge_model": judge_model,
        "gold_answer": evaluation["gold_answer"],
        "gold_answer_session_ids": evaluation["gold_answer_session_ids"],
        "gold_dialog_retrieval": {
            "retrieved_gold_session_ids": evaluation["retrieved_gold_session_ids"],
            "missing_gold_session_ids_after_step5": evaluation[
                "missing_gold_session_ids_after_step5"
            ],
            "all_gold_sessions_retrieved": evaluation[
                "all_gold_sessions_retrieved"
            ],
        },
        "evaluation": evaluation,
        "judge": judge,
        "judge_usage": usage,
    }
    write_json(query_dir / "07_judge.json", judged)
    result["judge"] = judge
    result["judge_model"] = judge_model
    result["judge_usage"] = usage
    result["evaluation"] = evaluation
    write_json(query_dir / "06_prediction.json", result)
    return result


def _compact_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer_target": plan["answer_target"],
        "reasoning_type": plan["reasoning_type"],
        "retrieval_scope": plan.get("retrieval_scope", "point"),
        "required_properties": plan.get("required_properties", []),
        "hops": [
            {
                "hop_id": hop["hop_id"],
                "purpose": hop.get("purpose", ""),
                "resolved_anchor": hop.get("resolved_anchor", hop.get("anchor", {})),
                "depends_on": hop.get("depends_on", []),
                "bridge_value": hop.get("bridge_value", ""),
                "selected_episode_ids": hop.get("selected_episode_ids", []),
            }
            for hop in plan.get("hop_results", [])
        ],
    }


def process_one(
    *,
    instance: dict[str, Any],
    query_dir: Path,
    partition_client: Any,
    gpt_client: Any,
    model: Any,
    stages: dict[str, ModuleType],
    args: argparse.Namespace,
    session_cache_dir: Path,
    session_cache_index: dict[str, dict[str, Any]],
    stop_after: str = "judge",
) -> dict[str, Any]:
    """Run all stages for exactly one LongMemEval item."""
    # Import after load_original_stages has placed the selected original EnSI
    # directory on sys.path; longmemeval_retrieval intentionally reuses the
    # original _common helpers.
    from longmemeval_retrieval import retrieve_one

    common = stages["common"]
    partition_stage = stages["partition"]
    index_stage = stages["index"]
    planner_stage = stages["planner"]
    retrieval_stage = stages["retrieval"]
    adapted = adapt_instance(instance)
    query_id = adapted["question_id"]
    query_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    stage_durations: dict[str, float] = {}
    parallel_usage: dict[str, list[dict[str, Any]]] = {
        "step2_local_vllm": [],
        "step3_to_step6_openai": [],
    }

    # Step 1: write only the adapted source shape.  Gold answer is deliberately
    # excluded; audit-only ``has_answer`` fields remain inside the source turns
    # but are not rendered by the original EnSI prompts.
    step_started = time.perf_counter()
    write_json(
        query_dir / "01_sessions.json",
        {
            "schema_version": "longmemeval-ensi-step1-adapted-sessions-v1",
            "conversation_id": query_id,
            "question_id": query_id,
            "question_type": adapted["question_type"],
            "question": adapted["question"],
            "question_date": adapted["question_date"],
            "source_session_count": adapted["source_session_count"],
            "source_turn_count": adapted["source_turn_count"],
            "sessions": adapted["sessions"],
        },
    )
    stage_durations["step1_prepare"] = time.perf_counter() - step_started

    # Step 2 remains on the original local vLLM client.  The original method is
    # reused unchanged, but completed source sessions are reused across query
    # contexts. Only cache misses are sent to vLLM.
    step_started = time.perf_counter()
    partitions: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    partition_by_session_id: dict[str, tuple[dict[str, Any], list[dict[str, Any]], bool]] = {}
    cache_payload_by_session_id: dict[str, dict[str, Any]] = {}
    pending_sessions: list[dict[str, Any]] = []
    for session in adapted["sessions"]:
        fingerprint = _session_fingerprint(session, args)
        payload = session_cache_index.get(fingerprint)
        materialized = (
            _materialize_cached_partition(session, payload, common, args)
            if payload is not None
            else None
        )
        if materialized is None:
            pending_sessions.append(session)
            continue
        partition_by_session_id[str(session["session_id"])] = materialized
        cache_payload_by_session_id[str(session["session_id"])] = payload
        print(
            f"    partition cache {session['session_id']}: "
            f"{len(materialized[1])} episode(s)"
        )
    partition_batches = [
        {
            "batch_id": f"{query_id}:step2_batch_{offset // args.partition_batch_size + 1:04d}",
            "sessions": pending_sessions[offset : offset + args.partition_batch_size],
        }
        for offset in range(0, len(pending_sessions), args.partition_batch_size)
    ]

    def partition_worker(client: Any, batch: dict[str, Any]) -> Any:
        batch_sessions = batch["sessions"]
        batch_partitioner = getattr(partition_stage, "partition_batch", None)
        if callable(batch_partitioner):
            return batch_partitioner(client, batch_sessions)
        # Keep compatibility with an explicitly supplied older read-only EnSI
        # source tree that has no batched helper.
        return [
            (session, partition_stage.partition_one(client, session))
            for session in batch_sessions
        ]

    partition_batch_results, partition_usages = _parallel_stage(
        items=partition_batches,
        client=partition_client,
        worker=partition_worker,
        max_workers=args.partition_workers,
        scope_prefix=f"{query_id}:step2",
    )
    parallel_usage["step2_local_vllm"].extend(partition_usages)
    for _batch, partition_rows in partition_batch_results:
        for session, partition_result in partition_rows:
            partition_by_session_id[str(session["session_id"])] = partition_result
    session_order = {
        str(session["session_id"]): position
        for position, session in enumerate(adapted["sessions"])
    }
    for session in adapted["sessions"]:
        raw_partition, session_episodes, repaired = partition_by_session_id[
            str(session["session_id"])
        ]
        cache_hit = str(session["session_id"]) in cache_payload_by_session_id
        partitions.append(
            {
                "session_id": session["session_id"],
                "observed_at": session["observed_at"],
                "repaired": repaired,
                "fallback_used": bool(raw_partition.get("fallback_used", False)),
                "segments": raw_partition["segments"],
                "cache_hit": cache_hit,
            }
        )
        episodes.extend(session_episodes)
        if not cache_hit:
            print(
                f"    partition {session_order[str(session['session_id'])] + 1}/"
                f"{len(adapted['sessions'])}: "
                f"{session['session_id']} -> {len(session_episodes)} episode(s)"
            )
    write_json(
        query_dir / "02_theme_episodes.json",
        {
            "schema_version": "longmemeval-ensi-step2-theme-episodes-v1",
            "conversation_id": query_id,
            "partitions": partitions,
            "episodes": episodes,
        },
    )
    stage_durations["step2_partition"] = time.perf_counter() - step_started

    # Step 3 onward uses the dedicated GPT API client (gpt-4.1-mini by default).
    # The index keeps the exact LoCoMo single-episode extraction/postprocessing
    # path, but independent episode requests are grouped into bounded batches.
    # Cached records are query-independent; episode IDs are rebound to this
    # query before planning/retrieval.
    step_started = time.perf_counter()
    records_by_episode_id: dict[str, list[dict[str, Any]]] = {}
    pending_episodes: list[dict[str, Any]] = []
    cached_episode_count = 0
    for episode in episodes:
        payload = cache_payload_by_session_id.get(str(episode["session_id"]))
        cached_records = (
            _cache_records_for_episode(payload, episode)
            if payload is not None
            else None
        )
        if cached_records is None:
            pending_episodes.append(episode)
            continue
        records_by_episode_id[str(episode["episode_id"])] = cached_records
        cached_episode_count += 1
        print(
            f"    index cache {episode['episode_id']} -> "
            f"{len(cached_records)} record(s)"
        )

    index_batches = [
        {
            "batch_id": f"{query_id}:step3_batch_{offset // args.index_batch_size + 1:04d}",
            "episodes": pending_episodes[offset : offset + args.index_batch_size],
        }
        for offset in range(0, len(pending_episodes), args.index_batch_size)
    ]

    def index_worker(client: Any, batch: dict[str, Any]) -> Any:
        batch_episodes = batch["episodes"]
        batch_extractor = getattr(index_stage, "extract_episode_batch", None)
        if callable(batch_extractor):
            return batch_extractor(client, batch_episodes)
        # Keep compatibility with an explicitly supplied older read-only EnSI
        # source tree that has no batched helper.
        return [
            (episode, index_stage.extract_episode(client, episode))
            for episode in batch_episodes
        ]

    completed_index_batches = 0

    def report_index_batch(batch: dict[str, Any], value: Any) -> None:
        nonlocal completed_index_batches
        completed_index_batches += 1
        for episode, extracted in value:
            print(
                f"    index batch {completed_index_batches}/{len(index_batches)}: "
                f"{episode['episode_id']} -> {len(extracted)} record(s)"
            )

    index_results, index_usages = _parallel_stage(
        items=index_batches,
        client=gpt_client,
        worker=index_worker,
        max_workers=args.index_workers,
        scope_prefix=f"{query_id}:step3",
        on_success=report_index_batch,
    )
    parallel_usage["step3_to_step6_openai"].extend(index_usages)
    for _batch, extracted_rows in index_results:
        for episode, extracted in extracted_rows:
            records_by_episode_id[str(episode["episode_id"])] = extracted
    records: list[dict[str, Any]] = []
    for episode in episodes:
        records.extend(
            records_by_episode_id.get(str(episode["episode_id"]), [])
        )
    records = common.deduplicate_records(records)
    index_source = {
        "schema_version": "longmemeval-ensi-step3-entity-index-v1",
        "conversation_id": query_id,
        "model": args.gpt_model,
        "extraction_unit": "one_complete_theme_episode",
        "records": records,
    }
    write_json(query_dir / "03_entity_index.json", index_source)
    # Persist content-derived work only after the complete current query has a
    # valid index. A later query can reuse this partition and these records,
    # while its planner/retrieval/answer remain query-specific.
    for session in adapted["sessions"]:
        session_id = str(session["session_id"])
        raw_partition, session_episodes, repaired = partition_by_session_id[session_id]
        records_by_number = {
            str(episode["episode_number"]): records_by_episode_id.get(
                str(episode["episode_id"]), []
            )
            for episode in session_episodes
        }
        payload = _make_session_cache_payload(
            session,
            args,
            raw_partition,
            repaired,
            records_by_number,
        )
        fingerprint = str(payload["session_fingerprint"])
        write_json(_session_cache_path(session_cache_dir, fingerprint), payload)
        session_cache_index[fingerprint] = payload
    print(
        f"    index summary: {cached_episode_count} cached episode(s), "
        f"{len(pending_episodes)} pending in {len(index_batches)} batch(es)"
    )
    stage_durations["step3_entity_index"] = time.perf_counter() - step_started

    if stop_after == "index":
        return {
            "schema_version": "longmemeval-ensi-preprocess-summary-v1",
            "question_id": query_id,
            "question_type": adapted["question_type"],
            "question": adapted["question"],
            "question_date": adapted["question_date"],
            "stage_durations_seconds": {
                key: round(value, 6) for key, value in stage_durations.items()
            },
            "stopped_after": "index",
            "_parallel_llm_usage": parallel_usage,
        }

    # Step 4: use the existing planner and its structural validation/audit.
    step_started = time.perf_counter()
    known_properties = {
        str(record.get("property", "")).strip()
        for record in records
        if str(record.get("property", "")).strip()
    }
    known_by_entity = planner_stage._index_properties_by_entity(index_source)
    plan = planner_stage.plan_question(
        gpt_client,
        adapted["question"],
        known_properties,
        known_by_entity,
    )
    plan_source = {
        "schema_version": "longmemeval-ensi-step4-query-plan-v1",
        "conversation_id": query_id,
        "question_id": query_id,
        "question": adapted["question"],
        "question_date": adapted["question_date"],
        "retrieval_plan": plan,
    }
    write_json(query_dir / "04_query_hop_plan.json", plan_source)
    stage_durations["step4_query_plan"] = time.perf_counter() - step_started

    # Step 5: retain the original structured + lexical + dense retrieval
    # strategy.  The embeddings are private to this query's artifact folder.
    step_started = time.perf_counter()
    retrieval = retrieve_one(
        client=gpt_client,
        question_id=query_id,
        question=adapted["question"],
        question_type=adapted["question_type"],
        question_date=adapted["question_date"],
        plan=plan,
        episodes=episodes,
        records=records,
        model=model,
        retrieval_stage=retrieval_stage,
        artifact_dir=query_dir,
        embedding_model_path=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        top_k_per_hop=args.top_k_per_hop,
        lexical_fallback_top_k=args.lexical_fallback_top_k,
        dense_fallback_top_k=args.dense_fallback_top_k,
        minimum_score=args.minimum_score,
        property_semantic_threshold=args.property_semantic_threshold,
        value_semantic_threshold=args.value_semantic_threshold,
        diagnostic_limit=args.diagnostic_limit,
        rebuild_embeddings=args.rebuild_embeddings,
    )
    write_json(query_dir / "05_retrieval.json", retrieval)
    stage_durations["step5_retrieval"] = time.perf_counter() - step_started

    # Step 6: use the old EnSI evidence-first generation strategy.  The
    # question date is available to the generator, while the gold answer and
    # has_answer labels remain unavailable until Step 7.
    step_started = time.perf_counter()
    answer_episodes = common.rank_episodes_for_answer(
        adapted["question"], retrieval["retrieved_original_episodes"]
    )
    generation = generate_answer(
        client=gpt_client,
        common=common,
        question_id=query_id,
        question_type=adapted["question_type"],
        question_date=adapted["question_date"],
        question=adapted["question"],
        plan=_compact_plan(retrieval),
        candidates=answer_episodes,
        all_episodes=episodes,
        max_reasoning_episodes=args.max_reasoning_episodes,
    )
    prediction = str(generation.get("prediction", "Unknown")).strip() or "Unknown"
    write_json(
        query_dir / "06_generation_trace.json",
        {
            "schema_version": "longmemeval-ensi-step6-evidence-first-v1",
            "question_id": query_id,
            "question_type": adapted["question_type"],
            "reasoning_operator": generation.get("reasoning_operator", "direct"),
            "selected_reasoning_episode_ids": generation.get(
                "selected_reasoning_episode_ids", []
            ),
            "cited_evidence_episode_ids": generation.get(
                "cited_evidence_episode_ids", []
            ),
            "generation": generation,
        },
    )
    stage_durations["step6_answer_generation"] = time.perf_counter() - step_started

    total_duration = time.perf_counter() - started
    result = {
        "schema_version": "longmemeval-ensi-query-result-v1",
        "question_id": query_id,
        "question_type": adapted["question_type"],
        "question": adapted["question"],
        "question_date": adapted["question_date"],
        "hypothesis": prediction,
        "prediction": prediction,
        "reasoning_operator": generation.get("reasoning_operator", "direct"),
        "selected_reasoning_episode_ids": generation.get(
            "selected_reasoning_episode_ids", []
        ),
        "retrieved_episode_ids": retrieval["selected_episode_ids"],
        "answer_context_episode_order": [
            episode["episode_id"] for episode in answer_episodes
        ],
        "stage_durations_seconds": {
            key: round(value, 6) for key, value in stage_durations.items()
        },
        "total_duration_seconds": round(total_duration, 6),
        "gold_isolation": (
            "gold answer is attached only after Steps 1-6 and supplied only to "
            "the Step-7 judge; has_answer labels are never sent to an LLM"
        ),
        "_parallel_llm_usage": parallel_usage,
    }
    disk_result = dict(result)
    disk_result.pop("_parallel_llm_usage", None)
    write_json(query_dir / "06_prediction.json", disk_result)
    return result


def _regenerate_saved_answer(
    *,
    instance: dict[str, Any],
    query_dir: Path,
    common: ModuleType,
    gpt_client: Any,
    max_reasoning_episodes: int,
) -> dict[str, Any]:
    """Regenerate Step 6 from existing Step 1-5 artifacts without rebuilding memory."""
    from longmemeval_generation import generate_answer

    # Accept both the current compact artifact names and the names written by
    # the earlier LongMemEval runner.  The old files contain one plan/retrieval
    # row in a wrapper, so their presence is still sufficient for answer-only.
    plan_path = next(
        (
            path
            for path in (
                query_dir / "04_query_hop_plan.json",
                query_dir / "04_query_hop_plans_v2.json",
            )
            if path.exists()
        ),
        None,
    )
    retrieval_path = next(
        (
            path
            for path in (
                query_dir / "05_retrieval.json",
                query_dir / "05_hop_retrievals.json",
            )
            if path.exists()
        ),
        None,
    )
    if plan_path is None or retrieval_path is None:
        raise FileNotFoundError(
            f"--answer-only requires existing Step 4/5 artifacts in {query_dir}"
        )
    adapted = adapt_instance(instance)
    retrieval_payload = read_json(retrieval_path)
    plan_payload = read_json(plan_path)

    if plan_path.name == "04_query_hop_plans_v2.json":
        plan_rows = plan_payload.get("plans", [])
        if not isinstance(plan_rows, list):
            plan_rows = []
        matching = [
            row
            for row in plan_rows
            if isinstance(row, dict)
            and str(row.get("question_id", row.get("query_id", "")))
            == str(adapted["question_id"])
        ]
        plan_row = matching[0] if matching else (plan_rows[0] if len(plan_rows) == 1 else None)
        if not isinstance(plan_row, dict):
            raise ValueError(
                f"no Step 4 plan for {adapted['question_id']} in {plan_path}"
            )
        plan = plan_row.get("retrieval_plan", plan_row)
    else:
        plan = plan_payload.get("retrieval_plan", plan_payload)

    if retrieval_path.name == "05_hop_retrievals.json":
        retrieval_rows = retrieval_payload.get("retrievals", [])
        if not isinstance(retrieval_rows, list):
            retrieval_rows = []
        matching = [
            row
            for row in retrieval_rows
            if isinstance(row, dict)
            and str(row.get("question_id", row.get("query_id", "")))
            == str(adapted["question_id"])
        ]
        retrieval = matching[0] if matching else (
            retrieval_rows[0] if len(retrieval_rows) == 1 else None
        )
        if not isinstance(retrieval, dict):
            raise ValueError(
                f"no Step 5 retrieval for {adapted['question_id']} in {retrieval_path}"
            )
    else:
        retrieval = retrieval_payload
    candidates = common.rank_episodes_for_answer(
        adapted["question"], retrieval.get("retrieved_original_episodes", [])
    )
    theme_payload = read_json(query_dir / "02_theme_episodes.json")
    all_episodes = theme_payload.get("episodes") or []
    started = time.perf_counter()
    generation = generate_answer(
        client=gpt_client,
        common=common,
        question_id=str(adapted["question_id"]),
        question_type=str(adapted["question_type"]),
        question_date=str(adapted["question_date"]),
        question=str(adapted["question"]),
        plan=plan,
        candidates=candidates,
        all_episodes=all_episodes,
        max_reasoning_episodes=max_reasoning_episodes,
    )
    elapsed = time.perf_counter() - started
    prediction = str(generation.get("prediction", "Unknown")).strip() or "Unknown"

    final_path = query_dir / "06_prediction.json"
    result = read_json(final_path) if final_path.exists() else {
        "schema_version": "longmemeval-ensi-query-result-v1",
        "question_id": adapted["question_id"],
        "question_type": adapted["question_type"],
        "question": adapted["question"],
        "question_date": adapted["question_date"],
    }
    durations = dict(result.get("stage_durations_seconds") or {})
    previous_step6 = float(durations.get("step6_answer_generation", 0.0) or 0.0)
    durations["step6_answer_generation"] = round(elapsed, 6)
    old_total = float(result.get("total_duration_seconds", 0.0) or 0.0)
    result.update(
        {
            "hypothesis": prediction,
            "prediction": prediction,
            "reasoning_operator": generation.get("reasoning_operator", "direct"),
            "selected_reasoning_episode_ids": generation.get(
                "selected_reasoning_episode_ids", []
            ),
            "retrieved_episode_ids": retrieval.get("selected_episode_ids", []),
            "answer_context_episode_order": [
                str(episode["episode_id"]) for episode in candidates
            ],
            "stage_durations_seconds": durations,
            "total_duration_seconds": round(
                max(0.0, old_total - previous_step6 + elapsed), 6
            ),
            "gold_isolation": (
                "gold answer is attached only after Steps 1-6 and supplied only to "
                "the Step-7 judge; has_answer labels are never sent to an LLM"
            ),
        }
    )
    write_json(
        query_dir / "06_generation_trace.json",
        {
            "schema_version": "longmemeval-ensi-step6-evidence-first-v1",
            "question_id": adapted["question_id"],
            "question_type": adapted["question_type"],
            "reasoning_operator": generation.get("reasoning_operator", "direct"),
            "selected_reasoning_episode_ids": generation.get(
                "selected_reasoning_episode_ids", []
            ),
            "cited_evidence_episode_ids": generation.get(
                "cited_evidence_episode_ids", []
            ),
            "generation": generation,
        },
    )
    write_json(query_dir / "06_prediction.json", result)
    result["_parallel_llm_usage"] = {"step3_to_step6_openai": []}
    return result


def _existing_answer_only_instances(
    *,
    output_dir: Path,
    instances: list[dict[str, Any]],
    question_type: str | None,
    excluded_question_ids: set[str],
) -> list[dict[str, Any]]:
    """Return only dataset items whose saved query directory is answer-ready.

    ``--answer-only`` is an artifact-level operation.  Selecting by dataset
    position can point at a different query set when the output directory was
    created with exclusions or a different slice.  For answer-only runs, the
    existing query directories are therefore the source of truth.
    """
    dataset_by_id = {
        str(item.get("question_id", "")): item for item in instances
    }
    queries_root = output_dir / "queries"
    if not queries_root.exists():
        return []

    selected: list[dict[str, Any]] = []
    for query_dir in sorted(path for path in queries_root.iterdir() if path.is_dir()):
        sessions_path = query_dir / "01_sessions.json"
        theme_path = query_dir / "02_theme_episodes.json"
        plan_path = next(
            (
                path
                for path in (
                    query_dir / "04_query_hop_plan.json",
                    query_dir / "04_query_hop_plans_v2.json",
                )
                if path.exists()
            ),
            None,
        )
        retrieval_path = next(
            (
                path
                for path in (
                    query_dir / "05_retrieval.json",
                    query_dir / "05_hop_retrievals.json",
                )
                if path.exists()
            ),
            None,
        )
        if not sessions_path.exists() or not theme_path.exists():
            continue
        if plan_path is None or retrieval_path is None:
            continue
        try:
            metadata = read_json(sessions_path)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        question_id = str(metadata.get("question_id") or query_dir.name)
        if question_id in excluded_question_ids:
            continue
        item = dataset_by_id.get(question_id)
        if item is None:
            continue
        if question_type is not None and str(
            item.get("question_type", metadata.get("question_type", ""))
        ) != question_type:
            continue
        selected.append(item)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EnSI-Memory sequentially on LongMemEval-S-cleaned"
    )
    # Loading the bundled common module is needed before its shared CLI args
    # can be registered.  The selected --original-dir is validated and loaded
    # again below.
    common_path = DEFAULT_ORIGINAL_DIR / "_common.py"
    if not common_path.exists():
        raise FileNotFoundError(f"original EnSI common module not found: {common_path}")
    if str(DEFAULT_ORIGINAL_DIR) not in sys.path:
        sys.path.insert(0, str(DEFAULT_ORIGINAL_DIR))
    common_for_args = _load_module("ensi_argument_common", common_path)
    _add_arguments(parser, common_for_args)
    args = parser.parse_args()
    _validate_args(args)

    dataset_path = Path(args.dataset).expanduser().resolve()
    instances = load_instances(dataset_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    excluded_question_ids = {str(value) for value in args.exclude_question_id}
    selectable_instances = [
        item
        for item in instances
        if str(item.get("question_id", "")) not in excluded_question_ids
    ]

    if args.per_category is not None:
        completed_ids = _completed_question_ids(output_dir)
        selected, selection_manifest = _select_per_category(
            instances=selectable_instances,
            per_category=args.per_category,
            seed=args.selection_seed,
            completed_ids=completed_ids,
            include_abstention=args.include_abstention,
        )
        selected_ids = {row["question_id"] for row in selection_manifest}
        write_json(
            output_dir / "balanced_selection_manifest.json",
            {
                "schema_version": "longmemeval-stratified-selection-v1",
                "dataset_filename": DATASET_FILENAME,
                "dataset_path": str(dataset_path),
                "per_category": args.per_category,
                "selection_seed": args.selection_seed,
                "include_abstention": args.include_abstention,
                "completed_ids_preserved": sorted(completed_ids & selected_ids),
                "selection": selection_manifest,
            },
        )
        progress_start = 1
        progress_total = len(selected)
        item_range = f"balanced {len(selected)} items"
    elif args.question_type is not None:
        if args.answer_only:
            # The output directory may be a slice created with a different
            # exclusion list.  In that mode select the already answer-ready
            # query directories, not the dataset's positional slice.
            typed_instances = _existing_answer_only_instances(
                output_dir=output_dir,
                instances=selectable_instances,
                question_type=args.question_type,
                excluded_question_ids=excluded_question_ids,
            )
        else:
            typed_instances = [
                item
                for item in selectable_instances
                if str(item.get("question_type", "")) == args.question_type
            ]
        if args.start >= len(typed_instances):
            raise ValueError(
                f"--start={args.start} is outside {args.question_type!r} subset "
                f"with {len(typed_instances)} items"
            )
        end = (
            len(typed_instances)
            if args.limit is None
            else min(len(typed_instances), args.start + args.limit)
        )
        selected = typed_instances[args.start:end]
        progress_start = 1
        progress_total = len(selected)
        item_range = (
            f"{args.question_type} {args.start}:{end} of "
            f"{len(typed_instances)} items"
        )
    else:
        if args.start >= len(selectable_instances):
            raise ValueError(
                f"--start={args.start} is outside selectable dataset with "
                f"{len(selectable_instances)} items"
            )
        end = (
            len(selectable_instances)
            if args.limit is None
            else min(len(selectable_instances), args.start + args.limit)
        )
        selected = selectable_instances[args.start:end]
        progress_start = args.start + 1
        progress_total = len(selected)
        item_range = f"{args.start}:{end} of {len(selectable_instances)}"

    stages = load_original_stages(Path(args.original_dir))
    common = stages["common"]
    session_cache_dir, session_cache_index = _build_session_cache_index(
        output_dir, args, common
    )
    _write_manifest(output_dir, args, dataset_path, len(instances))

    # Load the embedding model once for a full run.  Answer-only comparisons
    # reuse existing Step-5 artifacts and do not need GPU memory for it.
    print(f"Dataset: {dataset_path}")
    print(f"Items: {item_range}")
    if excluded_question_ids:
        print(f"Excluded question IDs: {len(excluded_question_ids)}")
    print(f"Output: {output_dir}")
    print(
        f"Session cache: {session_cache_dir} "
        f"({len(session_cache_index)} reusable source sessions)"
    )
    print(f"Original EnSI source (read-only): {Path(args.original_dir).resolve()}")
    # Step 2 keeps the existing local vLLM configuration.  Step 3 through
    # Step 6 use a separate OpenAI client and therefore cannot accidentally
    # fall back to the local qwen3-32B endpoint.
    partition_client = common.make_client(args)
    gpt_client = _make_gpt_client(common, args)
    judge_client = _make_judge_client(common, args)
    print(
        f"Step 2 LLM: {args.provider} / {args.model} @ {args.base_url}\n"
        f"Step 3-6 LLM: openai / {args.gpt_model} @ {args.gpt_base_url}\n"
        f"Step 7 judge: openai / {args.judge_model} @ {args.judge_base_url}\n"
        f"Concurrency: step2 workers={args.partition_workers}, batch={args.partition_batch_size}; "
        f"step3 workers={args.index_workers}, batch={args.index_batch_size}"
    )
    if not args.answer_only:
        partition_client.check()
    gpt_client.check()
    if args.stop_after != "index":
        judge_client.check()
    model = None
    if not args.answer_only and args.stop_after != "index":
        model = stages["retrieval"].load_model(
            args.embedding_model, args.embedding_device, args.embedding_max_length
        )
    dataset_order = [str(item["question_id"]) for item in instances]

    completed = 0
    for absolute_position, item in enumerate(selected, progress_start):
        question_id = str(item["question_id"])
        query_dir = _query_dir(output_dir, question_id)
        final_path = query_dir / "06_prediction.json"
        if args.answer_only:
            print(
                f"[{absolute_position}/{progress_total}] {question_id}: "
                f"regenerating Step 6 only ({item.get('question_type', '')})"
            )
            gpt_scope_started = False
            try:
                gpt_client.begin_usage_scope(question_id)
                gpt_scope_started = True
                result = _regenerate_saved_answer(
                    instance=item,
                    query_dir=query_dir,
                    common=common,
                    gpt_client=gpt_client,
                    max_reasoning_episodes=args.max_reasoning_episodes,
                )
            except Exception:
                if gpt_scope_started:
                    try:
                        gpt_client.end_usage_scope()
                    except RuntimeError:
                        pass
                raise
            gpt_usage = gpt_client.end_usage_scope()
            result.pop("_parallel_llm_usage", None)
            result["llm_usage_scope"] = {
                "step2_local_vllm": _combine_usage(
                    question_id, [{"llm_call_count": 0, "retry_count": 0}]
                ),
                "step3_to_step6_openai": _combine_usage(question_id, [gpt_usage]),
            }
            write_json(final_path, result)
            result = _judge_and_attach(
                judge_client=judge_client,
                sample=item,
                result=result,
                query_dir=query_dir,
                judge_model=args.judge_model,
                existing_judge=None,
            )
            completed += 1
            _write_prediction_indexes(output_dir, dataset_order)
            _write_evaluation_summary(output_dir, dataset_order, args.judge_model)
            print(
                f"    answer: {result['hypothesis'][:160]}\n"
                f"    judge: {result['judge']['correct']}; "
                f"operator: {result.get('reasoning_operator', 'direct')}\n"
                f"    Step 1-5 reused; Step 6 seconds: "
                f"{result.get('stage_durations_seconds', {}).get('step6_answer_generation', 0)}"
            )
            continue
        if final_path.exists() and not args.force:
            result = read_json(final_path)
            judge_path = query_dir / "07_judge.json"
            judge_payload: dict[str, Any] | None = None
            if judge_path.exists():
                try:
                    candidate = read_json(judge_path)
                    if isinstance(candidate, dict):
                        judge_payload = candidate
                except (OSError, json.JSONDecodeError, TypeError, ValueError):
                    judge_payload = None
            evaluation = result.get("evaluation")
            judge_is_complete = (
                isinstance(result.get("judge"), dict)
                and isinstance(evaluation, dict)
                and "gold_answer" in evaluation
                and "all_gold_sessions_retrieved" in evaluation
                and isinstance(judge_payload, dict)
                and isinstance(judge_payload.get("evaluation"), dict)
            )
            if not judge_is_complete:
                # A previous run may have generated the answer and stopped
                # before Step 7, or may have an older judge without gold
                # evidence diagnostics. Finish evaluation without repeating
                # Steps 1-6; reuse an old judge response when available.
                result = _judge_and_attach(
                    judge_client=judge_client,
                    sample=item,
                    result=result,
                    query_dir=query_dir,
                    judge_model=args.judge_model,
                    existing_judge=judge_payload,
                )
                _write_prediction_indexes(output_dir, dataset_order)
                _write_evaluation_summary(
                    output_dir, dataset_order, args.judge_model
                )
                print(
                    f"[{absolute_position}/{progress_total}] {question_id}: "
                    f"cached answer/evaluation attached; "
                    f"judge={result['judge']['correct']}; "
                    f"gold_dialogs_retrieved="
                    f"{result['evaluation']['all_gold_sessions_retrieved']}"
                )
                completed += 1
                continue
            print(
                f"[{absolute_position}/{progress_total}] {question_id}: "
                f"already complete, judge={result.get('judge', {}).get('correct')}, skipping"
            )
            completed += 1
            continue
        print(
            f"[{absolute_position}/{progress_total}] {question_id}: "
            f"{item.get('question_type', '')}"
        )
        partition_scope_started = False
        gpt_scope_started = False
        try:
            partition_client.begin_usage_scope(question_id)
            partition_scope_started = True
            gpt_client.begin_usage_scope(question_id)
            gpt_scope_started = True
            result = process_one(
                instance=item,
                query_dir=query_dir,
                partition_client=partition_client,
                gpt_client=gpt_client,
                model=model,
                stages=stages,
                args=args,
                session_cache_dir=session_cache_dir,
                session_cache_index=session_cache_index,
                stop_after=args.stop_after,
            )
        except Exception:
            # The scopes must be closed before the exception reaches the
            # caller; no final prediction is written, so the item can be
            # retried.
            if gpt_scope_started:
                try:
                    gpt_client.end_usage_scope()
                except RuntimeError:
                    pass
            if partition_scope_started:
                try:
                    partition_client.end_usage_scope()
                except RuntimeError:
                    pass
            raise
        partition_usage = partition_client.end_usage_scope()
        gpt_usage = gpt_client.end_usage_scope()
        parallel_usage = result.pop("_parallel_llm_usage", {})
        if result.get("stopped_after") == "index":
            result["llm_usage_scope"] = {
                "step2_local_vllm": _combine_usage(
                    question_id,
                    [
                        partition_usage,
                        *parallel_usage.get("step2_local_vllm", []),
                    ],
                ),
                "step3_to_step6_openai": _combine_usage(
                    question_id,
                    [
                        gpt_usage,
                        *parallel_usage.get("step3_to_step6_openai", []),
                    ],
                ),
            }
            write_json(query_dir / "03_preprocess_summary.json", result)
            completed += 1
            print(
                f"    stopped after Step 3; durations: "
                f"{result['stage_durations_seconds']}"
            )
            continue
        result["llm_usage_scope"] = {
            "step2_local_vllm": _combine_usage(
                question_id,
                [
                    partition_usage,
                    *parallel_usage.get("step2_local_vllm", []),
                ],
            ),
            "step3_to_step6_openai": _combine_usage(
                question_id,
                [
                    gpt_usage,
                    *parallel_usage.get("step3_to_step6_openai", []),
                ],
            ),
        }
        # process_one writes the complete artifact before returning; rewrite it
        # once with the two separated usage summaries attached.
        write_json(final_path, result)
        result = _judge_and_attach(
            judge_client=judge_client,
            sample=item,
            result=result,
            query_dir=query_dir,
            judge_model=args.judge_model,
        )
        completed += 1
        _write_prediction_indexes(output_dir, dataset_order)
        _write_evaluation_summary(output_dir, dataset_order, args.judge_model)
        print(
            f"    answer: {result['hypothesis'][:160]}\n"
            f"    judge: {result['judge']['correct']}\n"
            f"    retrieved episodes: {len(result['retrieved_episode_ids'])}; "
            f"total seconds: {result['total_duration_seconds']}"
        )

    _write_prediction_indexes(output_dir, dataset_order)
    _write_evaluation_summary(output_dir, dataset_order, args.judge_model)
    print(f"Completed in this invocation: {completed}/{len(selected)}")
    print(f"Official-style predictions: {output_dir / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
