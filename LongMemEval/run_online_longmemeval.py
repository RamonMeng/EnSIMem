#!/usr/bin/env python3
"""Run only Steps 4-6 on existing LongMemEval Step 1-3 artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from longmemeval_adapter import adapt_instance, load_instances, read_json, write_json
from longmemeval_generation import generate_answer
from longmemeval_retrieval import retrieve_one


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_stages(original_dir: Path) -> dict:
    if str(original_dir) not in sys.path:
        sys.path.insert(0, str(original_dir))
    required = {
        "common": "_common.py",
        "planner": "04_plan_query_hops.py",
        "retrieval": "05_retrieve_hops.py",
    }
    missing = [name for name, filename in required.items() if not (original_dir / filename).exists()]
    if missing:
        raise FileNotFoundError(f"missing EnSI modules in {original_dir}: {missing}")
    return {
        key: load_module(f"online_ensi_{key}", original_dir / filename)
        for key, filename in required.items()
    }


def query_dir(root: Path, question_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in question_id)
    return root / "queries" / (safe.strip("._") or "unknown_query")


def compact_plan(retrieval: dict) -> dict:
    return {
        "answer_target": retrieval["answer_target"],
        "reasoning_type": retrieval["reasoning_type"],
        "retrieval_scope": retrieval.get("retrieval_scope", "point"),
        "required_properties": retrieval.get("required_properties", []),
        "hops": [
            {
                "hop_id": hop["hop_id"],
                "purpose": hop.get("purpose", ""),
                "resolved_anchor": hop.get("resolved_anchor", hop.get("anchor", {})),
                "depends_on": hop.get("depends_on", []),
                "bridge_value": hop.get("bridge_value", ""),
                "selected_episode_ids": hop.get("selected_episode_ids", []),
            }
            for hop in retrieval.get("hop_results", [])
        ],
    }


def make_client(common, args):
    return common.make_client(
        SimpleNamespace(
            provider="openai",
            base_url=args.gpt_base_url,
            model=args.gpt_model,
            api_key=args.gpt_api_key,
            timeout=args.gpt_timeout,
        )
    )


def process_one(item, qdir, stages, client, model, args):
    common = stages["common"]
    planner = stages["planner"]
    retrieval_stage = stages["retrieval"]
    adapted = adapt_instance(item)
    qid = str(adapted["question_id"])
    question = str(adapted["question"])

    read_json(qdir / "01_sessions.json")
    episodes_source = read_json(qdir / "02_theme_episodes.json")
    index_source = read_json(qdir / "03_entity_index.json")
    episodes = episodes_source.get("episodes") or []
    records = index_source.get("records") or []
    if not episodes or not isinstance(records, list):
        raise ValueError(f"missing/empty Step 2 or Step 3 artifacts in {qdir}")

    known_properties = {
        str(record.get("property", "")).strip()
        for record in records
        if str(record.get("property", "")).strip()
    }
    known_by_entity = planner._index_properties_by_entity(index_source)

    print(f"[{qid}] Step 4 planning", flush=True)
    started = time.perf_counter()
    client.begin_usage_scope(f"{qid}:step4")
    try:
        plan = planner.plan_question(client, question, known_properties, known_by_entity)
    finally:
        client.end_usage_scope()
    write_json(
        qdir / "04_query_hop_plan.json",
        {
            "schema_version": "longmemeval-ensi-step4-query-plan-v1",
            "conversation_id": qid,
            "question_id": qid,
            "question": question,
            "question_date": adapted["question_date"],
            "retrieval_plan": plan,
        },
    )
    print(f"[{qid}] Step 4 done ({time.perf_counter() - started:.1f}s)", flush=True)

    print(f"[{qid}] Step 5 retrieval", flush=True)
    started = time.perf_counter()
    client.begin_usage_scope(f"{qid}:step5")
    try:
        retrieval = retrieve_one(
            client=client,
            question_id=qid,
            question=question,
            plan=plan,
            episodes=episodes,
            records=records,
            model=model,
            retrieval_stage=retrieval_stage,
            artifact_dir=qdir,
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
    finally:
        client.end_usage_scope()
    write_json(qdir / "05_retrieval.json", retrieval)
    print(
        f"[{qid}] Step 5 done ({time.perf_counter() - started:.1f}s); "
        f"retrieved={len(retrieval.get('selected_episode_ids', []))}",
        flush=True,
    )

    print(f"[{qid}] Step 6 generation", flush=True)
    started = time.perf_counter()
    answer_episodes = common.rank_episodes_for_answer(
        question, retrieval["retrieved_original_episodes"]
    )
    client.begin_usage_scope(f"{qid}:step6")
    try:
        generation = generate_answer(
            client=client,
            common=common,
            question_id=qid,
            question_type=adapted["question_type"],
            question_date=adapted["question_date"],
            question=question,
            plan=compact_plan(retrieval),
            candidates=answer_episodes,
            all_episodes=episodes,
            max_reasoning_episodes=args.max_reasoning_episodes,
        )
    finally:
        client.end_usage_scope()
    prediction = str(generation.get("prediction", "Unknown")).strip() or "Unknown"
    write_json(
        qdir / "06_generation_trace.json",
        {
            "schema_version": "longmemeval-ensi-step6-evidence-first-v1",
            "question_id": qid,
            "question_type": adapted["question_type"],
            "reasoning_operator": generation.get("reasoning_operator", "direct"),
            "selected_reasoning_episode_ids": generation.get("selected_reasoning_episode_ids", []),
            "cited_evidence_episode_ids": generation.get("cited_evidence_episode_ids", []),
            "generation": generation,
        },
    )
    write_json(
        qdir / "06_prediction.json",
        {
            "schema_version": "longmemeval-ensi-online-steps4-6-v1",
            "question_id": qid,
            "question_type": adapted["question_type"],
            "question": question,
            "question_date": adapted["question_date"],
            "hypothesis": prediction,
            "prediction": prediction,
            "reasoning_operator": generation.get("reasoning_operator", "direct"),
            "selected_reasoning_episode_ids": generation.get("selected_reasoning_episode_ids", []),
            "retrieved_episode_ids": retrieval["selected_episode_ids"],
            "answer_context_episode_order": [episode["episode_id"] for episode in answer_episodes],
            "stage_durations_seconds": {
                "step6_answer_generation": round(time.perf_counter() - started, 6)
            },
            "gold_isolation": "gold is not supplied to Steps 4-6",
        },
    )
    print(f"[{qid}] Step 6 done ({time.perf_counter() - started:.1f}s): {prediction[:160]}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run only Steps 4-6 on existing LongMemEval preprocessing")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--original-dir", default=None)
    parser.add_argument(
        "--question-type",
        default="knowledge-update",
        choices=[
            "single-session-user",
            "single-session-assistant",
            "single-session-preference",
            "multi-session",
            "knowledge-update",
            "temporal-reasoning",
        ],
        help="select only complete preprocessed query directories of this category",
    )
    parser.add_argument(
        "--question-id",
        action="append",
        default=[],
        help="rerun only these existing preprocessed query IDs; may be repeated",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gpt-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--gpt-model", default="gpt-4.1-mini-2025-04-14")
    parser.add_argument("--gpt-api-key", default=None)
    parser.add_argument("--gpt-timeout", type=int, default=900)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-max-length", type=int, default=256)
    parser.add_argument("--rebuild-embeddings", action="store_true")
    parser.add_argument("--top-k-per-hop", type=int, default=5)
    parser.add_argument("--lexical-fallback-top-k", type=int, default=12)
    parser.add_argument("--dense-fallback-top-k", type=int, default=8)
    parser.add_argument("--minimum-score", type=float, default=0.35)
    parser.add_argument("--property-semantic-threshold", type=float, default=0.78)
    parser.add_argument("--value-semantic-threshold", type=float, default=0.86)
    parser.add_argument("--diagnostic-limit", type=int, default=25)
    parser.add_argument("--max-reasoning-episodes", type=int, default=20)
    parser.add_argument("--force", action="store_true", help="overwrite existing Steps 4-6")
    args = parser.parse_args()

    dataset = load_instances(args.dataset)
    dataset_by_id = {str(item["question_id"]): item for item in dataset}
    root = Path(args.run_dir).expanduser().resolve()

    # Select from the actual preprocessed query directories, not from the
    # complete benchmark.  This is important when a run contains only a
    # balanced subset or a partially completed preprocessing sweep.
    existing_by_id = {}
    queries_root = root / "queries"
    if queries_root.exists():
        for qdir in sorted(path for path in queries_root.iterdir() if path.is_dir()):
            sessions_path = qdir / "01_sessions.json"
            episodes_path = qdir / "02_theme_episodes.json"
            index_path = qdir / "03_entity_index.json"
            if not (sessions_path.exists() and episodes_path.exists() and index_path.exists()):
                continue
            metadata = read_json(sessions_path)
            if str(metadata.get("question_type", "")) != args.question_type:
                continue
            qid = str(metadata.get("question_id") or qdir.name)
            item = dataset_by_id.get(qid)
            if item is None:
                raise KeyError(f"{qid}: not found in dataset {args.dataset}")
            existing_by_id[qid] = item

    if args.question_id:
        missing = [qid for qid in args.question_id if qid not in existing_by_id]
        if missing:
            raise FileNotFoundError(
                "requested query IDs do not have complete existing Step 1-3 artifacts: "
                + ", ".join(missing)
            )
        selected = [existing_by_id[qid] for qid in args.question_id]
    else:
        selected = [existing_by_id[qid] for qid in sorted(existing_by_id)]

    end = len(selected) if args.limit is None else min(len(selected), args.start + args.limit)
    selected = selected[args.start:end]
    if not selected:
        raise ValueError(
            f"no complete existing {args.question_type} query directories in {root / 'queries'}"
        )
    original_dir = Path(args.original_dir or Path(__file__).resolve().parent).resolve()
    if str(original_dir) not in sys.path:
        sys.path.insert(0, str(original_dir))
    stages = load_stages(original_dir)
    common = stages["common"]
    client = make_client(common, args)
    client.check()
    model = stages["retrieval"].load_model(
        args.embedding_model, args.embedding_device, args.embedding_max_length
    )
    print(
        f"Selected {len(selected)} existing {args.question_type} queries from {root / 'queries'}",
        flush=True,
    )
    for position, item in enumerate(selected, 1):
        qid = str(item["question_id"])
        qdir = query_dir(root, qid)
        if not args.force and (qdir / "06_prediction.json").exists():
            print(f"[{position}/{len(selected)}] {qid}: existing Step 6, skip (use --force to rerun)", flush=True)
            continue
        print(f"\n[{position}/{len(selected)}] {qid}", flush=True)
        process_one(item, qdir, stages, client, model, args)
    print("\nFinished Steps 4-6; Step 7 judge was not run.", flush=True)


if __name__ == "__main__":
    main()
