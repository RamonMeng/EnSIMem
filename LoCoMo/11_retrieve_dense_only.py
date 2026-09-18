#!/usr/bin/env python3
"""Dense-only retrieval ablation for the entity-structured memory pipeline.

This script keeps the prepared sessions, theme-coherent episodes, query plans,
answer model, and judge protocol fixed.  It deliberately does *not* read the
entity-property index.  Instead, it embeds overlapping chunks of the complete
episode text and ranks episodes using only the original question plus the
natural-language purpose of each planned hop.

The output uses the same retrieval schema consumed by ``06_answer_from_hops``.
It is intentionally a separate script so the main structured-retrieval step
and its checkpoints are never modified by this ablation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

from _common import format_turns, read_json, render_episodes, write_json
from online_efficiency import measure_context, stage_metrics, summarize_stage


def load_model(model_path: str, device: str | None, max_length: int):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Install retrieval dependencies with: pip install 'sentence-transformers>=3.4.1'"
        ) from exc
    kwargs = {"device": device} if device else {}
    model = SentenceTransformer(model_path, **kwargs)
    model.max_seq_length = max_length
    return model


def episode_fingerprint(episodes: list[dict[str, Any]]) -> str:
    payload = [
        {
            "episode_id": str(item.get("episode_id", "")),
            "text": str(item.get("text", "")),
        }
        for item in episodes
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def episode_chunks(
    episodes: list[dict[str, Any]],
    turns_per_chunk: int,
) -> list[dict[str, str]]:
    """Create overlapping retrieval-only chunks while preserving full episodes."""
    chunks: list[dict[str, str]] = []
    for episode in episodes:
        episode_id = str(episode.get("episode_id", ""))
        turns = episode.get("turns") or []
        include_image_fields = bool(
            episode.get("metadata", {}).get("include_image_captions")
            or any(
                turn.get(key)
                for turn in turns
                if isinstance(turn, dict)
                for key in ("blip_caption", "image_caption", "caption", "query", "img_url")
            )
        )
        if not turns:
            chunks.append(
                {
                    "chunk_id": f"{episode_id}::dense_chunk_001",
                    "episode_id": episode_id,
                    "text": str(episode.get("text", "")),
                }
            )
            continue

        prefix = "\n".join(
            (
                f"Episode timestamp: {episode.get('observed_at', '')}",
                f"Theme: {episode.get('theme', '')} ({episode.get('theme_type', '')})",
            )
        )
        step = max(1, turns_per_chunk - 1)
        number = 0
        for start in range(0, len(turns), step):
            selected = turns[start : start + turns_per_chunk]
            if not selected:
                continue
            number += 1
            chunks.append(
                {
                    "chunk_id": f"{episode_id}::dense_chunk_{number:03d}",
                    "episode_id": episode_id,
                    "text": prefix + "\n" + format_turns(selected, include_image_fields),
                }
            )
            if start + turns_per_chunk >= len(turns):
                break
    return chunks


def load_or_build_embeddings(
    model,
    chunks: list[dict[str, str]],
    output_dir: Path,
    model_path: str,
    batch_size: int,
    max_length: int,
    rebuild: bool,
) -> Any:
    import numpy as np

    fingerprint = episode_fingerprint(
        [{"episode_id": chunk["chunk_id"], "text": chunk["text"]} for chunk in chunks]
    )
    array_path = output_dir / "02_dense_episode_embeddings.npz"
    metadata_path = output_dir / "02_dense_episode_embeddings_meta.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    cache_valid = (
        not rebuild
        and array_path.exists()
        and metadata.get("episode_fingerprint") == fingerprint
        and metadata.get("embedding_model") == model_path
        and metadata.get("embedding_max_length") == max_length
        and metadata.get("chunk_ids") == chunk_ids
    )
    if cache_valid:
        archive = np.load(array_path)
        embeddings = archive["embeddings"]
        if embeddings.shape[0] == len(chunks):
            print(f"Loaded cached dense episode embeddings from {array_path}")
            return embeddings

    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    temporary = array_path.with_suffix(array_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, embeddings=embeddings)
    temporary.replace(array_path)
    write_json(
        metadata_path,
        {
            "episode_fingerprint": fingerprint,
            "embedding_model": model_path,
            "embedding_max_length": max_length,
            "chunk_ids": chunk_ids,
            "chunk_episode_ids": [chunk["episode_id"] for chunk in chunks],
            "normalized": True,
            "policy": "overlapping complete-episode chunks; dense-only query ranking; max chunk score per episode",
        },
    )
    print(f"Saved dense episode embeddings to {array_path}")
    return embeddings


def dense_rank(
    model,
    query_text: str,
    chunks: list[dict[str, str]],
    embeddings: Any,
    episodes_by_id: dict[str, dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    import numpy as np

    if not chunks or top_k <= 0:
        return []
    query_vector = model.encode(
        [query_text],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    similarities = embeddings @ query_vector
    best_by_episode: dict[str, float] = {}
    for index, chunk in enumerate(chunks):
        episode_id = str(chunk["episode_id"])
        score = float(similarities[index])
        if score > best_by_episode.get(episode_id, -math.inf):
            best_by_episode[episode_id] = score
    ordered = sorted(
        best_by_episode,
        key=lambda episode_id: (-best_by_episode[episode_id], episode_id),
    )
    return [
        {
            "episode_id": episode_id,
            "score": round(best_by_episode[episode_id], 6),
            "retrieval_method": "dense_episode_chunk",
        }
        for episode_id in ordered[: min(top_k, len(ordered))]
        if episode_id in episodes_by_id
    ]


def copy_or_write_inputs(
    source_dir: Path,
    output_dir: Path,
    plan_file: str,
    selected_ids: set[str],
    questions_source: dict[str, Any],
    gold_source: dict[str, Any],
    plan_source: dict[str, Any],
    theme_source: dict[str, Any],
) -> None:
    """Make the dense run self-contained for Steps 6--8."""
    if (source_dir / "01_sessions.json").exists():
        shutil.copy2(source_dir / "01_sessions.json", output_dir / "01_sessions.json")
    write_json(output_dir / "02_theme_episodes.json", theme_source)
    write_json(
        output_dir / "01_questions.json",
        {
            **questions_source,
            "questions": [
                item for item in questions_source.get("questions", [])
                if str(item.get("query_id")) in selected_ids
            ],
        },
    )
    write_json(
        output_dir / "01_gold_DO_NOT_USE_BEFORE_STEP_7.json",
        {
            **gold_source,
            "gold": [
                item for item in gold_source.get("gold", [])
                if str(item.get("query_id")) in selected_ids
            ],
        },
    )
    write_json(
        output_dir / plan_file,
        {
            **plan_source,
            "plans": [
                item for item in plan_source.get("plans", [])
                if str(item.get("query_id")) in selected_ids
            ],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the pure dense-retrieval ablation over fixed theme episodes"
    )
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--output-run-dir", required=True)
    parser.add_argument("--plan-file", required=True)
    parser.add_argument("--output-file", default="05_dense_only_retrievals.json")
    parser.add_argument("--top-k-per-hop", type=int, default=5)
    parser.add_argument(
        "--dense-all-matching-top-k",
        type=int,
        default=50,
        help="Fixed dense budget for all_matching/aggregation plans; chosen before evaluation",
    )
    parser.add_argument("--turns-per-chunk", type=int, default=4)
    parser.add_argument("--embedding-model", default="../litsearch/qwen3-embedding-8B")
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--embedding-max-length", type=int, default=256)
    parser.add_argument("--rebuild-embeddings", action="store_true")
    parser.add_argument("--query-start", type=int, default=1)
    parser.add_argument("--query-count", type=int, default=0)
    args = parser.parse_args()

    if args.top_k_per_hop <= 0 or args.dense_all_matching_top_k <= 0:
        raise ValueError("top-k budgets must be positive")
    if args.turns_per_chunk <= 0 or args.embedding_batch_size <= 0:
        raise ValueError("turns-per-chunk and embedding-batch-size must be positive")
    if args.query_start <= 0 or args.query_count < 0:
        raise ValueError("query-start must be positive and query-count cannot be negative")

    source_dir = Path(args.source_run_dir)
    output_dir = Path(args.output_run_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    questions_source = read_json(source_dir / "01_questions.json")
    gold_source = read_json(source_dir / "01_gold_DO_NOT_USE_BEFORE_STEP_7.json")
    theme_source = read_json(source_dir / "02_theme_episodes.json")
    plan_source = read_json(source_dir / args.plan_file)
    questions = questions_source.get("questions", [])
    start = args.query_start - 1
    stop = start + args.query_count if args.query_count else None
    selected_questions = questions[start:stop]
    if not selected_questions:
        raise ValueError("query range selected no questions")
    selected_query_ids = {str(item["query_id"]) for item in selected_questions}
    plans = [
        item for item in plan_source.get("plans", [])
        if str(item.get("query_id")) in selected_query_ids
    ]
    if len(plans) != len(selected_questions):
        raise ValueError("plan file does not contain every selected query")
    gold = [
        item for item in gold_source.get("gold", [])
        if str(item.get("query_id")) in selected_query_ids
    ]
    if len(gold) != len(selected_questions):
        raise ValueError("gold file does not contain every selected query")

    episodes = list(theme_source.get("episodes", []))
    if not episodes:
        raise ValueError("theme episode file contains no episodes")
    episodes_by_id = {str(item["episode_id"]): item for item in episodes}
    chunks = episode_chunks(episodes, args.turns_per_chunk)
    model = load_model(args.embedding_model, args.embedding_device, args.embedding_max_length)
    embeddings = load_or_build_embeddings(
        model=model,
        chunks=chunks,
        output_dir=output_dir,
        model_path=args.embedding_model,
        batch_size=args.embedding_batch_size,
        max_length=args.embedding_max_length,
        rebuild=args.rebuild_embeddings,
    )

    retrievals: list[dict[str, Any]] = []
    for position, plan_item in enumerate(plans, 1):
        stage_started = time.perf_counter()
        plan = plan_item["retrieval_plan"]
        retrieval_scope = plan.get("retrieval_scope", "point")
        union_ids: list[str] = []
        hop_results: list[dict[str, Any]] = []
        for hop in plan.get("hops", []):
            hop_purpose = str(hop.get("purpose", "")).strip()
            dense_query = (
                f"Question: {plan_item.get('question', '')}\n"
                f"Retrieval sub-question: {hop_purpose}"
            ).strip()
            budget = (
                args.dense_all_matching_top_k
                if retrieval_scope == "all_matching"
                else args.top_k_per_hop
            )
            ranked = dense_rank(
                model=model,
                query_text=dense_query,
                chunks=chunks,
                embeddings=embeddings,
                episodes_by_id=episodes_by_id,
                top_k=budget,
            )
            hop_selected_ids = [item["episode_id"] for item in ranked]
            for episode_id in hop_selected_ids:
                if episode_id not in union_ids:
                    union_ids.append(episode_id)
            hop_results.append(
                {
                    **hop,
                    "resolved_anchor": hop.get("anchor", {}),
                    "status": "retrieved_dense_only",
                    "candidate_record_count": 0,
                    "candidate_episode_count": len(episodes),
                    "ranked_episodes": ranked,
                    "selected_episode_ids": hop_selected_ids,
                    "retrieval_scope": retrieval_scope,
                    "bridge_value": "",
                    "bridge_supporting_episode_ids": [],
                    "dense_query_text": dense_query,
                    "dense_budget": budget,
                }
            )

        retrieved_episodes = [episodes_by_id[episode_id] for episode_id in union_ids]
        efficiency = stage_metrics(
            stage="step5_hop_retrieval",
            query_id=str(plan_item["query_id"]),
            wall_time_seconds=time.perf_counter() - stage_started,
            context=measure_context(
                render_episodes(retrieved_episodes), len(retrieved_episodes)
            ),
            context_role="retrieved_episode_context",
            search_steps=len(plan.get("hops", [])),
        )
        retrievals.append(
            {
                "query_id": plan_item["query_id"],
                "question": plan_item["question"],
                "answer_target": plan["answer_target"],
                "reasoning_type": plan["reasoning_type"],
                "retrieval_scope": retrieval_scope,
                "required_properties": plan.get("required_properties", []),
                "aligned_query_plan": plan,
                "query_plan_was_aligned": False,
                "top_k_per_hop": args.top_k_per_hop,
                "dense_all_matching_top_k": args.dense_all_matching_top_k,
                "hop_count": len(plan.get("hops", [])),
                "hop_results": hop_results,
                "resolved_bridges": {},
                "selected_episode_ids": union_ids,
                "selected_episode_count_after_deduplication": len(union_ids),
                "retrieved_original_episodes": retrieved_episodes,
                "step4_online_efficiency": plan_item.get("online_efficiency"),
                "online_efficiency": efficiency,
            }
        )
        print(
            f"[{position}/{len(plans)}] {plan_item['query_id']}: "
            f"{len(plan.get('hops', []))} hop(s), {len(union_ids)} dense episode(s)"
        )

    copy_or_write_inputs(
        source_dir=source_dir,
        output_dir=output_dir,
        plan_file=args.plan_file,
        selected_ids=selected_query_ids,
        questions_source=questions_source,
        gold_source=gold_source,
        plan_source=plan_source,
        theme_source=theme_source,
    )
    write_json(
        output_dir / args.output_file,
        {
            "schema_version": "entity-structured-ablation-dense-only-v1",
            "conversation_id": plan_source.get("conversation_id", ""),
            "retrieval_mode": "dense_only",
            "embedding_model": args.embedding_model,
            "episode_chunk_policy": {
                "turns_per_chunk": args.turns_per_chunk,
                "overlap_turns": max(0, args.turns_per_chunk - 1),
                "uses_complete_episode_evidence_after_selection": True,
                "includes_image_fields": True,
            },
            "ranking_policy": {
                "structured_index_used": False,
                "lexical_fallback_used": False,
                "query_text": "original question plus natural-language hop purpose; no entity/property/value fields",
                "point_budget": args.top_k_per_hop,
                "all_matching_budget": args.dense_all_matching_top_k,
                "episode_score": "maximum cosine similarity over overlapping episode chunks",
            },
            "efficiency_summary": summarize_stage(retrievals, "step5_hop_retrieval"),
            "retrievals": retrievals,
        },
    )
    print(f"Saved dense-only retrievals to {output_dir / args.output_file}")


if __name__ == "__main__":
    main()
