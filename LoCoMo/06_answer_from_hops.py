#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from _common import (
    add_vllm_arguments,
    expanded_evidence,
    make_client,
    normalize_temporal_prediction,
    rank_episodes_for_answer,
    read_json,
    render_episodes,
    write_json,
)
from prompts import ANSWER_SYSTEM, ANSWER_USER
from online_efficiency import measure_context, stage_metrics, summarize_stage


def compact_plan(retrieval):
    return {
        "answer_target": retrieval["answer_target"],
        "reasoning_type": retrieval["reasoning_type"],
        "retrieval_scope": retrieval.get("retrieval_scope", "point"),
        "required_properties": retrieval.get("required_properties", []),
        "hops": [
            {
                "hop_id": hop["hop_id"],
                "purpose": hop["purpose"],
                "resolved_anchor": hop["resolved_anchor"],
                "depends_on": hop["depends_on"],
                "bridge_value": hop.get("bridge_value", ""),
                "selected_episode_ids": hop["selected_episode_ids"],
            }
            for hop in retrieval["hop_results"]
        ],
    }


def gold_audit(gold, retrieval):
    gold_dia_ids = expanded_evidence(gold.get("evidence") or [])
    retrieved = retrieval["retrieved_original_episodes"]
    retrieved_dia_ids = list(
        dict.fromkeys(dia_id for episode in retrieved for dia_id in episode["dia_ids"])
    )
    relevant_episodes = []
    relevant_turns = []
    for episode in retrieved:
        matching = [dia_id for dia_id in episode["dia_ids"] if dia_id in gold_dia_ids]
        if not matching:
            continue
        relevant_episodes.append(episode["episode_id"])
        turns_by_id = {str(turn.get("dia_id", "")): turn for turn in episode["turns"]}
        relevant_turns.extend(turns_by_id[dia_id] for dia_id in matching if dia_id in turns_by_id)
    retrieved_gold = [dia_id for dia_id in gold_dia_ids if dia_id in retrieved_dia_ids]
    return {
        "gold_answer": gold.get("answer", ""),
        "gold_category": gold.get("category"),
        "gold_evidence_dia_ids": gold_dia_ids,
        "retrieved_gold_evidence_dia_ids": retrieved_gold,
        "missing_gold_evidence_dia_ids": [dia_id for dia_id in gold_dia_ids if dia_id not in retrieved_dia_ids],
        "gold_evidence_coverage": len(retrieved_gold) / len(gold_dia_ids) if gold_dia_ids else None,
        "relevant_retrieved_episode_ids": relevant_episodes,
        "relevant_retrieved_turns": relevant_turns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer from the union of complete episodes retrieved by all hops")
    parser.add_argument("--run-dir", default="runs/entity_condition_hop_v3")
    parser.add_argument("--retrieval-file", default="05_hop_retrievals.json")
    parser.add_argument("--output-file", default="06_hop_predictions.json")
    parser.add_argument("--gold-file", default="01_gold_DO_NOT_USE_BEFORE_STEP_7.json")
    add_vllm_arguments(parser)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    source = read_json(run_dir / args.retrieval_file)
    gold_path = run_dir / args.gold_file
    gold_by_id = {}
    if gold_path.exists():
        gold_by_id = {item["query_id"]: item for item in read_json(gold_path)["gold"]}
    destination = run_dir / args.output_file
    schema = "entity-structured-v2.7-prompt-driven-chunked-dense-recall-complete-theme-episode-answer-gpt41-efficiency-v3"
    checkpoint = read_json(destination) if destination.exists() else {
        "schema_version": schema,
        "conversation_id": source["conversation_id"],
        "model": args.model,
        "gold_isolation_policy": "gold appended only after model generation and never included in prompt",
        "predictions": [],
    }
    # Do not silently trust an older-schema answer checkpoint.  A legacy file
    # can belong to a different retrieval/prompt experiment, and its row count
    # is not evidence that this run completed those retrievals.  Migrate only a
    # known completed subset into a current-schema file before resuming.
    if not isinstance(checkpoint.get("predictions"), list):
        raise ValueError("Existing prediction checkpoint has no usable predictions list")
    recorded_model = checkpoint.get("model")
    if recorded_model and recorded_model != args.model:
        raise ValueError(
            f"Existing prediction checkpoint was generated with model {recorded_model!r}; "
            f"requested {args.model!r}. Use the same --model or a new output file."
        )
    if checkpoint.get("schema_version") != schema:
        raise ValueError(
            "Existing Step 6 checkpoint uses legacy schema "
            f"{checkpoint.get('schema_version')!r}. It may contain stale answers; "
            "use a new output file or migrate only definitely completed rows."
        )
    checkpoint["schema_version"] = schema
    checkpoint["model"] = args.model
    checkpoint.setdefault("conversation_id", source.get("conversation_id", ""))
    checkpoint.setdefault(
        "gold_isolation_policy",
        "gold appended only after model generation and never included in prompt",
    )
    completed = {item["query_id"] for item in checkpoint["predictions"]}
    print(
        f"Loaded answer checkpoint: {len(completed)}/{len(source['retrievals'])} complete; "
        f"pending={sum(item['query_id'] not in completed for item in source['retrievals'])}; "
        f"file={destination}"
    )
    if len(completed) >= len(source["retrievals"]):
        print("All predictions are already complete; no LLM calls were made.")
        return
    client = make_client(args)
    client.check()
    for position, retrieval in enumerate(source["retrievals"], 1):
        if retrieval["query_id"] in completed:
            print(f"[{position}/{len(source['retrievals'])}] skip {retrieval['query_id']}")
            continue
        plan = compact_plan(retrieval)
        # Retrieval preserves every structured and lexical candidate for
        # auditability.  Put the episodes whose full text best matches this
        # question first in the prompt, without dropping any evidence.  This
        # prevents a semantically related but answer-irrelevant episode from
        # anchoring the LLM when a recall guard added extra context.
        stage_started = time.perf_counter()
        client.begin_usage_scope(retrieval["query_id"])
        answer_episodes = rank_episodes_for_answer(
            retrieval["question"], retrieval["retrieved_original_episodes"]
        )
        selected_context = render_episodes(answer_episodes)
        prediction = client.chat(
                ANSWER_SYSTEM,
                ANSWER_USER.format(
                    question=retrieval["question"],
                    plan=json.dumps(plan, ensure_ascii=False, indent=2),
                    episodes=selected_context,
                ),
                # Lists and count explanations need a little headroom after the
                # exhaustive inventory instruction, while remaining concise.
                max_tokens=256,
                temperature=0.0,
            ).strip()
        client.end_usage_scope()
        efficiency = stage_metrics(
            stage="step6_answer_generation",
            query_id=retrieval["query_id"],
            wall_time_seconds=time.perf_counter() - stage_started,
            context=measure_context(selected_context, len(answer_episodes)),
            context_role="selected_episode_context",
        )
        # Memora reports search latency and end-to-end latency.  Preserve the
        # per-stage measurements and add this read-only per-query comparison
        # when Step 4/5 metrics are available in the input retrieval file.
        step4_efficiency = retrieval.get("step4_online_efficiency") or {}
        step5_efficiency = retrieval.get("online_efficiency") or {}
        component_latencies = [
            float(item["latency_seconds"])
            for item in (step4_efficiency, step5_efficiency, efficiency)
            if item.get("latency_seconds") is not None
        ]
        efficiency["memora_comparison"] = {
            "search_latency_seconds": step5_efficiency.get("latency_seconds"),
            "end_to_end_latency_seconds": (
                round(sum(component_latencies), 6)
                if len(component_latencies) == 3
                else None
            ),
            "search_steps": step5_efficiency.get("search_steps"),
            "context_tokens": efficiency["context"].get("tokens"),
        }
        raw_prediction = prediction
        prediction, temporal_action = normalize_temporal_prediction(
            retrieval["question"],
            prediction,
            retrieval.get("answer_target"),
            retrieval["retrieved_original_episodes"],
        )
        row = {
            "query_id": retrieval["query_id"],
            "question": retrieval["question"],
            "answer_target": retrieval["answer_target"],
            "reasoning_type": retrieval["reasoning_type"],
            "hop_count": retrieval["hop_count"],
            "hop_episode_ids": {
                hop["hop_id"]: hop["selected_episode_ids"] for hop in retrieval["hop_results"]
            },
            "retrieved_episode_ids": retrieval["selected_episode_ids"],
            "answer_context_episode_order": [
                episode["episode_id"] for episode in answer_episodes
            ],
            "prediction": prediction,
            "online_efficiency": efficiency,
        }
        if temporal_action != "none":
            row["raw_prediction_before_temporal_normalization"] = raw_prediction
            row["temporal_normalization"] = temporal_action
        if retrieval["query_id"] in gold_by_id:
            row.update(gold_audit(gold_by_id[retrieval["query_id"]], retrieval))
        checkpoint["predictions"].append(row)
        checkpoint["efficiency_summary"] = summarize_stage(
            checkpoint["predictions"], "step6_answer_generation"
        )
        write_json(destination, checkpoint)
        print(f"[{position}/{len(source['retrievals'])}] {retrieval['query_id']}: {prediction[:120]}")
    checkpoint["efficiency_summary"] = summarize_stage(
        checkpoint["predictions"], "step6_answer_generation"
    )
    write_json(destination, checkpoint)
    print(f"Saved predictions to {destination}")


if __name__ == "__main__":
    main()
