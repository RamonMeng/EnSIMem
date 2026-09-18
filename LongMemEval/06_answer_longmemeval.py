#!/usr/bin/env python3
"""LongMemEval-specific answer generation on top of the unchanged LoCoMo EnSI retrieval output.

Retrieval is closed at Step 5.  This stage may reason, audit, aggregate, order,
and verify the retrieved episodes, but it never launches another retrieval.  A
small direct-fact guard may also inspect the already-preprocessed Step-2 episode
file when Step 5 split one source session across multiple episodes; it does not
perform a new search.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import _common as common
from longmemeval_adapter import read_json, write_json
from longmemeval_generation import generate_answer
from online_efficiency import measure_context, stage_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="LongMemEval reasoning over one-shot EnSI retrieval")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--model", default="gpt-4.1-mini-2025-04-14")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--max-reasoning-episodes", type=int, default=20)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    meta = read_json(run_dir / "01_longmemeval_metadata.json")
    plan_source = read_json(run_dir / "04_query_hop_plans_v2.json")
    retrieval_source = read_json(run_dir / "05_hop_retrievals.json")
    if len(plan_source.get("plans", [])) != 1 or len(retrieval_source.get("retrievals", [])) != 1:
        raise ValueError("LongMemEval per-query run directory must contain exactly one plan and one retrieval")
    plan_row = plan_source["plans"][0]
    retrieval = retrieval_source["retrievals"][0]
    if str(retrieval["query_id"]) != str(meta["question_id"]):
        raise ValueError("Step 5 query_id does not match LongMemEval metadata")

    # The normal answer path remains closed over Step-5 results.  The current generator accepts
    # an optional preprocessed episode pool only for its narrow direct-fact fallback; load it when
    # this legacy one-query runner has the matching Step-2 artifact.
    theme_path = run_dir / "02_theme_episodes.json"
    all_episodes = []
    if theme_path.exists():
        theme_payload = read_json(theme_path)
        all_episodes = theme_payload.get("episodes") or []

    client = common.make_client(
        SimpleNamespace(
            provider="openai",
            base_url=args.base_url,
            model=args.model,
            api_key=args.api_key,
            timeout=args.timeout,
        )
    )
    client.check()

    started = time.perf_counter()
    client.begin_usage_scope(str(meta["question_id"]))
    try:
        result = generate_answer(
            client=client,
            common=common,
            question_id=str(meta["question_id"]),
            question_type=str(meta.get("question_type", "")),
            question_date=str(meta.get("question_date", "")),
            question=str(meta["question"]),
            plan=plan_row["retrieval_plan"],
            candidates=retrieval.get("retrieved_original_episodes", []),
            all_episodes=all_episodes,
            retrieval=retrieval,
            max_reasoning_episodes=args.max_reasoning_episodes,
        )
    finally:
        usage = client.end_usage_scope()

    selected_ids = result.get("selected_reasoning_episode_ids", [])
    by_id = {
        str(ep.get("episode_id")): ep
        for ep in retrieval.get("retrieved_original_episodes", [])
    }
    by_id.update({str(ep.get("episode_id")): ep for ep in all_episodes})
    selected = [by_id[x] for x in selected_ids if x in by_id]
    context_text = common.render_episodes(selected)
    efficiency = stage_metrics(
        stage="step6_longmemeval_answer_generation",
        query_id=str(meta["question_id"]),
        wall_time_seconds=time.perf_counter() - started,
        context=measure_context(context_text, len(selected)),
        context_role="reasoning_episode_context",
    )

    output = {
        "schema_version": "longmemeval-locomo-ensi-one-shot-plus-lme-reasoning-v1",
        "question_id": str(meta["question_id"]),
        "question_type": str(meta.get("question_type", "")),
        "question_date": str(meta.get("question_date", "")),
        "question": str(meta["question"]),
        "hypothesis": result["prediction"],
        "prediction": result["prediction"],
        "retrieval_method": "unchanged_locomo_entity_structured_one_shot",
        "answer_method": "longmemeval_evidence_first_reasoning",
        "retrieval_scope": retrieval.get("retrieval_scope", "point"),
        "reasoning_type": retrieval.get("reasoning_type", ""),
        "reasoning_operator": result.get("reasoning_operator", "direct"),
        "hop_count": retrieval.get("hop_count", 0),
        "retrieved_episode_ids": retrieval.get("selected_episode_ids", []),
        "retrieved_episode_count": len(retrieval.get("selected_episode_ids", [])),
        "selected_reasoning_episode_ids": selected_ids,
        "cited_evidence_episode_ids": result.get("cited_evidence_episode_ids", []),
        "reasoning": result.get("reasoning", ""),
        "generation_trace": result,
        "step4_online_efficiency": retrieval.get("step4_online_efficiency"),
        "step5_online_efficiency": retrieval.get("online_efficiency"),
        "step6_online_efficiency": efficiency,
        "step6_llm_usage_scope": usage,
        "closed_evidence_policy": (
            "Step 6 normally reasons over Step-5 retrieved episodes; the narrow direct-fact "
            "fallback may inspect the existing Step-2 episodes for an incomplete same-session "
            "inventory/count fact and never performs additional retrieval"
        ),
    }
    write_json(run_dir / "06_prediction.json", output)
    print(f"Answer: {output['hypothesis']}")
    print(
        f"Retrieved={output['retrieved_episode_count']} episodes; "
        f"reasoning={len(selected_ids)} episodes; operator={output['reasoning_operator']}"
    )


if __name__ == "__main__":
    main()
