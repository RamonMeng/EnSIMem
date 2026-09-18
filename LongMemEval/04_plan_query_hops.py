#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from _common import (
    add_vllm_arguments,
    make_client,
    normalize_query_plan,
    read_json,
    validate_query_plan,
    write_json,
)
from prompts import (
    ENTITY_TYPES,
    QUERY_AUDIT_SYSTEM,
    QUERY_AUDIT_USER,
    QUERY_REPAIR_SYSTEM,
    QUERY_REPAIR_USER,
    QUERY_SYSTEM,
    QUERY_USER,
)
from online_efficiency import measure_context, stage_metrics, summarize_stage


_QUESTION_INITIALS = {
    "when", "what", "where", "why", "who", "which", "how", "can", "could", "would",
    "did", "does", "do", "is", "are", "was", "were", "should", "please", "tell",
}


def _fallback_entity(question: str) -> tuple[str, str]:
    """Find an explicit person name without inventing an answer entity."""
    words = re.findall(r"\b[A-Z][A-Za-z'-]+\b", question)
    for word in words:
        if word.lower() not in _QUESTION_INITIALS:
            return word, "person"
    # LongMemEval/LoCoMo questions sometimes refer to the speaker as "my".
    # `user` is the only safe non-answer entity in that case.
    return "user", "person"


def _fallback_property(question: str) -> str:
    """Choose a broad retrieval predicate only when the LLM plan is unusable."""
    q = question.lower()
    if any(token in q for token in ("birthday", "born")):
        return "birthday"
    if any(token in q for token in ("how many", "how often", "total number", "all the")):
        return "activity"
    if any(token in q for token in ("when", "date", "what year", "how long")):
        return "time"
    if any(token in q for token in ("where", "which place", "location")):
        return "location"
    if any(token in q for token in ("destress", "relax", "do to", "activities")):
        return "activity"
    if any(token in q for token in ("recommend", "suggest", "prefer", "like")):
        return "preference"
    if any(token in q for token in ("move", "moved", "relocat")):
        return "move"
    return "activity"


def _make_last_resort_plan(question: str, raw: object) -> dict:
    """Build a minimal valid plan after both model attempts fail.

    This is deliberately used only on an invalid model response.  It keeps the
    batch resumable and gives retrieval a grounded broad anchor instead of
    dropping the query entirely.  All normally valid GPT plans are unchanged.
    """
    entity, entity_type = _fallback_entity(question)
    property_value = _fallback_property(question)
    requirements = raw.get("required_properties", []) if isinstance(raw, dict) else []
    if isinstance(requirements, list):
        for item in requirements:
            if not isinstance(item, dict):
                continue
            candidate = item.get("broad_property") or item.get("property") or item.get("property_text")
            candidate_text = str(candidate or "").strip()
            # Do not let an already-invalid compound label enter the
            # deterministic path; the fallback must itself remain broad.
            if candidate_text and len(re.findall(r"[A-Za-z0-9]+", candidate_text.replace("_", " "))) <= 2:
                property_value = candidate_text
                break
    return {
        "answer_target": {"type": "value", "description": "answer from the grounded memory evidence"},
        "reasoning_type": "direct",
        "retrieval_scope": "point",
        "required_properties": [
            {
                "entity": entity,
                "entity_type": entity_type,
                "broad_property": property_value,
                "property_text": property_value,
                "value": "",
                "role": "answer_property",
            }
        ],
        "hops": [
            {
                "hop_id": "h1",
                "purpose": "retrieve the explicit entity and broad property named or implied by the question",
                "anchor": {
                    "entity": entity,
                    "entity_type": entity_type,
                    "property": property_value,
                    "value": "",
                    "condition_property": "",
                    "condition_value": "",
                },
                "depends_on": [],
                "bridge_request": "",
            }
        ],
    }


def _index_properties(index_source: object) -> set[str]:
    """Return the property vocabulary actually present in Step 3."""
    if not isinstance(index_source, dict):
        return set()
    records = index_source.get("records") or []
    if not isinstance(records, list):
        return set()
    return {
        str(record.get("property", "")).strip()
        for record in records
        if isinstance(record, dict) and str(record.get("property", "")).strip()
    }


def _index_properties_by_entity(index_source: object) -> dict[tuple[str, str], set[str]]:
    """Return Step 3 property labels grouped by entity and entity type."""
    grouped: dict[tuple[str, str], set[str]] = {}
    if not isinstance(index_source, dict):
        return grouped
    records = index_source.get("records") or []
    if not isinstance(records, list):
        return grouped
    for record in records:
        if not isinstance(record, dict):
            continue
        entity = str(record.get("entity", "")).strip().lower()
        entity_type = str(record.get("entity_type", "")).strip().lower()
        prop = str(record.get("property", "")).strip()
        if entity and prop:
            grouped.setdefault((entity, entity_type), set()).add(prop)
    return grouped


def _format_index_properties(properties: set[str]) -> str:
    if not properties:
        return "(No Step 3 vocabulary was found; use short broad predicates.)"
    return ", ".join(sorted(properties))


def plan_question(
    client,
    question: str,
    known_properties: set[str],
    known_properties_by_entity: dict[tuple[str, str], set[str]] | None = None,
) -> dict:
    vocabulary = _format_index_properties(known_properties)

    def normalize_and_align(raw_plan: object) -> dict:
        # Semantic predicate choice is made by GPT-4.1-mini under QUERY_USER
        # and QUERY_REPAIR_USER.  Keep normalization structural only; do not
        # rewrite a model predicate from one dataset-specific vocabulary into
        # another after the fact.
        return normalize_query_plan(raw_plan)

    raw = client.chat_json(
        QUERY_SYSTEM,
        QUERY_USER.format(
            entity_types=ENTITY_TYPES,
            question=question,
            index_property_vocabulary=vocabulary,
        ),
        max_tokens=4096,
    )
    first_error = None
    second_error = None
    for attempt in range(2):
        try:
            plan = normalize_and_align(raw)
            validate_query_plan(question, plan)
            # A syntactically valid plan can still lose an explicit event,
            # temporal constraint, or multi-person requirement.  Ask the same
            # model for a conservative semantic audit; if the audit fails or
            # proposes an invalid plan, retain the already-valid candidate so
            # this guard cannot discard a usable query.
            try:
                audited_raw = client.chat_json(
                    QUERY_AUDIT_SYSTEM,
                    QUERY_AUDIT_USER.format(
                        question=question,
                        index_property_vocabulary=vocabulary,
                        plan=json.dumps(plan, ensure_ascii=False, indent=2),
                    ),
                    max_tokens=4096,
                )
                audited = normalize_and_align(audited_raw)
                validate_query_plan(question, audited)
                return audited
            except Exception:
                return plan
        except ValueError as error:
            if attempt:
                second_error = error
                break
            first_error = error
            raw = client.chat_json(
                QUERY_REPAIR_SYSTEM,
                QUERY_REPAIR_USER.format(
                    question=question,
                    error=str(error),
                    plan=json.dumps(raw, ensure_ascii=False, indent=2),
                    index_property_vocabulary=vocabulary,
                ),
                max_tokens=4096,
            )
    # A malformed response should not discard all later questions.  Keep this
    # deterministic fallback strictly after the initial and repair attempts so
    # it cannot alter any plan that already validates successfully.
    fallback = _make_last_resort_plan(question, raw)
    try:
        plan = normalize_and_align(fallback)
        validate_query_plan(question, plan)
        return plan
    except ValueError as fallback_error:
        raise ValueError(
            f"query plan remained invalid after repair and deterministic fallback; "
            f"first={first_error}; second={second_error}; fallback={fallback_error}"
        ) from fallback_error


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan evidence-seeking direct and multi-hop retrieval")
    parser.add_argument("--run-dir", default="runs/entity_condition_hop_v3")
    parser.add_argument("--questions-file", default="01_questions.json")
    parser.add_argument(
        "--index-file",
        default="03_entity_index.json",
        help="Step 3 index whose observed property labels define query-plan granularity",
    )
    parser.add_argument("--output-file", default="04_query_hop_plans_v2.json")
    parser.add_argument(
        "--force-replan",
        action="store_true",
        help="Discard only the selected output checkpoint and regenerate every question",
    )
    add_vllm_arguments(parser)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    source = read_json(run_dir / args.questions_file)
    index_path = run_dir / args.index_file
    index_source = read_json(index_path) if index_path.exists() else {}
    known_properties = _index_properties(index_source)
    known_properties_by_entity = _index_properties_by_entity(index_source)
    if known_properties:
        print(f"Loaded {len(known_properties)} observed Step 3 property labels from {index_path}")
    else:
        print(f"Warning: no Step 3 property vocabulary found at {index_path}; no label alignment will be applied")
    destination = run_dir / args.output_file
    schema = "entity-structured-v2.7-prompt-driven-audited-query-plan-with-retrieval-scope-gpt41-exhaustive-efficiency-v3"
    checkpoint = read_json(destination) if destination.exists() and not args.force_replan else {
        "schema_version": schema,
        "conversation_id": source["conversation_id"],
        "model": args.model,
        "planning_policy": "prompt-driven-minimum-sufficient-predicates-with-exhaustive-aggregation-and-semantic-audit-v4",
        "plans": [],
    }
    # Do not silently trust an older-schema checkpoint.  A legacy file can
    # contain plans from a different experiment (including query IDs after a
    # previous failure), so its row count is not evidence that this run
    # completed those questions.  The caller must explicitly migrate a known
    # completed prefix into a current-schema file before resuming.
    if not isinstance(checkpoint.get("plans"), list):
        raise ValueError("Existing plan checkpoint has no usable plans list")
    recorded_model = checkpoint.get("model")
    if recorded_model and recorded_model != args.model:
        raise ValueError(
            f"Existing plan checkpoint was generated with model {recorded_model!r}; "
            f"requested {args.model!r}. Use the same --model or a new output file."
        )
    if checkpoint.get("schema_version") != schema:
        raise ValueError(
            "Existing query-plan checkpoint uses legacy schema "
            f"{checkpoint.get('schema_version')!r}. It may contain stale plans; "
            "create a current-schema checkpoint containing only the queries that "
            "definitely completed, then rerun without --force-replan."
        )
    checkpoint["schema_version"] = schema
    checkpoint["model"] = args.model
    checkpoint.setdefault("conversation_id", source.get("conversation_id", ""))
    checkpoint.setdefault(
        "planning_policy",
        "prompt-driven-minimum-sufficient-predicates-with-exhaustive-aggregation-and-semantic-audit-v4",
    )
    completed = {item["query_id"] for item in checkpoint["plans"]}
    if args.force_replan:
        write_json(destination, checkpoint)
    pending = [
        (position, question)
        for position, question in enumerate(source["questions"], 1)
        if question["query_id"] not in completed
    ]
    print(
        f"Loaded query-plan checkpoint: {len(completed)}/{len(source['questions'])} complete; "
        f"pending={len(pending)}; file={destination}"
    )
    if not pending:
        print("All query plans are already complete; no LLM calls were made.")
        return
    client = make_client(args)
    client.check()
    for position, question in pending:
        # Efficiency accounting is observational: the exact same planner and
        # prompts run inside this scope.  The measured context is the user
        # planning prompt (query + observed Step-3 vocabulary), not hidden
        # system text and not corpus preprocessing.
        planning_context = QUERY_USER.format(
            entity_types=ENTITY_TYPES,
            question=question["question"],
            index_property_vocabulary=_format_index_properties(known_properties),
        )
        stage_started = time.perf_counter()
        client.begin_usage_scope(question["query_id"])
        try:
            plan = plan_question(
                client,
                question["question"],
                known_properties,
                known_properties_by_entity,
            )
        finally:
            client.end_usage_scope()
        efficiency = stage_metrics(
            stage="step4_query_planning",
            query_id=question["query_id"],
            wall_time_seconds=time.perf_counter() - stage_started,
            context=measure_context(planning_context, 0),
            context_role="query_planning_prompt",
        )
        checkpoint["plans"].append(
            {**question, "retrieval_plan": plan, "online_efficiency": efficiency}
        )
        checkpoint["efficiency_summary"] = summarize_stage(
            checkpoint["plans"], "step4_query_planning"
        )
        write_json(destination, checkpoint)
        print(f"[{position}/{len(source['questions'])}] {question['query_id']}: {len(plan['hops'])} hop(s)")
    checkpoint["efficiency_summary"] = summarize_stage(
        checkpoint["plans"], "step4_query_planning"
    )
    write_json(destination, checkpoint)
    print(f"Saved plans to {destination}")


if __name__ == "__main__":
    main()
