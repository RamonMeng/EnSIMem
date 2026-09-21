#!/usr/bin/env python3
"""Build controlled property-granularity ablation indexes.

The source episode and entity records are kept fixed.  Only the property
labels used by the index and query plans are changed, so retrieval differences
measure property granularity rather than a different corpus or answer model.

The four supported modes are ``fine`` (surface-faithful), ``broad`` (the
EnSIMem setting), ``broad_plus`` (a slightly broader intermediate vocabulary),
and ``very_broad`` (all non-empty predicates become ``fact``).  The last mode
is deliberately an over-compressed stress test rather than a proposed design.
"""
from __future__ import annotations

import argparse
import copy
import re
import shutil
from pathlib import Path
from typing import Any

from _common import normalize, normalize_property, read_json, write_json


# These are deliberately explicit rather than a universal activity ontology.
# They implement the intended intermediate-granularity comparison: clearly
# action-like predicates share ``activity`` while relations and attributes
# retain their semantic identity.
ACTIVITY_PROPERTIES = {
    "research", "researched", "researching", "investigate", "investigated",
    "investigating", "look_into", "explore", "explored", "exploring",
    "picnic", "picnicking", "camp", "camping", "camped",
    "paint", "painting", "painted", "draw", "drawing", "drew",
    "hike", "hiking", "hiked", "swim", "swimming", "swam",
    "exercise", "exercising", "workout", "working_out",
    "cook", "cooking", "bake", "baking", "pottery", "make_art",
    "create_art", "play", "playing", "read", "reading",
    "travel", "traveling", "travelled", "traveled", "fly_to", "flying",
    "drive_to", "go_to", "visit", "visited", "attend", "attended",
    "participate", "participated", "join", "joined", "volunteer",
    "volunteering", "speak", "speaking", "speech", "presentation",
}

BROAD_RELATIONS = {
    "support": "support",
    "help": "support",
    "assist": "support",
    "encourage": "support",
    "back": "support",
    "prefer": "preference",
    "preference": "preference",
    "like": "preference",
    "enjoy": "preference",
    "feel": "emotion",
    "feeling": "emotion",
    "emotion": "emotion",
    "sentiment": "emotion",
    "identity": "identity",
    "status": "identity",
    "attribute": "identity",
    "goal": "goal",
    "career_goal": "goal",
    "decision": "goal",
    "plan": "goal",
    "intent": "goal",
}

# ``broad`` is the paper configuration.  ``broad_plus`` keeps that policy and
# adds a small number of explicit action variants that are still naturally
# answerable as activities.  It does not collapse temporal, location,
# preference, identity, goal, or relationship fields into ``activity``.
BROAD_PLUS_ACTIVITY_PROPERTIES = ACTIVITY_PROPERTIES | {
    "buy", "buying", "bought", "purchase", "purchasing", "purchased",
    "learn", "learning", "learned", "study", "studying", "studied",
    "teach", "teaching", "taught", "celebrate", "celebrating", "celebrated",
    "birthday", "birthday_celebration", "volunteer_work", "present",
    "presenting", "presentation_event",
}


def original_property(item: dict[str, Any]) -> str:
    raw = normalize_property(item.get("raw_property", ""))
    current = normalize_property(item.get("property", ""))
    return raw or current


def fine_property(item: dict[str, Any]) -> str:
    """Keep the narrowest property already extracted by Step 3."""
    return original_property(item)


def broad_property(property_name: str, value: Any = "") -> str:
    prop = normalize_property(property_name)
    value_norm = normalize_property(value)
    if prop in ACTIVITY_PROPERTIES or value_norm in ACTIVITY_PROPERTIES:
        return "activity"
    if prop in BROAD_RELATIONS:
        return BROAD_RELATIONS[prop]
    return prop


def broad_plus_property(property_name: str, value: Any = "") -> str:
    """Apply a slightly broader version of the current intermediate policy."""
    prop = normalize_property(property_name)
    value_norm = normalize_property(value)
    if prop in BROAD_PLUS_ACTIVITY_PROPERTIES or value_norm in BROAD_PLUS_ACTIVITY_PROPERTIES:
        return "activity"
    # Keep the existing broad relation families exactly as they are.  The
    # only additional merge is the conservative action-like extension above.
    if prop in BROAD_RELATIONS:
        return BROAD_RELATIONS[prop]
    return prop


def very_broad_property(property_name: str, value: Any = "") -> str:
    """Collapse every non-empty property to one intentionally coarse label."""
    prop = normalize_property(property_name)
    value_norm = normalize_property(value)
    return "fact" if prop or value_norm else ""


def property_for_mode(item: dict[str, Any], mode: str) -> str:
    old_property = original_property(item)
    value = item.get("value", "")
    if mode == "fine":
        return fine_property(item)
    if mode == "broad":
        return broad_property(old_property, value)
    if mode == "broad_plus":
        return broad_plus_property(old_property, value)
    if mode == "very_broad":
        return very_broad_property(old_property, value)
    raise ValueError(f"unknown property granularity: {mode}")


def transform_record(record: dict[str, Any], mode: str, position: int) -> dict[str, Any]:
    item = copy.deepcopy(record)
    old_property = original_property(item)
    item["granularity_original_property"] = old_property
    item["granularity_mode"] = mode
    item["raw_property"] = item.get("raw_property", "") or old_property
    item["property"] = property_for_mode(item, mode)
    item["index_id"] = f"{item.get('episode_id', '')}::index_{position:06d}"
    return item


def transform_query_object(obj: Any, mode: str, parent: dict[str, Any] | None = None) -> Any:
    if isinstance(obj, list):
        return [transform_query_object(item, mode, parent) for item in obj]
    if not isinstance(obj, dict):
        return obj

    result = {key: transform_query_object(value, mode, obj) for key, value in obj.items()}
    value = result.get("value", "")

    # Query plans use ``property`` and required-property records use
    # ``broad_property``/``property_text``.  Keep all representations aligned.
    for key in ("property", "broad_property", "property_text"):
        if key not in result or not str(result[key]).strip():
            continue
        current = normalize_property(result[key])
        if mode == "fine":
            # A generic query anchor such as activity:value=research should
            # recover the narrow predicate for the fine-grained condition.
            if current in {"activity", "event", "thing", "other"} and value:
                candidate = normalize_property(value)
                result[key] = candidate or result[key]
            else:
                result[key] = current
        elif mode == "broad":
            result[key] = broad_property(current, value)
        elif mode == "broad_plus":
            result[key] = broad_plus_property(current, value)
        elif mode == "very_broad":
            result[key] = very_broad_property(current, value)
        else:
            raise ValueError(f"unknown property granularity: {mode}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build controlled property-granularity ablation files")
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--output-run-dir", required=True)
    parser.add_argument(
        "--granularity",
        choices=("fine", "broad", "broad_plus", "very_broad"),
        required=True,
        help=(
            "fine=surface-faithful; broad=EnSIMem setting; "
            "broad_plus=slightly broader intermediate; "
            "very_broad=collapse every predicate to fact"
        ),
    )
    parser.add_argument("--plan-file", required=True)
    parser.add_argument("--query-start", type=int, default=1)
    parser.add_argument("--query-count", type=int, default=0)
    args = parser.parse_args()

    source_dir = Path(args.source_run_dir)
    output_dir = Path(args.output_run_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    theme_source = read_json(source_dir / "02_theme_episodes.json")
    index_source = read_json(source_dir / "03_entity_index.json")
    plan_source = read_json(source_dir / args.plan_file)
    questions_source = read_json(source_dir / "01_questions.json")
    gold_source = read_json(source_dir / "01_gold_DO_NOT_USE_BEFORE_STEP_7.json")

    questions = questions_source.get("questions", [])
    start = args.query_start - 1
    stop = start + args.query_count if args.query_count else None
    selected_questions = questions[start:stop]
    if not selected_questions:
        raise ValueError("query range selected no questions")
    selected_ids = {str(item["query_id"]) for item in selected_questions}
    selected_gold = [item for item in gold_source.get("gold", []) if str(item.get("query_id")) in selected_ids]
    selected_plans = [item for item in plan_source.get("plans", []) if str(item.get("query_id")) in selected_ids]
    if len(selected_gold) != len(selected_questions):
        raise ValueError("gold file does not contain every selected query")
    if len(selected_plans) != len(selected_questions):
        raise ValueError("plan file does not contain every selected query")

    records = [
        transform_record(record, args.granularity, position)
        for position, record in enumerate(index_source.get("records", []), 1)
    ]
    # Remove duplicates introduced by broad label collapsing while preserving
    # the first record's provenance and evidence pointers.
    deduped, seen = [], set()
    for record in records:
        identity = (
            record.get("episode_id", ""), record.get("entity", ""),
            record.get("entity_type", ""), record.get("property", ""),
            record.get("value", ""), tuple(record.get("evidence_dia_ids", [])),
        )
        if identity not in seen:
            seen.add(identity)
            deduped.append(record)
    for position, record in enumerate(deduped, 1):
        record["index_id"] = f"{record.get('episode_id', '')}::index_{position:06d}"

    transformed_plans = copy.deepcopy(selected_plans)
    transformed_plans = transform_query_object(transformed_plans, args.granularity)

    # Preserve the original episode store and session/question provenance.
    for filename in ("01_sessions.json",):
        if (source_dir / filename).exists():
            shutil.copy2(source_dir / filename, output_dir / filename)
    write_json(output_dir / "02_theme_episodes.json", theme_source)
    write_json(output_dir / "01_questions.json", {
        **questions_source,
        "questions": selected_questions,
        "query_subset": {"start": args.query_start, "count": len(selected_questions)},
    })
    write_json(output_dir / "01_gold_DO_NOT_USE_BEFORE_STEP_7.json", {
        **gold_source,
        "gold": selected_gold,
        "query_subset": {"start": args.query_start, "count": len(selected_questions)},
    })
    write_json(output_dir / args.plan_file, {
        **plan_source,
        "plans": transformed_plans,
        "query_subset": {"start": args.query_start, "count": len(selected_questions)},
        "property_granularity": args.granularity,
    })
    write_json(output_dir / "03_entity_index.json", {
        **index_source,
        "schema_version": "entity-structured-property-granularity-ablation-v1",
        "property_granularity": args.granularity,
        "property_policy": {
            "fine": "preserve the narrowest extracted property",
            "broad": "collapse explicit action predicates to activity and retain relation distinctions",
            "broad_plus": (
                "apply the broad policy and additionally collapse conservative action-like "
                "variants (buy, learn, study, teach, and celebrate) to activity"
            ),
            "very_broad": (
                "stress-test over-compression by mapping every non-empty property to fact; "
                "answer-bearing values and provenance remain unchanged"
            ),
        }[args.granularity],
        "records": deduped,
    })
    print(f"Created {args.granularity} property-granularity run: {output_dir}")
    print(f"Records: {len(deduped)}; selected queries: {len(selected_questions)}")


if __name__ == "__main__":
    main()
