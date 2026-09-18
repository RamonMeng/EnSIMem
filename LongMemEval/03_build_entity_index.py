#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from _common import add_vllm_arguments, deduplicate_records, format_turns, make_client, normalize, normalize_episode_record, read_json, write_json
from prompts import ENTITY_TYPES, INDEX_SYSTEM, INDEX_USER


def extract_episode(client, episode):
    include_image_fields = bool(
        episode.get("metadata", {}).get("include_image_captions")
        or any(
            turn.get(key)
            for turn in episode.get("turns", [])
            if isinstance(turn, dict)
            for key in ("blip_caption", "image_caption", "caption", "query", "img_url")
        )
    )
    response = client.chat_json(
        INDEX_SYSTEM,
        INDEX_USER.format(
            entity_types=ENTITY_TYPES,
            episode_id=episode["episode_id"],
            conversation_id=episode["conversation_id"],
            session_id=episode["session_id"],
            observed_at=episode["observed_at"],
            episode_theme=episode["theme"],
            episode_theme_type=episode["theme_type"],
            episode_text=(
                format_turns(episode["turns"], include_image_fields)
                if episode.get("turns")
                else episode.get("text", "")
            ),
        ),
        # Exhaustive extraction can legitimately produce many records for a long episode.
        # GPT-4.1-mini supports substantially larger outputs; leave headroom to avoid truncation.
        max_tokens=8192,
    )
    raw_records = response.get("records") or []
    if not isinstance(raw_records, list):
        raise ValueError("extractor output field records must be a list")
    records = []
    for position, raw in enumerate(raw_records, 1):
        if isinstance(raw, dict):
            record = normalize_episode_record(raw, episode, position)
            if record:
                records.append(record)
    # Speaker-grounded projection: captions and turns often say “my
    # painting” while the model indexes only the object (painting -> creator).
    # Add a lightweight person/paint navigation handle without inventing a new
    # fact; the source turn and DIA provenance are copied verbatim.  Keep the
    # predicate narrow so this projection does not become a generic activity
    # record that competes with unrelated actions.
    projected = []
    speaker_names = {
        str(turn.get("speaker", "")).strip()
        for turn in episode.get("turns", [])
        if str(turn.get("speaker", "")).strip()
    }
    for record in records:
        object_name = normalize(record.get("entity", ""))
        if record.get("entity_type") != "object" or object_name not in {
            "painting", "paintings", "drawing", "drawings", "art", "stained glass"
        }:
            continue
        creator = str(record.get("value", "")).strip() if record.get("property") == "creator" else ""
        source = str(record.get("source", "")).strip()
        person = creator if creator in speaker_names else source if source in speaker_names else ""
        if not person:
            continue
        projected.append(
            {
                "entity": person,
                "entity_type": "person",
                "property": "paint",
                "value": f"made {record['entity']}",
                "condition_property": "",
                "condition_value": "",
                "property_kind": "relation",
                "modality": record.get("modality", "observed"),
                "conditions": [],
                "valid_time": record.get("valid_time", ""),
                "source": source or person,
                "evidence_dia_ids": record.get("evidence_dia_ids", []),
                "projection_kind": "speaker_object_projection",
                "confidence": min(1.0, float(record.get("confidence", 1.0))),
            }
        )
    for position, raw in enumerate(projected, len(raw_records) + 1):
        record = normalize_episode_record(raw, episode, position)
        if record:
            records.append(record)

    # Preserve explicit speaker-to-event relations that an extractor may
    # represent only as an event/person record.  This is deliberately
    # pattern-gated: a named event merely mentioned by a co-speaker is not
    # projected, while “I saw/attended/went to <event>” is an answer-bearing
    # navigation handle for later entity/property retrieval.
    event_projected = []
    turns_by_id = {str(turn.get("dia_id", "")): turn for turn in episode.get("turns", [])}
    speaker_names = {
        str(turn.get("speaker", "")).strip()
        for turn in episode.get("turns", [])
        if str(turn.get("speaker", "")).strip()
    }
    action_patterns = (
        ("attend", re.compile(r"\b(?:i|we)\b[^.?!]{0,180}\b(?:attend(?:ed|ing)?|went to|visited|joined|participat(?:e|ed|ing))\b", re.I)),
        ("see", re.compile(r"\b(?:i|we)\b[^.?!]{0,180}\b(?:saw|see|seen|watched|heard|listened to)\b", re.I)),
    )
    for record in records:
        if record.get("entity_type") not in {"event", "organization", "place", "person", "activity"}:
            continue
        entity_text = normalize(record.get("entity", ""))
        if not entity_text:
            continue
        for dia_id in record.get("evidence_dia_ids", []):
            turn = turns_by_id.get(str(dia_id))
            if not turn:
                continue
            speaker = str(turn.get("speaker", "")).strip()
            text = str(turn.get("text", ""))
            if not speaker or speaker not in speaker_names or entity_text not in normalize(text):
                continue
            for predicate, pattern in action_patterns:
                if not pattern.search(text):
                    continue
                event_projected.append(
                    {
                        "entity": speaker,
                        "entity_type": "person",
                        "property": predicate,
                        "value": str(record.get("entity", "")).strip(),
                        "condition_property": "",
                        "condition_value": "",
                        "property_kind": "relation",
                        "modality": record.get("modality", "observed"),
                        "conditions": [],
                        "valid_time": record.get("valid_time", ""),
                        "source": speaker,
                        "evidence_dia_ids": [str(dia_id)],
                        "projection_kind": "speaker_event_projection",
                        "confidence": min(1.0, float(record.get("confidence", 1.0))),
                    }
                )
                break

    # Resolve the common local answer pattern “These are for running” only
    # when the immediately preceding exchange names shoes/sneakers.  Both
    # DIA IDs are retained for auditability; no visual information is guessed.
    for position, turn in enumerate(episode.get("turns", [])):
        text = str(turn.get("text", ""))
        match = re.search(r"\b(?:these|this|they)\b[^.?!]{0,80}\b(?:are|is)\s+for\s+([a-z][a-z -]{2,40})", text, re.I)
        if not match or position == 0:
            continue
        previous_text = str(episode["turns"][position - 1].get("text", ""))
        if not re.search(r"\b(?:shoe|shoes|sneaker|sneakers)\b", previous_text, re.I):
            continue
        speaker = str(turn.get("speaker", "")).strip()
        if not speaker:
            continue
        event_projected.append(
            {
                "entity": speaker,
                "entity_type": "person",
                "property": "activity",
                "value": match.group(1).strip(" .,!"),
                "condition_property": "with",
                "condition_value": "shoes",
                "property_kind": "relation",
                "modality": "observed",
                "conditions": [],
                "valid_time": "",
                "source": speaker,
                "evidence_dia_ids": [
                    str(episode["turns"][position - 1].get("dia_id", "")),
                    str(turn.get("dia_id", "")),
                ],
                "projection_kind": "local_coreference_projection",
                "confidence": 1.0,
            }
        )
    for position, raw in enumerate(event_projected, len(raw_records) + len(projected) + 1):
        record = normalize_episode_record(raw, episode, position)
        if record:
            records.append(record)
    return deduplicate_records(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build [entity][entity_type][property:value][condition_property:condition_value] indexes"
    )
    parser.add_argument("--run-dir", default="runs/entity_condition_hop_v3")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Start a new Step-3 checkpoint in the destination file (required to re-extract an existing run with the v2.3 prompt)",
    )
    add_vllm_arguments(parser)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    run_dir = Path(args.run_dir)
    source = read_json(run_dir / "02_theme_episodes.json")
    destination = run_dir / "03_entity_index.json"
    schema = "entity-structured-v2.6-prompt-driven-semantic-atomic-index-gpt41-exhaustive-image-aware"
    checkpoint = read_json(destination) if destination.exists() and not args.force_rebuild else {
        "schema_version": schema,
        "conversation_id": source["conversation_id"],
        "model": args.model,
        "episode_policy": "contiguous theme-coherent episode",
        "principle": "[entity][entity_type][property:value][condition_property:condition_value] points to complete theme episode",
        "extraction_unit": "one_complete_theme_episode",
        "property_policy": "minimum-sufficient semantic-head predicate chosen by the extraction prompt; activity only when no narrower predicate is explicit",
        "normalization_policy": "structural key normalization only; no post-hoc semantic predicate canonicalization",
        "speaker_projection_policy": "explicit first-person event/object actions receive speaker-grounded navigation handles; local object pronouns may receive a two-DIA coreference handle",
        "completed_episode_ids": [],
        "records": [],
    }
    if checkpoint.get("schema_version") != schema or checkpoint.get("model") != args.model:
        raise ValueError(
            "Existing index checkpoint uses a different schema or model; "
            "use a new output file or rerun with --force-rebuild"
        )
    completed = set(checkpoint["completed_episode_ids"])
    pending = [episode for episode in source["episodes"] if episode["episode_id"] not in completed]
    client = make_client(args)
    client.check()
    total = len(source["episodes"])
    print(f"Index status: {len(completed)}/{total} episodes; pending={len(pending)}; batch={args.batch_size}")
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset : offset + args.batch_size]
        successes, failures = [], []
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {executor.submit(extract_episode, client, episode): episode for episode in batch}
            for future in as_completed(futures):
                episode = futures[future]
                try:
                    successes.append((episode, future.result()))
                except Exception as exc:
                    failures.append((episode, exc))
        for episode, records in sorted(successes, key=lambda item: item[0]["episode_id"]):
            checkpoint["records"].extend(records)
            checkpoint["completed_episode_ids"].append(episode["episode_id"])
            completed.add(episode["episode_id"])
            print(f"[{len(completed)}/{total}] {episode['episode_id']}: {len(records)} records")
        checkpoint["records"] = deduplicate_records(checkpoint["records"])
        write_json(destination, checkpoint)
        if failures:
            details = "; ".join(f"{episode['episode_id']}: {exc}" for episode, exc in failures)
            raise RuntimeError(f"Extraction failures; successful work was checkpointed: {details}")
    print(f"Saved {len(checkpoint['records'])} records to {destination}")


if __name__ == "__main__":
    main()
