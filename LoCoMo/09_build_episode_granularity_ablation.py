#!/usr/bin/env python3
"""Build deterministic per-turn/per-session episode ablations.

The existing Step-3 entity-property records are reused and remapped; no LLM
is called and no property is re-extracted.
"""
from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path
from typing import Any

from _common import format_turns, read_json, write_json


def make_episode(session: dict[str, Any], turns: list[dict[str, Any]], mode: str, number: int) -> dict[str, Any]:
    conversation_id = str(session["conversation_id"])
    session_id = str(session["session_id"])
    dia_ids = [str(turn["dia_id"]) for turn in turns]
    episode_id = f"{conversation_id}::{session_id}::ablation_{mode}_{number:04d}"
    if mode == "per-session":
        theme = f"Complete session {session_id}"
        description = "All turns in one original conversation session."
    else:
        theme = f"Single turn {dia_ids[0]}"
        description = "One dialogue turn used as an ablation episode."
    include_images = bool(session.get("include_image_captions"))
    text = "\n".join(
        (
            f"Episode timestamp: {session.get('observed_at', '')}",
            f"Theme: {theme} (topic)",
            format_turns(turns, include_images),
        )
    )
    return {
        "episode_id": episode_id,
        "conversation_id": conversation_id,
        "session_id": session_id,
        "episode_number": number,
        "observed_at": session.get("observed_at", ""),
        "theme": theme,
        "theme_type": "topic",
        "theme_description": description,
        "boundary_reason": "deterministic episode-granularity ablation",
        "start_dia_id": dia_ids[0],
        "end_dia_id": dia_ids[-1],
        "dia_ids": dia_ids,
        "speakers": list(dict.fromkeys(str(turn.get("speaker", "Unknown")) for turn in turns)),
        "turns": turns,
        "text": text,
        "metadata": {
            "source_session_id": session_id,
            "source_session_timestamp": session.get("observed_at", ""),
            "partition_is_contiguous": True,
            "include_image_captions": include_images,
            "ablation_granularity": mode,
            "property_records_reused": True,
        },
    }


def build_episodes(sessions: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    episodes = []
    for session in sessions:
        turns = [copy.deepcopy(turn) for turn in session.get("turns", [])]
        groups = [turns] if mode == "per-session" else [[turn] for turn in turns]
        for number, group in enumerate(groups, 1):
            episodes.append(make_episode(session, group, mode, number))
    return episodes


def remap_records(records: list[dict[str, Any]], old_episodes: list[dict[str, Any]], new_episodes: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    old_by_id = {str(item["episode_id"]): item for item in old_episodes}
    session_targets: dict[str, dict[str, Any]] = {}
    turn_targets: dict[tuple[str, str], dict[str, Any]] = {}
    for episode in new_episodes:
        session_id = str(episode["session_id"])
        if mode == "per-session":
            session_targets[session_id] = episode
        else:
            for dia_id in episode["dia_ids"]:
                turn_targets[(session_id, str(dia_id))] = episode

    output, seen = [], set()
    for record in records:
        old_episode = old_by_id.get(str(record.get("episode_id", "")))
        if old_episode is None:
            continue
        session_id = str(old_episode["session_id"])
        evidence = record.get("evidence_dia_ids") or old_episode.get("dia_ids", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        evidence = [str(value) for value in evidence if str(value)]
        if mode == "per-session":
            targets = [(session_targets[session_id], evidence)]
        else:
            targets = [
                (turn_targets[(session_id, dia_id)], [dia_id])
                for dia_id in evidence
                if (session_id, dia_id) in turn_targets
            ]
        for target, target_evidence in targets:
            item = copy.deepcopy(record)
            item["episode_id"] = target["episode_id"]
            item["evidence_dia_ids"] = list(dict.fromkeys(target_evidence))
            item["index_id"] = ""
            provenance = dict(item.get("provenance") or {})
            provenance.update({
                "episode_id": target["episode_id"],
                "session_id": session_id,
                "extraction_unit": "reused_step3_record",
                "ablation_granularity": mode,
            })
            item["provenance"] = provenance
            identity = (
                item["episode_id"], item.get("entity", ""), item.get("entity_type", ""),
                item.get("property", ""), item.get("value", ""), tuple(item["evidence_dia_ids"]),
            )
            if identity not in seen:
                seen.add(identity)
                output.append(item)
    for position, item in enumerate(output, 1):
        item["index_id"] = f"{item['episode_id']}::index_{position:06d}"
    return output


def copy_if_exists(source: Path, destination: Path) -> None:
    if source.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a no-LLM episode-granularity ablation")
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--output-run-dir", required=True)
    parser.add_argument("--granularity", choices=("per-turn", "per-session"), required=True)
    parser.add_argument("--plan-file", required=True)
    args = parser.parse_args()

    source_dir, output_dir = Path(args.source_run_dir), Path(args.output_run_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sessions_source = read_json(source_dir / "01_sessions.json")
    theme_source = read_json(source_dir / "02_theme_episodes.json")
    index_source = read_json(source_dir / "03_entity_index.json")
    new_episodes = build_episodes(sessions_source["sessions"], args.granularity)
    new_records = remap_records(index_source["records"], theme_source["episodes"], new_episodes, args.granularity)

    write_json(output_dir / "02_theme_episodes.json", {
        **theme_source,
        "schema_version": "entity-structured-ablation-episode-v1",
        "episode_policy": f"deterministic {args.granularity} episode units",
        "ablation_granularity": args.granularity,
        "episodes": new_episodes,
    })
    write_json(output_dir / "03_entity_index.json", {
        **index_source,
        "schema_version": "entity-structured-ablation-index-v1",
        "episode_policy": f"deterministic {args.granularity} episode units",
        "ablation_granularity": args.granularity,
        "property_extraction_policy": "reused unchanged from the theme-episode baseline",
        "records": new_records,
        "completed_episode_ids": [item["episode_id"] for item in new_episodes],
    })
    for filename in ("01_sessions.json", "01_questions.json", "01_gold_DO_NOT_USE_BEFORE_STEP_7.json"):
        copy_if_exists(source_dir / filename, output_dir / filename)
    copy_if_exists(source_dir / args.plan_file, output_dir / args.plan_file)
    print(f"Created {args.granularity}: {output_dir}")
    print(f"Episodes: {len(new_episodes)}")
    print(f"Reused/remapped records: {len(new_records)}")


if __name__ == "__main__":
    main()
