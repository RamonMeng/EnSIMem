#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from _common import (
    add_vllm_arguments,
    construct_theme_episodes,
    format_turns,
    make_client,
    read_json,
    write_json,
)
from prompts import (
    PARTITION_REPAIR_SYSTEM,
    PARTITION_REPAIR_USER,
    PARTITION_SYSTEM,
    PARTITION_USER,
    THEME_POLICY,
)

SCHEMA = "entity-structured-v2.4-prompt-driven-theme-partitioned-episodes"
VERSION_MARKER = "ENSI_STEP2_RESILIENT_V2"


def fallback_partition(session, error):
    dia_ids = [str(x) for x in session["dia_ids"]]
    if not dia_ids:
        raise ValueError(f"{session.get('session_id', '')} has no dialogue IDs")

    return {
        "segments": [
            {
                "start_dia_id": dia_ids[0],
                "end_dia_id": dia_ids[-1],
                "theme": f"Complete conversation session {session['session_id']}",
                "theme_type": "topic",
                "theme_description": "Fallback episode preserving every dialogue turn.",
                "boundary_reason": "session_start",
            }
        ],
        "_fallback_used": True,
        "_fallback_error": str(error),
    }


def call_partition(client, session):
    return client.chat_json(
        PARTITION_SYSTEM,
        PARTITION_USER.format(
            theme_policy=THEME_POLICY,
            conversation_id=session["conversation_id"],
            session_id=session["session_id"],
            observed_at=session["observed_at"],
            session_text=format_turns(
                session["turns"],
                bool(session.get("include_image_captions")),
            ),
        ),
        max_tokens=4096,
    )


def call_repair(client, session, proposal, error):
    return client.chat_json(
        PARTITION_REPAIR_SYSTEM,
        PARTITION_REPAIR_USER.format(
            error=str(error),
            dia_ids=json.dumps(session["dia_ids"], ensure_ascii=False),
            proposal=json.dumps(proposal, ensure_ascii=False, indent=2),
        ),
        max_tokens=4096,
    )


def partition_one(client, session):
    try:
        response = call_partition(client, session)
    except Exception as error:
        fallback = fallback_partition(session, error)
        print(
            f"{session['session_id']}: fallback after initial model error: {error}",
            flush=True,
        )
        return (
            fallback,
            construct_theme_episodes(session, fallback["segments"]),
            True,
        )

    try:
        if not isinstance(response, dict):
            raise ValueError("model response is not a JSON object")

        episodes = construct_theme_episodes(
            session,
            response.get("segments"),
        )
        return response, episodes, False

    except Exception as first_error:
        try:
            repaired = call_repair(
                client,
                session,
                response,
                first_error,
            )

            if not isinstance(repaired, dict):
                raise ValueError("repair response is not a JSON object")

            episodes = construct_theme_episodes(
                session,
                repaired.get("segments"),
            )
            return repaired, episodes, True

        except Exception as repair_error:
            fallback = fallback_partition(session, repair_error)
            print(
                f"{session['session_id']}: fallback after invalid partition/repair: "
                f"{repair_error}",
                flush=True,
            )
            return (
                fallback,
                construct_theme_episodes(session, fallback["segments"]),
                True,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Resilient Step 2 theme partitioning"
    )
    parser.add_argument(
        "--run-dir",
        default="runs/entity_condition_hop_v3",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )
    add_vllm_arguments(parser)
    args = parser.parse_args()

    print(
        f"{VERSION_MARKER} loaded from {Path(__file__).resolve()}",
        flush=True,
    )

    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")

    run_dir = Path(args.run_dir)
    source = read_json(run_dir / "01_sessions.json")
    destination = run_dir / "02_theme_episodes.json"

    if destination.exists():
        checkpoint = read_json(destination)

        old_model = checkpoint.get("model")
        if old_model not in (None, args.model):
            raise ValueError(
                f"Existing checkpoint model={old_model!r}, "
                f"but current model={args.model!r}"
            )

        checkpoint.setdefault("partitions", [])
        checkpoint.setdefault("episodes", [])
        checkpoint["schema_version"] = SCHEMA
        checkpoint["model"] = args.model
        checkpoint.setdefault(
            "conversation_id",
            source.get("conversation_id", ""),
        )
    else:
        checkpoint = {
            "schema_version": SCHEMA,
            "conversation_id": source.get("conversation_id", ""),
            "model": args.model,
            "extraction_context_radius": source.get(
                "extraction_context_radius",
                2,
            ),
            "episode_definition": (
                "one contiguous self-contained set of turns mainly focused "
                "on one theme"
            ),
            "partition_invariants": [
                "ordered",
                "contiguous",
                "non_overlapping",
                "complete_session_coverage",
            ],
            "partitions": [],
            "episodes": [],
        }

    sessions = source["sessions"]
    completed = {
        item["session_id"]
        for item in checkpoint["partitions"]
        if item.get("session_id")
    }

    pending = [
        session
        for session in sessions
        if session["session_id"] not in completed
    ]

    order = {
        session["session_id"]: index
        for index, session in enumerate(sessions)
    }

    print(
        f"Partition status: {len(completed)}/{len(sessions)} complete; "
        f"pending={len(pending)}; batch={args.batch_size}",
        flush=True,
    )

    if not pending:
        print(
            f"All sessions already complete; no model calls were made. "
            f"File={destination}",
            flush=True,
        )
        return

    client = make_client(args)
    client.check()

    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset : offset + args.batch_size]
        successes = []
        failures = []

        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {
                executor.submit(partition_one, client, session): session
                for session in batch
            }

            for future in as_completed(futures):
                session = futures[future]
                try:
                    raw, episodes, repaired = future.result()
                    successes.append(
                        (session, raw, episodes, repaired)
                    )
                except Exception as error:
                    failures.append((session, error))

        for session, raw, episodes, repaired in sorted(
            successes,
            key=lambda item: order[item[0]["session_id"]],
        ):
            checkpoint["partitions"].append(
                {
                    "session_id": session["session_id"],
                    "observed_at": session["observed_at"],
                    "repaired": repaired,
                    "fallback_used": bool(
                        raw.get("_fallback_used", False)
                    ),
                    "fallback_error": raw.get(
                        "_fallback_error",
                        "",
                    ),
                    "segments": raw["segments"],
                }
            )

            checkpoint["episodes"].extend(episodes)

            print(
                f"{session['session_id']}: "
                f"{len(session['turns'])} turns -> "
                f"{len(episodes)} theme episodes",
                flush=True,
            )

        checkpoint["partitions"].sort(
            key=lambda item: order[item["session_id"]]
        )
        checkpoint["episodes"].sort(
            key=lambda item: (
                order.get(item.get("session_id"), 10**9),
                item.get("episode_number", 0),
            )
        )

        write_json(destination, checkpoint)

        if failures:
            details = "; ".join(
                f"{session['session_id']}: {error}"
                for session, error in failures
            )
            raise RuntimeError(
                "Unexpected partition failures; successful sessions "
                f"were checkpointed: {details}"
            )

    print(
        f"Saved {len(checkpoint['episodes'])} theme episodes to "
        f"{destination}",
        flush=True,
    )


if __name__ == "__main__":
    main()
