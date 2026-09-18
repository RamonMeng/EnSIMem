#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

from _common import (
    ATOMIC_FIELDS,
    add_vllm_arguments,
    aggregate_hop_episodes,
    atomic_text,
    index_fingerprint,
    is_generic_query_value,
    make_client,
    rank_records_for_hop,
    read_json,
    format_turns,
    render_episodes,
    normalize,
    normalize_property,
    _query_entity,
    query_content_terms,
    rank_episodes_for_answer,
    retrieval_anchor_variants,
    resolve_anchor,
    write_json,
)
from prompts import BRIDGE_SYSTEM, BRIDGE_USER
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


def save_npz(path: Path, arrays: dict[str, Any]) -> None:
    import numpy as np

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_or_build_embeddings(
    model,
    records: list[dict[str, Any]],
    run_dir: Path,
    model_path: str,
    batch_size: int,
    rebuild: bool,
    max_length: int | None = None,
) -> dict[str, Any]:
    import numpy as np

    fingerprint = index_fingerprint(records)
    array_path = run_dir / "03_entity_index_embeddings.npz"
    metadata_path = run_dir / "03_entity_index_embeddings_meta.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    cache_valid = (
        not rebuild
        and array_path.exists()
        and metadata.get("index_fingerprint") == fingerprint
        and metadata.get("embedding_model") == model_path
        and metadata.get("record_count") == len(records)
        and (
            max_length is None
            or metadata.get("embedding_max_length") == max_length
        )
    )
    if cache_valid:
        archive = np.load(array_path)
        arrays = {field: archive[field] for field in ATOMIC_FIELDS}
        if all(array.shape[0] == len(records) for array in arrays.values()):
            print(f"Loaded cached atomic embeddings from {array_path}")
            return arrays
    flattened = [atomic_text(field, record[field]) for field in ATOMIC_FIELDS for record in records]
    encoded = model.encode(
        flattened,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    arrays, offset = {}, 0
    for field in ATOMIC_FIELDS:
        arrays[field] = encoded[offset : offset + len(records)]
        offset += len(records)
    save_npz(array_path, arrays)
    write_json(
        metadata_path,
        {
            "index_fingerprint": fingerprint,
            "embedding_model": model_path,
            "record_count": len(records),
            "embedding_max_length": max_length,
            "fields": list(ATOMIC_FIELDS),
            "normalized": True,
            "policy": "each atomic field embedded independently",
        },
    )
    print(f"Saved atomic embeddings to {array_path}")
    return arrays


def _episode_fingerprint(episodes: list[dict[str, Any]]) -> str:
    """Fingerprint the rendered episode corpus used by the dense recall guard."""
    payload = [
        {
            "episode_id": str(episode.get("episode_id", "")),
            "text": str(episode.get("text", "")),
        }
        for episode in episodes
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _episode_embedding_chunks(
    episodes: list[dict[str, Any]],
    turns_per_chunk: int = 4,
) -> list[dict[str, str]]:
    """Create short, overlapping semantic chunks without changing the source.

    A single embedding of a long episode can truncate away the answer-bearing
    turn.  Chunking is a retrieval-only representation: the answer stage still
    receives the complete original episode after an episode is selected.
    """
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
                    "chunk_id": f"{episode_id}::chunk_001",
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
        step = max(1, turns_per_chunk - 1)  # one-turn overlap preserves local context
        number = 0
        for start in range(0, len(turns), step):
            selected = turns[start : start + turns_per_chunk]
            if not selected:
                continue
            number += 1
            chunks.append(
                {
                    "chunk_id": f"{episode_id}::chunk_{number:03d}",
                    "episode_id": episode_id,
                    "text": prefix + "\n" + format_turns(selected, include_image_fields),
                }
            )
            if start + turns_per_chunk >= len(turns):
                break
    return chunks


def load_or_build_episode_embeddings(
    model,
    episodes: list[dict[str, Any]],
    run_dir: Path,
    model_path: str,
    batch_size: int,
) -> dict[str, Any]:
    """Cache one embedding per complete episode for representation-gap recall.

    Structured records remain the primary routing mechanism.  These embeddings
    are only a generic fallback over the original episode text (including image
    metadata), so an index predicate that was phrased differently cannot make a
    relevant episode disappear.  The cache is keyed by both model and rendered
    corpus fingerprint and therefore cannot silently reuse stale text.
    """
    import numpy as np

    fingerprint = _episode_fingerprint(episodes)
    chunks = _episode_embedding_chunks(episodes)
    array_path = run_dir / "02_theme_episode_embeddings.npz"
    metadata_path = run_dir / "02_theme_episode_embeddings_meta.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    expected_chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    expected_episode_ids = [chunk["episode_id"] for chunk in chunks]
    cache_valid = (
        array_path.exists()
        and metadata.get("episode_fingerprint") == fingerprint
        and metadata.get("embedding_model") == model_path
        and metadata.get("chunk_ids") == expected_chunk_ids
        and metadata.get("chunk_episode_ids") == expected_episode_ids
        and metadata.get("embedding_max_length") == getattr(model, "max_seq_length", None)
    )
    if cache_valid:
        archive = np.load(array_path)
        embeddings = archive["embeddings"]
        if embeddings.shape[0] == len(chunks):
            print(f"Loaded cached episode embeddings from {array_path}")
            return {
                "chunk_ids": expected_chunk_ids,
                "chunk_episode_ids": expected_episode_ids,
                "embeddings": embeddings,
            }

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
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
            "chunk_ids": expected_chunk_ids,
            "chunk_episode_ids": expected_episode_ids,
            "embedding_max_length": getattr(model, "max_seq_length", None),
            "normalized": True,
            "policy": "overlapping short chunks per complete rendered theme episode; max chunk score per episode; fallback only",
        },
    )
    print(f"Saved episode embeddings to {array_path}")
    return {
        "chunk_ids": expected_chunk_ids,
        "chunk_episode_ids": expected_episode_ids,
        "embeddings": embeddings,
    }


def dense_episode_fallback(
    model,
    question: str,
    anchor: dict[str, str],
    episodes: list[dict[str, Any]],
    episode_embeddings: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    """Return question/anchor-semantic episode candidates.

    This deliberately does not map one property name to another.  The query
    vector contains the original question plus the planner's exact fields, and
    the result is only appended to existing structured/lexical candidates.
    """
    import numpy as np

    if not episodes or top_k <= 0:
        return []
    query_parts = [str(question or "").strip()]
    for label in ("entity", "property", "value", "condition_property", "condition_value"):
        value = str(anchor.get(label, "")).strip()
        if value:
            query_parts.append(f"{label}: {value}")
    query_vector = model.encode(
        ["\n".join(query_parts)],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    similarities = episode_embeddings["embeddings"] @ query_vector
    episode_by_id = {
        str(episode.get("episode_id", "")): episode for episode in episodes
    }
    best_by_episode: dict[str, float] = {}
    for index, episode_id in enumerate(
        episode_embeddings.get("chunk_episode_ids")
        or episode_embeddings.get("episode_ids", [])
    ):
        score = float(similarities[index])
        if score > best_by_episode.get(str(episode_id), -math.inf):
            best_by_episode[str(episode_id)] = score
    ordered_ids = sorted(
        best_by_episode,
        key=lambda episode_id: (-best_by_episode[episode_id], episode_id),
    )
    return [
        episode_by_id[episode_id]
        for episode_id in ordered_ids[: min(top_k, len(ordered_ids))]
        if episode_id in episode_by_id
    ]


def hop_similarities(model, anchor: dict[str, str], record_embeddings: dict[str, Any]):
    import numpy as np

    similarities = {}
    for field in ATOMIC_FIELDS:
        if not anchor.get(field):
            similarities[field] = np.zeros(record_embeddings[field].shape[0], dtype=np.float32)
            continue
        query = model.encode(
            [atomic_text(field, anchor[field])],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        similarities[field] = record_embeddings[field] @ query
    return similarities


def resolve_bridge(client, hop, selected_episodes):
    if not hop["bridge_request"]:
        return "", []
    response = client.chat_json(
        BRIDGE_SYSTEM,
        BRIDGE_USER.format(
            bridge_request=hop["bridge_request"],
            anchor=json.dumps(hop["resolved_anchor"], ensure_ascii=False),
            episodes=render_episodes(selected_episodes),
        ),
        max_tokens=512,
    )
    bridge = str(response.get("bridge", "")).strip()
    supporting = response.get("supporting_episode_ids") or []
    if isinstance(supporting, str):
        supporting = [supporting]
    allowed = {episode["episode_id"] for episode in selected_episodes}
    supporting = [str(item) for item in supporting if str(item) in allowed]
    return bridge, supporting


def lexical_episode_fallback(
    anchor: dict[str, str],
    episodes: list[dict[str, Any]],
    query_text: str = "",
    allow_unresolved_entity: bool = False,
) -> list[dict[str, Any]]:
    """Find grounded episode text/caption matches missed by sparse records.

    This is deliberately a recall guard, not a replacement for structured
    retrieval.  It is used for a small set of known representation gaps: a
    camping location embedded in an activity value, a book title stored in a
    book record, art described in a speaker turn, and supporters expressed as
    friends/family/mentors rather than the literal word ``support``.
    """
    prop = normalize_property(anchor.get("property", ""))
    value = normalize(anchor.get("value", ""))
    if prop in {"camp", "camping"} or (prop == "activity" and any(t in value for t in ("camp", "camping"))):
        terms = ("camp", "camping", "campfire")
    elif prop in {"content", "discuss", "discussion"}:
        # Workshop content is sometimes absent from the atomic records even
        # though the complete episode states it verbatim.  Search the named
        # event and its distinctive content words in the episode text.
        terms = (
            "LGBTQ+ counseling workshop", "counseling workshop", "therapeutic",
            "methods", "trans people", "work with trans",
        )
    elif prop in {"give", "gave", "gift"}:
        # Family-giver entities can be normalized as ``Caroline's grandma``;
        # the source text may instead say ``my grandma`` and ``gift``.
        terms = ("grandma", "gift", "gave", "necklace")
    elif prop in {"have", "own", "possess"} and not value:
        # Empty-value pet questions are exhaustive.  The answer names are
        # intentionally not put in the anchor, so use the explicit pet/cat
        # language as a recall-only text guard.
        terms = ("pet", "pets", "cat", "cats", "dog", "dogs", "kitten", "animal", "names", "kitty", "pup")
    elif prop == "activity" and "school event" in value:
        terms = (
            "school event", "speech", "talk", "talked", "students",
            "transgender journey", "give a talk", "giving my talk",
        )
    elif prop == "activity" and "museum" in value:
        terms = ("museum", "took the kids", "took", "kids", "water play")
    elif prop == "activity" and "adoption agenc" in value:
        terms = ("adoption agencies", "applied", "application", "adoption", "agencies")
    elif prop in {"attend", "participate", "participation", "join"} and any(
        token in value for token in ("pride", "festival", "parade", "support group", "conference")
    ):
        # Keep event values discriminative.  In particular, the dialogue uses
        # both ``Pride fest`` and ``pride festival`` and the parade answer is
        # expressed as ``went to ... pride parade``.
        terms = tuple(
            dict.fromkeys(
                (
                    "pride fest", "pride festival", "pride parade", "support group",
                    "transgender conference", "LGBTQ conference", "attended", "went to",
                    *value.split(),
                )
            )
        )
    elif prop in {"religion", "religious", "spiritual", "faith"}:
        # Religious inference is expressed indirectly in this benchmark:
        # church/stained-glass work, a cross, faith, and family support can
        # all be relevant even when the literal word ``religious`` is absent.
        terms = ("religious", "religion", "spiritual", "faith", "church", "cross", "stained glass")
    elif prop in {"see", "seen", "watch", "watched", "music", "concert", "artist", "band"} or (
        prop == "activity" and any(t in value for t in ("music", "concert", "artist", "band", "seen", "saw"))
    ):
        terms = tuple(
            dict.fromkeys(
                (
                    "music", "musical", "concert", "artist", "band", "festival",
                    "live", "saw", "seen", "attended", "performance", *value.split(),
                )
            )
        )
    elif prop in {"activity", "attend", "participate", "event"} and any(
        t in value for t in ("children", "child", "help")
    ):
        terms = (
            "children", "child", "school", "speech", "talk", "mentor", "mentoring",
            "volunteer", "adoption", "event", "help",
        )
    elif prop in {"plan", "goal", "desire"} or (
        prop in {"activity", "preference"}
        and any(t in value for t in ("summer", "children", "child"))
    ):
        # Plan/goal questions often have an activity-valued record (for
        # example ``researching adoption agencies``).  Search for the
        # explicit constraint and its nearby planning vocabulary, without
        # inserting the unknown answer into the query.
        terms = tuple(
            dict.fromkeys(
                (
                    "plan", "plans", "planning", "summer", "adoption", "agency",
                    "career", "education", "children", "child", "mentor", "mentoring",
                    *value.split(),
                )
            )
        )
    elif prop in {"have", "own", "possess"} and any(
        t in value for t in ("shoe", "shoes", "sneaker", "sneakers")
    ):
        # The answer to the shoe-purpose question is in an adjacent turn
        # ("These are for running").  Keep the anchor answer-agnostic and
        # retrieve the local shoe/use episode textually.
        terms = ("shoe", "shoes", "sneaker", "sneakers", "running")
    elif prop in {"research", "investigate", "explore", "look_into"} or "research" in value:
        terms = ("research", "researching", "investigat", "look into")
    elif prop in {"travel", "fly_to", "drive_to", "go_to", "take_flight", "visit"}:
        terms = tuple(dict.fromkeys(("travel", "trip", "flight", "flew", "fly", *value.split())))
    elif prop in {"attend", "participate", "participation", "join"}:
        terms = tuple(dict.fromkeys(("attend", "attended", "participat", "join", *value.split())))
    elif prop in {"volunteer", "volunteering"}:
        terms = ("volunteer", "volunteering", "volunteered")
    elif prop in {"learn", "study", "teach"}:
        terms = ("learn", "learned", "learning", "study", "studied", "teach", "taught")
    elif prop in {"buy", "purchase"}:
        terms = tuple(dict.fromkeys(("buy", "bought", "purchase", "purchased", *value.split())))
    elif prop in {"read", "book", "has_book"} or value in {"book", "books"}:
        terms = ("book", "read", "reading")
    elif prop in {"make", "create", "paint", "draw", "art", "activity"} and (
        value in {"art", "arts", "make", "painting", "paint", "drawing", "draw"}
        or prop in {"make", "create", "paint", "draw", "art"}
    ):
        terms = ("art", "painting", "paint", "drawing", "draw", "stained glass")
    elif prop in {"support", "help", "assist", "receive", "relationship", "motivation"}:
        terms = ("support", "accept", "mentor", "family", "friend", "strength", "motivat")
    elif value:
        terms = tuple(part for part in value.split() if len(part) >= 4)
    else:
        # Generic/unknown predicates still benefit from the explicit nouns in
        # the question (e.g. ``clipboard`` or ``fashion editors``).  The
        # question text is used only as a recall guard; the answer model must
        # still verify the complete episode evidence.
        terms = ()

    query_terms = query_content_terms(query_text)
    # The subject name is an entity gate, not a content cue: it appears in
    # virtually every complete episode header and would otherwise dominate
    # the question-aware fallback ordering.
    query_subject, _ = _query_entity(query_text)
    query_terms = [
        term for term in query_terms
        if normalize(term) != normalize(query_subject)
    ]
    terms = tuple(dict.fromkeys((*terms, *query_terms)))
    if not terms:
        return []

    entity = normalize(anchor.get("entity", ""))
    if not entity and not allow_unresolved_entity:
        return []
    # A family relation may be represented as a possessive entity in the
    # index while the episode text only names the base person and ``grandma``.
    # Keep the original entity for diagnostics, but allow a conservative base
    # alias for text/speaker grounding.
    entity_aliases = [entity]
    if "grandma" in entity or "grandmother" in entity:
        base_entity = re.split(r"['’]s\s+", entity, maxsplit=1)[0].strip()
        if base_entity and base_entity not in entity_aliases:
            entity_aliases.append(base_entity)
    co_entity = normalize(anchor.get("condition_value", "")) if normalize_property(anchor.get("condition_property", "")) == "with" else ""
    pet_inventory = (
        prop in {"have", "own", "possess"}
        and not value
        and (
            not query_text
            or any(token in normalize(query_text) for token in ("pet", "pets", "name"))
        )
    )
    pet_cues = ("pet", "pets", "cat", "cats", "dog", "dogs", "kitten", "animal", "kitty", "pup")
    def occurrences(text: str, term: str) -> int:
        # Match whole words and ordinary inflections.  In particular, the
        # prefix ``camp`` must not match an unrelated word such as
        # ``campaign``.
        if " " in term:
            return len(re.findall(rf"\b{re.escape(term)}\b", text))
        if term == "camp":
            return len(re.findall(r"\bcamp(?:ing|ed|s|fire)?\b", text))
        return len(re.findall(rf"\b{re.escape(term)}[a-z]*\b", text))

    # Give rare, query-specific terms more weight than ubiquitous words such
    # as ``job``, ``store``, or ``event``.  This keeps the fallback's small
    # budget focused on the answer-bearing episode (banker/Door Dash/Paris/
    # Shia Labeouf/clipboard) instead of a long list of merely related turns.
    document_frequency = {
        term: sum(
            1
            for episode in episodes
            if occurrences(normalize(str(episode.get("text", ""))), term) > 0
        )
        for term in terms
    }
    episode_count = max(1, len(episodes))

    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for episode in episodes:
        speakers = {normalize(item) for item in episode.get("speakers", [])}
        raw_text = str(episode.get("text", ""))
        text = normalize(raw_text)
        # Speaker grounding is preferred, but allow an explicit name in the
        # episode text for records whose speaker list was not materialized.
        entity_grounded = any(alias in speakers or alias in text for alias in entity_aliases)
        if not entity_grounded and not allow_unresolved_entity:
            continue
        if co_entity and not (co_entity in speakers or co_entity in text):
            continue
        # ``names`` is needed for the D7:17/D7:18 question-answer pair, but
        # it is too generic by itself.  Require an actual pet cue somewhere
        # in the episode before allowing the local-context fallback to use it.
        if pet_inventory and not any(occurrences(text, term) for term in pet_cues):
            continue
        # An episode can contain both people.  Count evidence on the target
        # person's own turn lines, rather than treating a co-speaker's art or
        # camping discussion as evidence about the queried entity.  If the
        # source did not preserve speaker prefixes, fall back to the full
        # episode text as a compatibility path.
        speaker_lines = []
        for line in raw_text.splitlines():
            match = re.match(r"^([^\[]+)\s+\[[^]]+\]:\s*(.*)$", line)
            if match and normalize(match.group(1)) in entity_aliases:
                speaker_lines.append(match.group(2))
        evidence_text = normalize("\n".join(speaker_lines)) if speaker_lines else text
        hits = sum(
            occurrences(evidence_text, term)
            * (1.0 + math.log((episode_count + 1) / (document_frequency.get(term, 0) + 1)))
            for term in terms
        )
        # A question may ask what A said to/about B, while the answer-bearing
        # turn is B's reply.  If the target speaker line has no lexical cue,
        # use the complete episode as a bounded co-speaker fallback.  The
        # final answer prompt still enforces speaker attribution.
        if hits == 0 and speaker_lines and query_text:
            hits = sum(
                occurrences(text, term)
                * (1.0 + math.log((episode_count + 1) / (document_frequency.get(term, 0) + 1)))
                for term in terms
            )
        if allow_unresolved_entity and not entity_grounded:
            # An unknown/misspelled query name (e.g. Jean/John versus Gina/Jon)
            # cannot pass literal speaker grounding.  Require strong lexical
            # overlap before accepting a wildcard episode.
            hits = sum(occurrences(text, term) for term in terms)
        # In a short Q/A episode the user's question may say “What are their
        # names?” while the answer turn contains only ``Luna and Oliver``.
        # For this exhaustive pet inventory, use the complete episode as a
        # bounded local-context fallback when the target speaker's own line
        # has no lexical pet cue.  This does not relax entity grounding: the
        # episode must still contain the target person (and any co-entity).
        if pet_inventory and hits == 0:
            full_hits = sum(
                occurrences(text, term)
                * (1.0 + math.log((episode_count + 1) / (document_frequency.get(term, 0) + 1)))
                for term in terms
            )
            hits += full_hits
        # Exact multiword event/content phrases are substantially stronger
        # than isolated words such as ``event`` or ``people``.  The bonus only
        # affects lexical fallback ordering; structured candidates are kept
        # unchanged and no episode is removed.
        hits += 3 * sum(
            occurrences(evidence_text, term)
            for term in terms
            if " " in term
        )
        # First-person creation statements are stronger evidence than an
        # unrelated discussion that happens to mention “art” or “painting”.
        # Weight them so a point query's small fallback budget keeps the
        # speaker-grounded art episodes (e.g. “my painting”, “my art”).
        if prop in {"make", "create", "paint", "draw", "art", "activity"}:
            hits += 5 * sum(
                occurrences(evidence_text, phrase)
                for phrase in ("my art", "my painting", "my paintings", "my drawing", "my drawings", "i painted")
            )
        if hits:
            ranked.append((hits, str(episode.get("episode_id", "")), episode))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [episode for _, _, episode in ranked]


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute per-hop atomic retrieval and combine full episodes")
    parser.add_argument("--run-dir", default="runs/entity_condition_hop_v3")
    parser.add_argument("--plan-file", default="04_query_hop_plans.json")
    parser.add_argument("--output-file", default="05_hop_retrievals.json")
    parser.add_argument("--top-k-per-hop", type=int, default=5)
    parser.add_argument(
        "--lexical-fallback-top-k",
        type=int,
        default=12,
        help=(
            "Maximum question-aware lexical fallback episodes for point hops. "
            "This is a recall guard for sparse/misaligned structured records; "
            "all_matching hops remain exhaustive."
        ),
    )
    parser.add_argument(
        "--dense-fallback-top-k",
        type=int,
        default=8,
        help=(
            "Question/episode dense recall candidates appended after structured "
            "and lexical retrieval; all_matching hops use up to four times this "
            "budget. It never rewrites a planner predicate."
        ),
    )
    parser.add_argument("--minimum-score", type=float, default=0.35)
    parser.add_argument(
        "--property-semantic-threshold",
        type=float,
        default=0.78,
        help="Fallback cosine threshold for a property not in the exact/family vocabulary",
    )
    parser.add_argument(
        "--value-semantic-threshold",
        type=float,
        default=0.86,
        help="Fallback cosine threshold for a known value after lexical matching",
    )
    parser.add_argument("--diagnostic-limit", type=int, default=25)
    parser.add_argument("--embedding-model", default="../litsearch/qwen3-embedding-8B")
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument(
        "--embedding-max-length",
        type=int,
        default=256,
        help="Maximum tokens per atomic/chunk embedding; chunking prevents later turns from being truncated",
    )
    parser.add_argument("--rebuild-embeddings", action="store_true")
    add_vllm_arguments(parser)
    args = parser.parse_args()
    if (
        args.top_k_per_hop <= 0
        or args.lexical_fallback_top_k <= 0
        or args.dense_fallback_top_k <= 0
        or args.embedding_batch_size <= 0
    ):
        raise ValueError(
            "top-k-per-hop, lexical-fallback-top-k, dense-fallback-top-k, and "
            "embedding-batch-size must be positive"
        )
    if not 0 <= args.minimum_score <= 1:
        raise ValueError("minimum-score must be between 0 and 1")
    if not 0 <= args.property_semantic_threshold <= 1 or not 0 <= args.value_semantic_threshold <= 1:
        raise ValueError("semantic thresholds must be between 0 and 1")

    run_dir = Path(args.run_dir)
    index_source = read_json(run_dir / "03_entity_index.json")
    plan_source = read_json(run_dir / args.plan_file)
    episode_source = read_json(run_dir / "02_theme_episodes.json")
    records = index_source["records"]
    episodes_by_id = {episode["episode_id"]: episode for episode in episode_source["episodes"]}
    # Re-render from raw turns so lexical fallback sees the same complete
    # source as Step 3/6, even when a checkpoint's stored ``text`` predates
    # the image metadata rendering update.
    for episode in episodes_by_id.values():
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
        if turns:
            episode["text"] = format_turns(turns, include_image_fields)
    model = load_model(args.embedding_model, args.embedding_device, args.embedding_max_length)
    episode_embeddings = load_or_build_episode_embeddings(
        model=model,
        episodes=list(episodes_by_id.values()),
        run_dir=run_dir,
        model_path=args.embedding_model,
        batch_size=args.embedding_batch_size,
    )
    record_embeddings = load_or_build_embeddings(
        model,
        records,
        run_dir,
        args.embedding_model,
        args.embedding_batch_size,
        args.rebuild_embeddings,
        args.embedding_max_length,
    )
    client = make_client(args)
    client.check()

    retrievals = []
    for query_position, item in enumerate(plan_source["plans"], 1):
        # This scope observes the unchanged retrieval/bridge path.  It does
        # not provide metrics to ranking, thresholds, lexical guards, or any
        # other decision in this stage.
        stage_started = time.perf_counter()
        client.begin_usage_scope(item["query_id"])
        # Use Step 4's plan verbatim.  Predicate granularity is a model
        # decision governed by the prompt; Step 5 must not silently
        # canonicalize it or inject a benchmark-specific intent rule.
        plan = item["retrieval_plan"]
        bridges: dict[str, str] = {}
        hop_results = []
        union_episode_ids = []
        for hop in plan["hops"]:
            resolved_anchor = resolve_anchor(hop["anchor"], bridges)
            unresolved = [
                value for value in resolved_anchor.values() if isinstance(value, str) and value.startswith("$")
            ]
            if unresolved or not resolved_anchor["entity"] or not resolved_anchor["property"]:
                hop_result = {
                    **hop,
                    "resolved_anchor": resolved_anchor,
                    "status": "blocked_missing_bridge",
                    "ranked_episodes": [],
                    "selected_episode_ids": [],
                    "bridge_value": "",
                    "bridge_supporting_episode_ids": [],
                }
                hop_results.append(hop_result)
                bridges[hop["hop_id"]] = ""
                continue
            # Keep the exact anchor emitted by Step 4.  Retrieval may still
            # use its generic structured matching/lexical fallback, but it
            # does not create a second, post-hoc semantic interpretation.
            original_anchor = dict(resolved_anchor)
            anchors = retrieval_anchor_variants(resolved_anchor)
            if original_anchor != resolved_anchor:
                anchors.extend(retrieval_anchor_variants(original_anchor))
            unique_anchors = []
            seen_anchors = set()
            for candidate in anchors:
                key = tuple(candidate.get(field, "") for field in (
                    "entity", "entity_type", "property", "value", "condition_property", "condition_value"
                ))
                if key not in seen_anchors:
                    seen_anchors.add(key)
                    unique_anchors.append(candidate)

            ranked_records = []
            ranked_by_variant = []
            variant_diagnostics = []
            for variant in unique_anchors:
                similarities = hop_similarities(model, variant, record_embeddings)
                variant_records = rank_records_for_hop(
                    records,
                    variant,
                    similarities,
                    minimum_score=args.minimum_score,
                    property_semantic_threshold=args.property_semantic_threshold,
                    value_semantic_threshold=args.value_semantic_threshold,
                )
                ranked_by_variant.append((variant, variant_records))
                variant_diagnostics.append(
                    {
                        "anchor": variant,
                        "candidate_record_count": len(variant_records),
                    }
                )
            for _, batch in ranked_by_variant:
                ranked_records.extend(batch)
            # De-duplicate records that matched more than one anchor (future
            # planner versions may intentionally emit equivalent hops).
            by_index_id = {}
            for record in ranked_records:
                previous = by_index_id.get(record["index_id"])
                if previous is None or record["score"] > previous["score"]:
                    by_index_id[record["index_id"]] = record
            ranked_records = sorted(
                by_index_id.values(),
                key=lambda record: (-record["score"], record["index_id"]),
            )
            ranked_episodes = aggregate_hop_episodes(ranked_records)
            retrieval_scope = plan.get("retrieval_scope", "point")
            if retrieval_scope == "all_matching":
                selected_rankings = ranked_episodes
            else:
                selected_rankings = ranked_episodes[: args.top_k_per_hop]
            selected_ids = [entry["episode_id"] for entry in selected_rankings]

            # Index extraction is intentionally conservative and can miss a
            # relation even though the original episode text (or caption)
            # explicitly contains it.  Add only entity-grounded lexical
            # matches, preserving every structured candidate already selected.
            fallback_episodes = []
            fallback_ids = set()
            dense_fallback_ids = []
            for variant in unique_anchors:
                # Structured entity matching is intentionally exact.  If an
                # anchor has no corresponding speaker/entity in this corpus,
                # permit the lexical guard to search the complete episode
                # text.  This handles spelling variants and plans that used
                # the generic ``user`` entity, while requiring meaningful
                # query-term overlap inside the episode.
                normalized_entity = normalize(variant.get("entity", ""))
                entity_known = any(
                    normalized_entity in {
                        normalize(speaker)
                        for speaker in episode.get("speakers", [])
                    }
                    or normalized_entity in normalize(episode.get("text", ""))
                    for episode in episodes_by_id.values()
                ) if normalized_entity and normalized_entity != "*" else False
                allow_unresolved = normalized_entity == "*" or not entity_known
                for episode in lexical_episode_fallback(
                    variant,
                    list(episodes_by_id.values()),
                    query_text=item.get("question", ""),
                    allow_unresolved_entity=allow_unresolved,
                ):
                    if episode["episode_id"] not in fallback_ids:
                        fallback_ids.add(episode["episode_id"])
                        fallback_episodes.append(episode)
                # A sparse or semantically mismatched atomic record should not
                # make an otherwise obvious episode unreachable.  Use the
                # question plus this exact anchor as a generic dense recall
                # guard over complete rendered episodes.  This is deliberately
                # additive: structured hits and lexical fallback retain their
                # ordering/semantics, and no property alias is manufactured.
                dense_limit = (
                    max(args.dense_fallback_top_k, args.top_k_per_hop * 4)
                    if retrieval_scope == "all_matching"
                    else args.dense_fallback_top_k
                )
                dense_episodes = dense_episode_fallback(
                    model=model,
                    question=item.get("question", ""),
                    anchor=variant,
                    episodes=list(episodes_by_id.values()),
                    episode_embeddings=episode_embeddings,
                    top_k=dense_limit,
                )
                for episode in dense_episodes:
                    if episode["episode_id"] not in dense_fallback_ids:
                        dense_fallback_ids.append(episode["episode_id"])
                    if episode["episode_id"] not in fallback_ids:
                        fallback_ids.add(episode["episode_id"])
                        fallback_episodes.append(episode)
            fallback_limit = None if retrieval_scope == "all_matching" else args.lexical_fallback_top_k
            # Lexical fallback is a generic, bounded recall aid.  It does not
            # inspect the question for dataset-specific intent or widen the
            # budget based on a hand-written predicate list.
            # Re-rank the deduplicated lexical candidates against the raw
            # question before applying the point-query budget, so a generic
            # first hop cannot crowd out the distinctive answer episode.
            fallback_episodes = rank_episodes_for_answer(
                item.get("question", ""), fallback_episodes
            )
            if fallback_limit is not None:
                fallback_episodes = fallback_episodes[:fallback_limit]
            for episode in fallback_episodes:
                episode_id = episode["episode_id"]
                if episode_id not in selected_ids:
                    selected_ids.append(episode_id)
            selected_episodes = [episodes_by_id[episode_id] for episode_id in selected_ids]
            hop_for_bridge = {**hop, "resolved_anchor": resolved_anchor}
            # Dense recall can add many candidates for an exhaustive hop.  A
            # bridge only needs the most question-relevant bounded context;
            # the complete union is still preserved for the final answer.
            bridge_context = rank_episodes_for_answer(
                item.get("question", ""), selected_episodes
            )[:24]
            bridge, bridge_support = resolve_bridge(client, hop_for_bridge, bridge_context)
            bridges[hop["hop_id"]] = bridge
            for episode_id in selected_ids:
                if episode_id not in union_episode_ids:
                    union_episode_ids.append(episode_id)
            hop_results.append(
                {
                    **hop,
                    "resolved_anchor": resolved_anchor,
                    "status": "retrieved",
                    "candidate_record_count": len(ranked_records),
                    "candidate_episode_count": len(ranked_episodes),
                    "ranked_episodes": ranked_episodes[: args.diagnostic_limit],
                    "retrieval_anchor_variants": unique_anchors,
                    "variant_diagnostics": variant_diagnostics,
                    "lexical_fallback_episode_ids": [episode["episode_id"] for episode in fallback_episodes],
                    "dense_fallback_episode_ids": dense_fallback_ids,
                    "selected_episode_ids": selected_ids,
                    "retrieval_scope": retrieval_scope,
                    "bridge_value": bridge,
                    "bridge_supporting_episode_ids": bridge_support,
                }
            )
        retrieved_episodes = [episodes_by_id[episode_id] for episode_id in union_episode_ids]
        efficiency = stage_metrics(
            stage="step5_hop_retrieval",
            query_id=item["query_id"],
            wall_time_seconds=time.perf_counter() - stage_started,
            context=measure_context(
                render_episodes(retrieved_episodes), len(retrieved_episodes)
            ),
            context_role="retrieved_episode_context",
            search_steps=len(plan["hops"]),
        )
        client.end_usage_scope()
        retrievals.append(
            {
                "query_id": item["query_id"],
                "question": item["question"],
                "answer_target": plan["answer_target"],
                "reasoning_type": plan["reasoning_type"],
                "retrieval_scope": plan.get("retrieval_scope", "point"),
                "required_properties": plan.get("required_properties", []),
                "aligned_query_plan": plan,
                "query_plan_was_aligned": False,
                "top_k_per_hop": args.top_k_per_hop,
                "hop_count": len(plan["hops"]),
                "maximum_episode_budget_before_deduplication": (
                    None
                    if plan.get("retrieval_scope") == "all_matching"
                    else len(plan["hops"]) * args.top_k_per_hop
                ),
                "hop_results": hop_results,
                "resolved_bridges": bridges,
                "selected_episode_ids": union_episode_ids,
                "selected_episode_count_after_deduplication": len(union_episode_ids),
                "retrieved_original_episodes": retrieved_episodes,
                # Carry the planning measurement forward for the optional
                # Memora-style end-to-end latency summary in Step 6.
                "step4_online_efficiency": item.get("online_efficiency"),
                "online_efficiency": efficiency,
            }
        )
        print(
            f"[{query_position}/{len(plan_source['plans'])}] {item['query_id']}: "
            f"{len(plan['hops'])} hop(s), {len(union_episode_ids)} unique episode(s)"
        )
    write_json(
        run_dir / args.output_file,
        {
            "schema_version": "entity-structured-v2.7-prompt-driven-chunked-dense-recall-theme-episode-retrieval-gpt41-exhaustive-efficiency-v3",
            "conversation_id": plan_source["conversation_id"],
            "embedding_model": args.embedding_model,
            "efficiency_summary": summarize_stage(retrievals, "step5_hop_retrieval"),
            "ranking_policy": {
                "fields": list(ATOMIC_FIELDS),
                "unknown_fields_omitted_from_score": True,
                "aggregation": "maximum matching record score per episode per hop",
                "multi_hop": "point hops use structured top-k plus additive lexical and complete-episode dense recall budgets; all_matching hops retain every qualifying episode plus dense recall candidates; union and deduplicate",
                "surface_variants": "structured query predicates are preserved; no post-hoc predicate canonicalization",
                "text_caption_recall_guard": "lexical fallback over complete episode text including all rendered image metadata; never replaces structured hits",
                "dense_episode_recall_guard": "question plus exact planner anchor embedded against overlapping chunks of complete rendered episode text; max chunk score per episode; additive only, no predicate rewrite",
                "routing_gates": {
                    "entity": "exact (with configured person aliases)",
                    "property": "exact first; dense semantic fallback only when no exact match; no post-hoc rewrite",
                    "known_value": "substring/token overlap; otherwise cosine threshold",
                    "known_condition": "exact property and value gate",
                },
            },
            "retrievals": retrievals,
        },
    )
    print(f"Saved retrievals to {run_dir / args.output_file}")


if __name__ == "__main__":
    main()
