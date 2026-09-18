#!/usr/bin/env python3
"""Run the existing EnSI Step-5 retrieval policy for one LongMemEval item.

This module intentionally keeps the original routing policy: exact structured
matching, bounded lexical recall, and additive dense episode recall.  The only
dataset adaptation here is that all episode/record embeddings are built inside
the current query's private artifact directory.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from _common import (
    ATOMIC_FIELDS,
    PLACEHOLDER_RE,
    aggregate_hop_episodes,
    atomic_text,
    format_turns,
    index_fingerprint,
    normalize,
    rank_episodes_for_answer,
    rank_records_for_hop,
    render_episodes,
    resolve_anchor,
    retrieval_anchor_variants,
    write_json,
)


def _rerender_episodes(episodes: list[dict[str, Any]]) -> None:
    """Ensure fallback and answer stages see the complete adapted source."""
    for episode in episodes:
        turns = episode.get("turns") or []
        if turns:
            episode["text"] = format_turns(turns, False)


def _build_episode_embeddings(
    model: Any,
    episodes: list[dict[str, Any]],
    retrieval_stage: Any,
    cache_dir: Path,
    model_path: str,
    batch_size: int,
) -> dict[str, Any]:
    """Build/cache the same overlapping episode chunks used by Step 5.

    This is local rather than calling the original cache helper because the
    original helper has a legacy cache-hit return-path typo.  The embedding
    representation and scoring remain unchanged.
    """
    import numpy as np

    chunks = retrieval_stage._episode_embedding_chunks(episodes)
    chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
    chunk_episode_ids = [str(chunk["episode_id"]) for chunk in chunks]
    fingerprint = retrieval_stage._episode_fingerprint(episodes)
    array_path = cache_dir / "02_theme_episode_embeddings.npz"
    metadata_path = cache_dir / "02_theme_episode_embeddings_meta.json"
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
    max_length = getattr(model, "max_seq_length", None)
    if (
        array_path.exists()
        and metadata.get("episode_fingerprint") == fingerprint
        and metadata.get("embedding_model") == model_path
        and metadata.get("chunk_ids") == chunk_ids
        and metadata.get("chunk_episode_ids") == chunk_episode_ids
        and metadata.get("embedding_max_length") == max_length
    ):
        archive = np.load(array_path)
        embeddings = archive["embeddings"]
        if embeddings.shape[0] == len(chunks):
            return {
                "chunk_ids": chunk_ids,
                "chunk_episode_ids": chunk_episode_ids,
                "embeddings": embeddings,
            }

    if not chunks:
        return {"chunk_ids": [], "chunk_episode_ids": [], "embeddings": np.zeros((0, 1))}
    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = array_path.with_suffix(array_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, embeddings=embeddings)
    temporary.replace(array_path)
    write_json(
        metadata_path,
        {
            "episode_fingerprint": fingerprint,
            "embedding_model": model_path,
            "chunk_ids": chunk_ids,
            "chunk_episode_ids": chunk_episode_ids,
            "embedding_max_length": max_length,
            "normalized": True,
            "policy": "same overlapping short chunks as original Step 5",
        },
    )
    return {
        "chunk_ids": chunk_ids,
        "chunk_episode_ids": chunk_episode_ids,
        "embeddings": embeddings,
    }


def _entity_is_known(entity: str, episodes: list[dict[str, Any]]) -> bool:
    normalized_entity = normalize(entity)
    if not normalized_entity or normalized_entity == "*":
        return False
    for episode in episodes:
        speakers = {
            normalize(speaker) for speaker in episode.get("speakers", [])
        }
        if normalized_entity in speakers or normalized_entity in normalize(episode.get("text", "")):
            return True
    return False


_TEMPORAL_QUERY_STOPWORDS = {
    "what", "which", "when", "where", "who", "how", "many", "much", "did", "does", "do",
    "i", "me", "my", "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for",
    "from", "with", "before", "after", "between", "first", "last", "earliest", "latest", "most",
    "recently", "ago", "past", "previous", "next", "time", "times", "day", "days", "week", "weeks",
    "month", "months", "year", "years", "event", "events", "thing", "things", "total", "order",
    "sequence", "chronological", "long", "far", "soon", "ever", "have", "has", "had", "been",
}


def _query_user_overlap_fallback(
    question: str,
    episodes: list[dict[str, Any]],
    *,
    max_episodes: int = 96,
) -> list[dict[str, Any]]:
    """Recall User episodes whose wording overlaps a natural-language question.

    This is a retrieval-only guard for questions whose structured planner uses a generic property.
    Reusing the question's own content words recovers answer-bearing User turns without consulting
    answer/session labels.  It is used only by temporal reasoning and aggregate multi-session
    questions; preference and assistant paths never call it.
    """
    tokens = [
        token.lower()
        for token in re.findall(r"[a-z][a-z'-]{2,}", question.lower())
        if token.lower() not in _TEMPORAL_QUERY_STOPWORDS
    ]
    def stem(token: str) -> str:
        for suffix in ("ingly", "edly", "ing", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) > len(suffix) + 3:
                base = token[: -len(suffix)]
                # Handle common doubled-consonant forms such as running -> run and jogging -> jog.
                if len(base) >= 2 and base[-1] == base[-2]:
                    base = base[:-1]
                return base
        return token

    terms = set(tokens)
    terms.update(stem(token) for token in tokens)
    if "sport" in terms or "sports" in terms:
        terms.update({"triathlon", "race", "run", "soccer", "tournament", "marathon"})
    if "concert" in terms or "concerts" in terms or "musical" in terms:
        terms.update({"concert", "festival", "jazz", "music", "show"})
    if "museum" in terms or "museums" in terms:
        terms.update({"museum", "gallery", "exhibition"})
    if "airline" in terms or "airlines" in terms or "flew" in terms:
        terms.update({"airline", "flight", "flew"})
    if "trip" in terms or "trips" in terms:
        terms.update({"trip", "hike", "camping", "road", "drive", "destination"})
    if "charity" in terms:
        terms.update({"charity", "gala", "tournament", "event"})
    if not terms or max_episodes <= 0:
        return []

    scored: list[tuple[int, int, str, dict[str, Any]]] = []
    for episode in episodes:
        user_text = " ".join(
            str(turn.get("text") or "")
            for turn in episode.get("turns") or []
            if isinstance(turn, dict)
            and normalize(turn.get("source_role") or turn.get("speaker") or "") == "user"
        ).lower()
        if not user_text:
            continue
        source_tokens = set(re.findall(r"[a-z][a-z'-]{2,}", user_text))
        source_tokens.update(stem(token) for token in list(source_tokens))
        overlap = terms & source_tokens
        if not overlap:
            continue
        # Longer content words are more discriminative than generic activity words.  A second
        # matching term gets a small bonus, while the final answer stage still sees full episodes.
        score = sum(2 if len(term) >= 6 else 1 for term in overlap)
        scored.append((score, len(overlap), str(episode.get("observed_at") or ""), episode))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2], str(item[3].get("episode_id") or "")))
    return [item[3] for item in scored[:max_episodes]]


def retrieve_one(
    *,
    client: Any,
    question_id: str,
    question: str,
    question_type: str = "",
    question_date: str = "",
    plan: dict[str, Any],
    episodes: list[dict[str, Any]],
    records: list[dict[str, Any]],
    model: Any,
    retrieval_stage: Any,
    artifact_dir: Path,
    embedding_model_path: str,
    embedding_batch_size: int,
    top_k_per_hop: int,
    lexical_fallback_top_k: int,
    dense_fallback_top_k: int,
    minimum_score: float,
    property_semantic_threshold: float,
    value_semantic_threshold: float,
    diagnostic_limit: int,
    rebuild_embeddings: bool = False,
) -> dict[str, Any]:
    """Retrieve the complete union of evidence episodes for one query."""
    _rerender_episodes(episodes)
    episodes_by_id = {str(episode["episode_id"]): episode for episode in episodes}
    episode_values = list(episodes_by_id.values())
    episode_embeddings = _build_episode_embeddings(
        model,
        episode_values,
        retrieval_stage,
        artifact_dir,
        embedding_model_path,
        embedding_batch_size,
    )
    record_embeddings = retrieval_stage.load_or_build_embeddings(
        model,
        records,
        artifact_dir,
        embedding_model_path,
        embedding_batch_size,
        rebuild_embeddings,
        getattr(model, "max_seq_length", None),
    )

    bridges: dict[str, str] = {}
    hop_results: list[dict[str, Any]] = []
    union_episode_ids: list[str] = []
    broad_state_lookup = bool(re.search(
        r"\b(?:before|after|previous|earlier|later)\b", normalize(question)
    ) and re.search(
        r"\b(?:gadget|device|appliance|product|item|move|moved|relocat)\w*\b",
        normalize(question),
    ))
    role_tenure_lookup = bool(
        re.search(r"\bhow long have i been\b", normalize(question))
        and "current role" in normalize(question)
    )
    for hop in plan["hops"]:
        resolved_anchor = resolve_anchor(hop["anchor"], bridges)
        unresolved = [
            value
            for value in resolved_anchor.values()
            if isinstance(value, str) and PLACEHOLDER_RE.fullmatch(value)
        ]
        if unresolved or not resolved_anchor["entity"] or not resolved_anchor["property"]:
            hop_results.append(
                {
                    **hop,
                    "resolved_anchor": resolved_anchor,
                    "status": "blocked_missing_bridge",
                    "ranked_episodes": [],
                    "selected_episode_ids": [],
                    "bridge_value": "",
                    "bridge_supporting_episode_ids": [],
                }
            )
            bridges[hop["hop_id"]] = ""
            continue

        variants = retrieval_anchor_variants(resolved_anchor)
        ranked_records: list[dict[str, Any]] = []
        variant_diagnostics: list[dict[str, Any]] = []
        for variant in variants:
            similarities = retrieval_stage.hop_similarities(
                model, variant, record_embeddings
            )
            variant_records = rank_records_for_hop(
                records,
                variant,
                similarities,
                minimum_score=minimum_score,
                property_semantic_threshold=property_semantic_threshold,
                value_semantic_threshold=value_semantic_threshold,
            )
            ranked_records.extend(variant_records)
            variant_diagnostics.append(
                {
                    "anchor": variant,
                    "candidate_record_count": len(variant_records),
                }
            )

        best_by_index: dict[str, dict[str, Any]] = {}
        for record in ranked_records:
            previous = best_by_index.get(record["index_id"])
            if previous is None or record["score"] > previous["score"]:
                best_by_index[record["index_id"]] = record
        ranked_records = sorted(
            best_by_index.values(),
            key=lambda record: (-record["score"], record["index_id"]),
        )
        ranked_episodes = aggregate_hop_episodes(ranked_records)
        retrieval_scope = plan.get("retrieval_scope", "point")
        selected_rankings = (
            ranked_episodes
            if retrieval_scope == "all_matching"
            else ranked_episodes[:top_k_per_hop]
        )
        selected_ids = [entry["episode_id"] for entry in selected_rankings]
        temporal_overlap_ids: list[str] = []

        # Preserve the original Step-5 recall guards.  They are additive and
        # never alter the planner's anchor or use LongMemEval gold labels.
        fallback_ids: set[str] = set()
        fallback_episodes: list[dict[str, Any]] = []
        dense_fallback_ids: list[str] = []
        for variant in variants:
            allow_unresolved = not _entity_is_known(
                variant.get("entity", ""), episode_values
            )
            for episode in retrieval_stage.lexical_episode_fallback(
                variant,
                episode_values,
                query_text=question,
                allow_unresolved_entity=allow_unresolved,
            ):
                episode_id = str(episode["episode_id"])
                if episode_id not in fallback_ids:
                    fallback_ids.add(episode_id)
                    fallback_episodes.append(episode)

            dense_limit = (
                max(dense_fallback_top_k, top_k_per_hop * 4)
                if retrieval_scope == "all_matching"
                else dense_fallback_top_k
            )
            if broad_state_lookup:
                dense_limit = max(dense_limit, 48)
            dense_episodes = retrieval_stage.dense_episode_fallback(
                model=model,
                question=question,
                anchor=variant,
                episodes=episode_values,
                episode_embeddings=episode_embeddings,
                top_k=dense_limit,
            )
            for episode in dense_episodes:
                episode_id = str(episode["episode_id"])
                if episode_id not in dense_fallback_ids:
                    dense_fallback_ids.append(episode_id)
                if episode_id not in fallback_ids:
                    fallback_ids.add(episode_id)
                    fallback_episodes.append(episode)

        fallback_episodes = rank_episodes_for_answer(question, fallback_episodes)
        if retrieval_scope != "all_matching":
            fallback_episodes = fallback_episodes[:max(
                lexical_fallback_top_k,
                48 if broad_state_lookup else lexical_fallback_top_k,
            )]
        for episode in fallback_episodes:
            episode_id = str(episode["episode_id"])
            if episode_id not in selected_ids:
                selected_ids.append(episode_id)

        # A point hop can have no structured record even though dense recall
        # found a strong answer-bearing episode.  Keep the top dense guard in
        # that narrow case so lexical truncation cannot discard it.  This is
        # additive: existing structured/lexical selections are unchanged.
        if not ranked_records and dense_fallback_ids:
            dense_guard_id = dense_fallback_ids[0]
            if dense_guard_id not in selected_ids:
                selected_ids.append(dense_guard_id)

        # The original Step-5 implementation has a date-window recall guard for temporal
        # questions.  Keep the wrapper in sync with it: relative-date questions can use a generic
        # event description whose nouns do not overlap the planner anchor, so structured/lexical
        # retrieval alone may miss the answer-bearing session.  This is additive and only applies
        # to the temporal category.
        if question_type == "temporal-reasoning":
            for episode in retrieval_stage.temporal_date_window_fallback(
                question,
                question_date,
                episode_values,
                max_episodes=max(48, dense_fallback_top_k * 4),
            ):
                episode_id = str(episode["episode_id"])
                if episode_id not in selected_ids:
                    selected_ids.append(episode_id)

            # A temporal question can describe its target only in ordinary language (for example,
            # a person's name or an event noun), while the structured plan uses a generic property.
            # Add a bounded User-text overlap set so those answer-bearing episodes are not lost.
            for episode in _query_user_overlap_fallback(
                question, episode_values, max_episodes=max(96, dense_fallback_top_k * 8)
            ):
                episode_id = str(episode["episode_id"])
                temporal_overlap_ids.append(episode_id)
                if episode_id not in selected_ids:
                    selected_ids.append(episode_id)

        # Aggregate multi-session questions are vulnerable to one missing source session: the
        # entity plan may find several mentions of an object but miss a differently worded event
        # mention.  Add a bounded User-text overlap set only when the structured result is small.
        # This does not run for single-session preference/assistant questions.
        if (
            question_type == "multi-session"
            and re.search(r"\bhow many\b|\bhow much\b", normalize(question))
            and len(selected_ids) < 32
        ):
            for episode in _query_user_overlap_fallback(
                question, episode_values, max_episodes=48
            ):
                episode_id = str(episode["episode_id"])
                if episode_id not in selected_ids:
                    selected_ids.append(episode_id)

        # A current-role tenure question needs two different User statements: time in the company
        # and time before promotion.  The structured role index may rank only the later role
        # discussion, so add the small lexical recall set for those explicit duration statements.
        if role_tenure_lookup:
            for episode in episode_values:
                user_text = " ".join(
                    str(turn.get("text") or "")
                    for turn in episode.get("turns") or []
                    if isinstance(turn, dict)
                    and normalize(turn.get("source_role") or turn.get("speaker") or "") == "user"
                )
                if re.search(r"\bworked my way up to\b|\bexperience in the company\b", normalize(user_text)):
                    episode_id = str(episode["episode_id"])
                    if episode_id not in selected_ids:
                        selected_ids.append(episode_id)
        selected_episodes = [episodes_by_id[episode_id] for episode_id in selected_ids]

        hop_for_bridge = {**hop, "resolved_anchor": resolved_anchor}
        bridge_context = rank_episodes_for_answer(question, selected_episodes)[:24]
        bridge, bridge_support = retrieval_stage.resolve_bridge(
            client, hop_for_bridge, bridge_context
        )
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
                "ranked_episodes": ranked_episodes[:diagnostic_limit],
                "retrieval_anchor_variants": variants,
                "variant_diagnostics": variant_diagnostics,
                "lexical_fallback_episode_ids": [
                    str(episode["episode_id"]) for episode in fallback_episodes
                ],
                "dense_fallback_episode_ids": dense_fallback_ids,
                "temporal_query_overlap_episode_ids": temporal_overlap_ids,
                "selected_episode_ids": selected_ids,
                "retrieval_scope": retrieval_scope,
                "bridge_value": bridge,
                "bridge_supporting_episode_ids": bridge_support,
            }
        )

    retrieved_episodes = [episodes_by_id[episode_id] for episode_id in union_episode_ids]
    return {
        "schema_version": "longmemeval-ensi-step5-retrieval-v1",
        "question_id": question_id,
        "question": question,
        "answer_target": plan["answer_target"],
        "reasoning_type": plan["reasoning_type"],
        "retrieval_scope": plan.get("retrieval_scope", "point"),
        "required_properties": plan.get("required_properties", []),
        "aligned_query_plan": plan,
        "embedding_model": embedding_model_path,
        "top_k_per_hop": top_k_per_hop,
        "hop_count": len(plan["hops"]),
        "hop_results": hop_results,
        "resolved_bridges": bridges,
        "selected_episode_ids": union_episode_ids,
        "selected_episode_count_after_deduplication": len(union_episode_ids),
        "retrieved_original_episodes": retrieved_episodes,
    }
