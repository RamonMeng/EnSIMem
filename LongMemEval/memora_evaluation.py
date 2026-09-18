#!/usr/bin/env python3
"""Memora-style LongMemEval answer evaluator.

This is the evaluator used by the earlier EnSI LongMemEval implementation.
It is intentionally separate from answer generation: the model answer is
generated first, then the gold answer is supplied only to this judge.
"""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any


def _chat_user_only(client: Any, prompt: str, max_tokens: int = 64) -> str:
    """Call Chat Completions with exactly one user message.

    The prior Memora-style evaluator uses this protocol.  The bundled EnSI
    OpenAIClient already owns request/retry/usage handling, so this helper
    reuses its private transport without changing generation behavior.
    """
    payload = {
        "model": client.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
    }
    # Qwen3 served by vLLM may spend a short max-token budget on hidden
    # reasoning and never reach its yes/no answer.  This request is only for
    # the local judge endpoint; normal OpenAI requests remain unchanged.
    base_url = str(getattr(client, "base_url", "")).lower()
    if "127.0.0.1:8000" in base_url or "localhost:8000" in base_url:
        payload["seed"] = 0
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    response = client._request("chat/completions", payload)
    try:
        return str(response["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"malformed judge response: {response}") from error


def judge_prediction(client: Any, sample: dict[str, Any], hypothesis: str) -> dict[str, Any]:
    """Apply the task-specific checker from the existing Memora-style evaluator."""
    question = str(sample["question"])
    answer = str(sample["answer"])
    question_id = str(sample["question_id"])
    question_type = str(sample.get("question_type", ""))

    if question_id.endswith("_abs"):
        prompt = (
            "I will give you an unanswerable question, an explanation, and a response from a model. "
            "Answer yes if the model correctly identifies the question as unanswerable. It may say "
            "the information is incomplete or that the requested information is not given.\n\n"
            f"Question: {question}\n\nExplanation: {answer}\n\n"
            f"Model Response: {hypothesis}\n\n"
            "Does the model correctly identify the question as unanswerable? Answer yes or no only."
        )
    elif question_type == "single-session-preference":
        prompt = (
            "I will give you a question, a rubric for the desired personalized response, and a model "
            "response. Answer yes if the response satisfies the rubric. It need not reflect every rubric "
            "point; it is correct if it recalls and uses the user's personal information correctly.\n\n"
            f"Question: {question}\n\nRubric: {answer}\n\n"
            f"Model Response: {hypothesis}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    elif question_type == "knowledge-update":
        prompt = (
            "I will give you a question, a correct answer, and a model response. Answer yes if the "
            "response contains or is equivalent to the correct answer. If it also contains previous "
            "information, still answer yes as long as the updated answer is present.\n\n"
            f"Question: {question}\n\nCorrect Answer: {answer}\n\n"
            f"Model Response: {hypothesis}\n\n"
            "Is the model response correct? Answer yes or no only."
        )
    else:
        temporal_note = (
            " Do not penalize an off-by-one error when the question asks for a number of days, weeks, "
            "or months."
            if question_type == "temporal-reasoning"
            else ""
        )
        prompt = (
            "I will give you a question, a correct answer, and a model response. Answer yes if the "
            "response contains or is equivalent to the correct answer, or contains all intermediate "
            "steps needed to obtain it. Answer no if it contains only a subset of required information."
            f"{temporal_note}\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\n"
            f"Model Response: {hypothesis}\n\nIs the model response correct? Answer yes or no only."
        )

    raw = _chat_user_only(client, prompt, max_tokens=64)
    # Prefer the final standalone yes/no token.  This is robust to a
    # compatible judge returning a short explanation despite the instruction.
    decisions = re.findall(r"\b(yes|no)\b", raw.lower())
    correct = decisions[-1] == "yes" if decisions else False
    return {
        "correct": correct,
        "raw_evaluator_response": raw,
        "evaluation_protocol": "Memora-style LongMemEval task-specific answer checker",
    }


def build_evaluation_metadata(
    sample: dict[str, Any],
    retrieval: dict[str, Any] | None,
    selected_episode_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Attach official gold data and retrieval diagnostics after generation.

    ``answer_session_ids`` is the official LongMemEval evidence-dialog label.
    Retrieved episodes retain the original source ``session_id``, so the
    diagnostic can compare the two without exposing gold data to Steps 1-6.
    The retrieval flag is recorded for analysis and is not passed into the
    correctness-judge prompt.
    """
    gold_source = [str(value) for value in sample.get("answer_session_ids", [])]
    retrieval = retrieval if isinstance(retrieval, dict) else {}
    retrieved_episodes = retrieval.get("retrieved_original_episodes")
    if not isinstance(retrieved_episodes, list):
        retrieved_episodes = retrieval.get("retrieved_episodes") or []

    def source_ids(episodes: list[Any]) -> list[str]:
        values: list[str] = []
        for episode in episodes:
            if not isinstance(episode, dict):
                continue
            session_id = str(
                episode.get("source_session_id", episode.get("session_id", ""))
            ).strip()
            if session_id and session_id not in values:
                values.append(session_id)
        return values

    retrieved_source = source_ids(retrieved_episodes)
    episode_by_id = {
        str(episode.get("episode_id")): episode
        for episode in retrieved_episodes
        if isinstance(episode, dict) and episode.get("episode_id")
    }
    selected_episodes = [
        episode_by_id[str(episode_id)]
        for episode_id in (selected_episode_ids or [])
        if str(episode_id) in episode_by_id
    ]
    selected_source = source_ids(selected_episodes)
    retrieved_gold = [item for item in gold_source if item in retrieved_source]
    reasoning_gold = [item for item in gold_source if item in selected_source]

    return {
        "gold_attached_after_generation": True,
        "gold_answer": str(sample.get("answer", "")),
        "gold_answer_session_ids": gold_source,
        "gold_dialog_count": len(gold_source),
        "retrieved_gold_session_ids": retrieved_gold,
        "missing_gold_session_ids_after_step5": [
            item for item in gold_source if item not in retrieved_source
        ],
        "reasoning_gold_session_ids": reasoning_gold,
        "missing_gold_session_ids_after_step6": [
            item for item in gold_source if item not in selected_source
        ],
        "all_gold_sessions_retrieved": all(
            item in retrieved_source for item in gold_source
        ),
        "all_gold_sessions_in_reasoning": all(
            item in selected_source for item in gold_source
        ),
    }


def summarize_judges(rows: list[dict[str, Any]], judge_model: str) -> dict[str, Any]:
    """Aggregate judge results overall, by type, and for abstention items."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    abstention_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row.get("judge"), dict):
            continue
        grouped[str(row.get("question_type", "unknown"))].append(row)
        if str(row.get("question_id", "")).endswith("_abs"):
            abstention_rows.append(row)

    def metrics(values: list[dict[str, Any]]) -> dict[str, Any]:
        correct = sum(bool(row["judge"].get("correct")) for row in values)
        return {
            "judged_count": len(values),
            "correct_count": correct,
            "accuracy": round(correct / len(values), 4) if values else None,
        }

    by_type = {key: metrics(value) for key, value in sorted(grouped.items())}
    overall = metrics([row for values in grouped.values() for row in values])
    retrieval_rows = [
        row.get("evaluation", {})
        for row in rows
        if isinstance(row.get("evaluation"), dict)
        and row.get("evaluation", {}).get("gold_answer_session_ids") is not None
    ]
    gold_dialog_total = sum(
        len(item.get("gold_answer_session_ids", [])) for item in retrieval_rows
    )
    retrieved_gold_dialog_total = sum(
        len(item.get("retrieved_gold_session_ids", [])) for item in retrieval_rows
    )
    all_retrieved_count = sum(
        bool(item.get("all_gold_sessions_retrieved")) for item in retrieval_rows
    )
    return {
        "schema_version": "longmemeval-memora-style-evaluation-v1",
        "judge_model": judge_model,
        "overall": overall,
        "by_question_type": by_type,
        "abstention": metrics(abstention_rows),
        "gold_dialog_retrieval": {
            "diagnostic_query_count": len(retrieval_rows),
            "queries_with_all_gold_dialogs_retrieved": all_retrieved_count,
            "all_gold_dialogs_retrieved_rate": (
                round(all_retrieved_count / len(retrieval_rows), 4)
                if retrieval_rows
                else None
            ),
            "gold_dialog_count": gold_dialog_total,
            "retrieved_gold_dialog_count": retrieved_gold_dialog_total,
            "gold_dialog_recall": (
                round(retrieved_gold_dialog_total / gold_dialog_total, 4)
                if gold_dialog_total
                else None
            ),
        },
    }
