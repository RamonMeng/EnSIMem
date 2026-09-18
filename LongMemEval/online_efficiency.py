"""Memora-compatible efficiency accounting for the online entity pipeline.

This module is deliberately observational.  It does not alter prompts, ranking,
selection, retries, or answer normalization.  Each stage records only the two
quantities used for the Memora comparison: wall-clock latency and the amount of
memory context exposed to the model (episode count and token estimate).
"""

from __future__ import annotations

import math
import statistics
import threading
from typing import Any


SCHEMA_VERSION = "ensi-memora-efficiency-v1"
TOKENIZER_NAME = "o200k_base"


class UsageTracker:
    """Thread-local request scopes used only to audit stage API calls.

    The entity pipeline currently exposes Memora's context/latency metrics, not
    billed-token metrics.  We still keep call metadata internally so a failed
    request scope cannot change control flow or be mistaken for a stage timer.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    def begin(self, label: str) -> None:
        if getattr(self._local, "scope", None) is not None:
            raise RuntimeError("an efficiency usage scope is already active")
        self._local.scope = {"label": str(label), "calls": []}

    def record(
        self,
        response: dict[str, Any],
        *,
        endpoint: str,
        model: str,
        wall_time_seconds: float,
        attempt_count: int,
    ) -> None:
        # Do not expose API usage in the output: Memora comparison uses context
        # load and latency.  Recording this internally makes the tracker safe
        # for retries without becoming an extra algorithmic signal.
        scope = getattr(self._local, "scope", None)
        if (
            scope is None
            or endpoint.strip("/") != "chat/completions"
            or not isinstance(response, dict)
        ):
            return
        scope["calls"].append(
            {
                "model": model,
                "attempt_count": int(attempt_count),
                "wall_time_seconds": round(float(wall_time_seconds), 6),
                "reported_usage": isinstance(response.get("usage"), dict),
            }
        )

    def end(self) -> dict[str, Any]:
        scope = getattr(self._local, "scope", None)
        if scope is None:
            raise RuntimeError("no efficiency usage scope is active")
        self._local.scope = None
        calls = scope["calls"]
        return {
            "scope_label": scope["label"],
            "llm_call_count": len(calls),
            "retry_count": sum(max(0, call["attempt_count"] - 1) for call in calls),
        }


def measure_context(text: str, episode_count: int) -> dict[str, Any]:
    """Return a non-invasive Memora-style context-load measurement.

    We prefer the same o200k_base tokenizer used in the benchmark summaries.
    The fallback is deterministic and explicitly labelled so measurements are
    never silently presented as exact API token counts.
    """
    raw = str(text or "")
    try:
        import tiktoken  # type: ignore

        tokens = len(tiktoken.get_encoding(TOKENIZER_NAME).encode(raw))
        method = f"tiktoken:{TOKENIZER_NAME}"
    except Exception:
        tokens = math.ceil(len(raw) / 4)
        method = "estimate:ceil(characters/4)"
    return {
        "episode_count": int(episode_count),
        "characters": len(raw),
        "utf8_bytes": len(raw.encode("utf-8")),
        "tokens": int(tokens),
        "token_count_method": method,
    }


def stage_metrics(
    *,
    stage: str,
    wall_time_seconds: float,
    context: dict[str, Any],
    context_role: str,
    query_id: str = "",
    search_steps: int | None = None,
) -> dict[str, Any]:
    """Create the stable per-query stage record consumed by reports."""
    return {
        "schema_version": SCHEMA_VERSION,
        "metric_source": "Memora-compatible online efficiency accounting",
        "stage": stage,
        "query_id": query_id,
        "latency_seconds": round(float(wall_time_seconds), 6),
        "context_role": context_role,
        "context": context,
        "search_steps": None if search_steps is None else int(search_steps),
    }


def summarize_stage(rows: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    """Summarize latency/context without affecting individual predictions."""
    metrics = [
        row.get("online_efficiency", {})
        for row in rows
        if row.get("online_efficiency", {}).get("stage") == stage
    ]
    latencies = [float(item["latency_seconds"]) for item in metrics]
    contexts = [
        float(item.get("context", {}).get("tokens"))
        for item in metrics
        if item.get("context", {}).get("tokens") is not None
    ]
    search_steps = [
        float(item["search_steps"])
        for item in metrics
        if item.get("search_steps") is not None
    ]
    pipeline_end_to_end = [
        float(item["memora_comparison"]["end_to_end_latency_seconds"])
        for item in metrics
        if item.get("memora_comparison", {}).get("end_to_end_latency_seconds") is not None
    ]
    pipeline_search = [
        float(item["memora_comparison"]["search_latency_seconds"])
        for item in metrics
        if item.get("memora_comparison", {}).get("search_latency_seconds") is not None
    ]

    def dist(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
        ordered = sorted(values)
        p50 = ordered[max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.50) - 1))]
        p95 = ordered[max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1))]
        return {
            "count": len(values),
            "mean": round(statistics.fmean(values), 6),
            "p50": round(p50, 6),
            "p95": round(p95, 6),
            "max": round(max(values), 6),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "metric_source": "Memora-compatible online efficiency accounting",
        "stage": stage,
        "query_count": len(metrics),
        "latency_seconds": dist(latencies),
        "context_tokens": dist(contexts),
        "search_steps": dist(search_steps),
        "memora_comparison": {
            "end_to_end_latency_seconds": dist(pipeline_end_to_end),
            "search_latency_seconds": dist(pipeline_search),
        },
    }
