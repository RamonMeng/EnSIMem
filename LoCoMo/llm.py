from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from json import JSONDecoder
from typing import Any

from online_efficiency import UsageTracker


class LLMError(RuntimeError):
    pass


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object, tolerating markdown and Qwen think blocks."""
    cleaned = text.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[-1].strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        cleaned = "\n".join(lines).strip()
    decoder = JSONDecoder()
    for position, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise LLMError(f"No valid JSON object in model output: {text[:500]!r}")


class VLLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout: int = 180,
        retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        # Thread-local and observational: calls made outside an explicitly
        # opened scope are not recorded and therefore behave exactly as before.
        self.usage_tracker = UsageTracker()

    def begin_usage_scope(self, label: str) -> None:
        self.usage_tracker.begin(label)

    def end_usage_scope(self) -> dict[str, Any]:
        return self.usage_tracker.end()

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/{path.lstrip('/')}",
            data=None if payload is None else json.dumps(payload).encode("utf-8"),
            method="GET" if payload is None else "POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        last_error: Exception | None = None
        started = time.perf_counter()
        attempt_count = 0
        for attempt in range(self.retries):
            attempt_count = attempt + 1
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    value = json.loads(response.read().decode("utf-8"))
                    self.usage_tracker.record(
                        value,
                        endpoint=path,
                        model=self.model,
                        wall_time_seconds=time.perf_counter() - started,
                        attempt_count=attempt_count,
                    )
                    return value
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                TimeoutError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
        raise LLMError(f"vLLM request failed: {last_error}")

    def check(self) -> dict[str, Any]:
        return self._request("models")

    def chat(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        response = self._request(
            "chat/completions",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "top_p": 1.0,
                "max_tokens": max_tokens,
                "seed": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        try:
            content = str(response["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected vLLM response: {response}") from exc
        if "</think>" in content:
            content = content.rsplit("</think>", 1)[-1].strip()
        return content

    def chat_json(self, system: str, user: str, *, max_tokens: int = 2048) -> dict[str, Any]:
        first = self.chat(system, user, max_tokens=max_tokens)
        try:
            return extract_json_object(first)
        except LLMError:
            repaired = self.chat(
                "Repair the supplied output into one valid JSON object. Output JSON only.",
                first,
                max_tokens=max_tokens,
            )
            return extract_json_object(repaired)


class OpenAIClient(VLLMClient):
    """OpenAI-compatible client without vLLM-only request fields."""

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/{path.lstrip('/')}",
            data=None if payload is None else json.dumps(payload).encode("utf-8"),
            method="GET" if payload is None else "POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        last_error: Exception | None = None
        started = time.perf_counter()
        attempt_count = 0
        for attempt in range(self.retries):
            attempt_count = attempt + 1
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    value = json.loads(response.read().decode("utf-8"))
                    self.usage_tracker.record(
                        value,
                        endpoint=path,
                        model=self.model,
                        wall_time_seconds=time.perf_counter() - started,
                        attempt_count=attempt_count,
                    )
                    return value
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
        raise LLMError(f"OpenAI request failed: {last_error}")

    def chat(self, system: str, user: str, *, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        response = self._request(
            "chat/completions",
            {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        try:
            return str(response["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected OpenAI response: {response}") from exc

    def chat_json(self, system: str, user: str, *, max_tokens: int = 2048) -> dict[str, Any]:
        """Use Chat Completions JSON mode for extraction/planning, then validate locally."""
        response = self._request(
            "chat/completions",
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            },
        )
        try:
            content = str(response["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected OpenAI response: {response}") from exc
        try:
            return extract_json_object(content)
        except LLMError:
            # A single repair request is retained for transient/truncated responses.
            repaired = self.chat(
                "Repair the supplied output into one valid JSON object. Output JSON only.",
                content,
                max_tokens=max_tokens,
            )
            return extract_json_object(repaired)
