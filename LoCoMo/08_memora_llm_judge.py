#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from llm import extract_json_object
from _common import read_json, write_json


# Verbatim from the Mem0 evaluation/metrics/llm_judge.py revision used by
# Memora's Appendix B evaluation protocol (latest revision before the paper).
ACCURACY_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a ’gold’ (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


class OpenAIJudgeClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int, retries: int):
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.retries = retries

    def judge(self, question: str, gold_answer: str, generated_answer: str) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": ACCURACY_PROMPT.format(
                        question=question,
                        gold_answer=gold_answer,
                        generated_answer=generated_answer,
                    ),
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "seed": 42,
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content = str(body["choices"][0]["message"]["content"] or "")
                parsed = extract_json_object(content)
                label = str(parsed.get("label", "")).strip().upper()
                if label not in {"CORRECT", "WRONG"}:
                    raise ValueError(f"invalid judge label: {label!r}")
                return {"label": label, "raw_response": parsed}
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                TimeoutError,
                json.JSONDecodeError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ) as error:
                last_error = error
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Memora judge request failed: {last_error}")


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(int(row["score"]))
    scores = [int(row["score"]) for row in rows]
    return {
        "count": len(scores),
        "correct": sum(scores),
        "llm_judge_accuracy": round(sum(scores) / len(scores), 4) if scores else None,
        "by_category": {
            category: {
                "count": len(values),
                "correct": sum(values),
                "llm_judge_accuracy": round(sum(values) / len(values), 4),
            }
            for category, values in sorted(by_category.items(), key=lambda item: int(item[0]))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 8: Memora-identical LoCoMo LLM-as-a-judge")
    parser.add_argument("--run-dir", default="runs/property_coverage_v12")
    parser.add_argument("--input-file", default="06_structured_reasoning_predictions_gpt41_mini.json")
    parser.add_argument("--output-file", default="08_memora_llm_judge.json")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    input_path = run_dir / args.input_file
    source = read_json(input_path)
    predictions = source.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("input file must contain a predictions list")
    destination = run_dir / args.output_file
    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    prompt_sha256 = hashlib.sha256(ACCURACY_PROMPT.encode("utf-8")).hexdigest()
    checkpoint = read_json(destination) if destination.exists() else {
        "schema_version": "memora-locomo-llm-judge-v1",
        "protocol_source": (
            "Memora Appendix B -> Mem0 evaluation/metrics/llm_judge.py at "
            "aae5989e78a6188b3b047c104d960c9ad0927e75"
        ),
        "model": args.model,
        "temperature": 0.0,
        "seed": 42,
        "category_policy": "categories 1-4 scored; category 5 skipped",
        "input_file": args.input_file,
        "input_sha256": input_sha256,
        "prompt_sha256": prompt_sha256,
        "judgments": [],
    }
    expected = {
        "schema_version": "memora-locomo-llm-judge-v1",
        "model": args.model,
        "input_file": args.input_file,
        "input_sha256": input_sha256,
        "prompt_sha256": prompt_sha256,
    }
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError("Existing Step 8 checkpoint belongs to another configuration")

    client = OpenAIJudgeClient(args.base_url, args.api_key, args.model, args.timeout, args.retries)
    completed = {row["query_id"] for row in checkpoint["judgments"]}
    eligible = [row for row in predictions if int(row.get("gold_category", 0)) != 5]
    for position, row in enumerate(eligible, 1):
        query_id = str(row["query_id"])
        if query_id in completed:
            print(f"[{position}/{len(eligible)}] skip {query_id}")
            continue
        result = client.judge(
            str(row["question"]),
            str(row.get("gold_answer", "")),
            str(row.get("prediction", "")),
        )
        score = int(result["label"] == "CORRECT")
        checkpoint["judgments"].append(
            {
                "query_id": query_id,
                "category": int(row.get("gold_category", 0)),
                "question": row["question"],
                "gold_answer": row.get("gold_answer", ""),
                "generated_answer": row.get("prediction", ""),
                "label": result["label"],
                "score": score,
                "raw_response": result["raw_response"],
            }
        )
        checkpoint["summary"] = aggregate(checkpoint["judgments"])
        write_json(destination, checkpoint)
        print(f"[{position}/{len(eligible)}] {query_id}: {result['label']}")
    checkpoint["summary"] = aggregate(checkpoint["judgments"])
    write_json(destination, checkpoint)
    print(f"Saved Memora judge results to {destination}")
    print(checkpoint["summary"])


if __name__ == "__main__":
    main()
