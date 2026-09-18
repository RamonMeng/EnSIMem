#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from _common import evidence_coverage, expanded_evidence, is_unknown, read_json, token_f1, write_json


def aggregate(rows):
    f1 = [row["answer_f1"] for row in rows]
    coverage = [row["evidence_coverage"] for row in rows if row["evidence_coverage"] is not None]
    adversarial = [row["adversarial_correct"] for row in rows if row["adversarial_correct"] is not None]
    return {
        "count": len(rows),
        "answer_f1": round(sum(f1) / len(f1), 4) if f1 else None,
        "evidence_coverage": round(sum(coverage) / len(coverage), 4) if coverage else None,
        "adversarial_accuracy": round(sum(adversarial) / len(adversarial), 4) if adversarial else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score entity-hop retrieval and predictions")
    parser.add_argument("--run-dir", default="runs/entity_condition_hop_v3")
    parser.add_argument("--gold-file", default="01_gold_DO_NOT_USE_BEFORE_STEP_7.json")
    parser.add_argument("--retrieval-file", default="05_hop_retrievals.json")
    parser.add_argument("--predictions-file", default="06_hop_predictions.json")
    parser.add_argument("--output-file", default="07_hop_scores.json")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    gold_source = read_json(run_dir / args.gold_file)
    retrieval_source = read_json(run_dir / args.retrieval_file)
    prediction_source = read_json(run_dir / args.predictions_file)
    gold = {item["query_id"]: item for item in gold_source["gold"]}
    retrievals = {item["query_id"]: item for item in retrieval_source["retrievals"]}
    predictions = {item["query_id"]: item for item in prediction_source["predictions"]}
    details = []
    for query_id, target in gold.items():
        if query_id not in retrievals or query_id not in predictions:
            raise ValueError(f"Missing retrieval or prediction for {query_id}")
        retrieval = retrievals[query_id]
        prediction = predictions[query_id]["prediction"]
        retrieved_dia_ids = list(
            dict.fromkeys(
                dia_id
                for episode in retrieval["retrieved_original_episodes"]
                for dia_id in episode["dia_ids"]
            )
        )
        category = int(target["category"])
        answer_f1 = token_f1(prediction, str(target.get("answer", "")))
        coverage = None if category == 5 else evidence_coverage(
            expanded_evidence(target.get("evidence") or []), retrieved_dia_ids
        )
        adversarial = None
        if category == 5:
            adversarial = answer_f1 >= 0.999 if target.get("has_explicit_answer") else is_unknown(prediction)
        details.append(
            {
                "query_id": query_id,
                "category": category,
                "question": predictions[query_id]["question"],
                "gold_answer": target.get("answer", ""),
                "prediction": prediction,
                "answer_f1": round(answer_f1, 4),
                "gold_evidence": target.get("evidence") or [],
                "retrieved_episode_ids": retrieval["selected_episode_ids"],
                "retrieved_dia_ids": retrieved_dia_ids,
                "evidence_coverage": coverage,
                "adversarial_correct": adversarial,
                "hop_count": retrieval["hop_count"],
            }
        )
    by_category = defaultdict(list)
    for row in details:
        by_category[str(row["category"])].append(row)
    result = {
        "conversation_id": gold_source["conversation_id"],
        "overall": aggregate(details),
        "by_category": {category: aggregate(rows) for category, rows in sorted(by_category.items())},
        "details": details,
    }
    write_json(run_dir / args.output_file, result)
    print(f"Saved scores to {run_dir / args.output_file}")
    print(result["overall"])


if __name__ == "__main__":
    main()
