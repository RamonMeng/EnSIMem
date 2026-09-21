# EnSI-Memory on LongMemEval

This directory contains the LongMemEval adaptation of EnSI-Memory. It keeps the
LoCoMo-style structured memory construction and retrieval pipeline, while adding
a LongMemEval adapter and evidence-grounded answer generation.

The benchmark data, embedding checkpoint, API keys, and generated run artifacts
are intentionally kept outside this source directory.

## 1. Pipeline

Each query is processed independently and sequentially:

~~~text
LongMemEval item
  -> Step 1: adapt and prepare sessions
  -> Step 2: partition sessions into theme episodes
  -> Step 3: build an entity/property index
  -> Step 4: plan query hops
  -> Step 5: retrieve evidence episodes
  -> Step 6: generate an answer from retrieved evidence
  -> Step 7: evaluate with the Memora-style judge
~~~

Gold answers and gold evidence labels are not supplied to Steps 1--6. They are
used only by Step 7 after the answer is generated.

The main entry point is run_longmemeval.py. It completes one query before
moving to the next query. run_online_longmemeval.py reuses existing Step 1--3
artifacts and runs only Steps 4--6.

## 2. Source files

The current executable and dependency set is:

~~~text
run_longmemeval.py
run_online_longmemeval.py

01_prepare_longmemeval.py
02_partition_theme_episodes.py
03_build_entity_index.py
04_plan_query_hops.py
05_retrieve_hops.py
06_answer_longmemeval.py
07_evaluate_longmemeval.py

_common.py
llm.py
online_efficiency.py
prompts.py

longmemeval_adapter.py
longmemeval_generation.py
longmemeval_retrieval.py
memora_evaluation.py
aggregate_longmemeval_results.py
__init__.py
~~~

aggregate_longmemeval_results.py is only needed for cross-run score aggregation.

## 3. Requirements

Use Python 3.10 or newer. The code uses urllib from the Python standard
library for API requests; an OpenAI Python SDK is not required.

~~~bash
pip install numpy sentence-transformers torch
~~~

Install a CUDA-compatible PyTorch build when the local embedding model runs on
a GPU. tiktoken is optional and is used only for detailed token accounting:

~~~bash
pip install tiktoken
~~~

If Step 2 or Step 7 is served by a local vLLM OpenAI-compatible server, install
and launch vLLM separately. The Python code only needs the HTTP endpoint.

External resources:

~~~text
longmemeval_s_cleaned.json
qwen3-embedding-8B/
~~~

Use exactly the same cleaned dataset and embedding checkpoint used for the
reported experiment. Do not commit API keys or model weights.

## 4. Configuration

Set the API key in the environment:

~~~bash
export OPENAI_API_KEY="..."
~~~

The scripts use OPENAI_API_KEY by default for OpenAI-compatible API calls.
For a local vLLM endpoint that accepts a placeholder key, use:

~~~text
--api-key EMPTY
~~~

Run commands from this directory. Passing --original-dir "$(pwd)" is
recommended so the runner loads all shared EnSI modules from this checkout.

## 5. Complete Step 1--7 run

This example runs 50 knowledge-update queries with GPT for Step 2 through
Step 6 and GPT-4o-mini for Step 7. Remove --question-type, --start, and
--limit to process the complete dataset.

~~~bash
cd /path/to/LongMemEval

DATASET=/path/to/data/longmemeval_s_cleaned.json
OUT=/path/to/runs/knowledge_update_full
EMBEDDING_MODEL=/path/to/litsearch/qwen3-embedding-8B

export OPENAI_API_KEY="..."

CUDA_VISIBLE_DEVICES=3,4 python3 run_longmemeval.py \
  --dataset "$DATASET" \
  --original-dir "$(pwd)" \
  --output-dir "$OUT" \
  --question-type knowledge-update \
  --start 0 \
  --limit 50 \
  --stop-after judge \
  --partition-workers 4 \
  --partition-batch-size 4 \
  --index-workers 12 \
  --index-batch-size 4 \
  --provider openai \
  --base-url https://api.openai.com/v1 \
  --model gpt-4.1-mini-2025-04-14 \
  --timeout 900 \
  --gpt-base-url https://api.openai.com/v1 \
  --gpt-model gpt-4.1-mini-2025-04-14 \
  --gpt-timeout 900 \
  --judge-base-url https://api.openai.com/v1 \
  --judge-model gpt-4o-mini-2024-07-18 \
  --judge-timeout 900 \
  --embedding-model "$EMBEDDING_MODEL" \
  --embedding-batch-size 32
~~~

The all-GPT configuration uses gpt-4.1-mini for Step 2 through Step 6.
To use local Qwen3 for Step 2 while retaining GPT for Steps 3--6, replace the
Step-2 options with:

~~~bash
  --provider vllm \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen3-32B \
  --api-key EMPTY
~~~

Supported question types are:

~~~text
single-session-user
single-session-assistant
single-session-preference
multi-session
knowledge-update
temporal-reasoning
~~~

Useful selection options:

~~~bash
# First 45 items of one category.
--question-type single-session-user --start 0 --limit 45

# Exclude previous queries; repeat as needed.
--exclude-question-id 0a995998 \
--exclude-question-id 6d550036

# Select the same number from each category.
--per-category 7 --selection-seed 42
~~~

## 6. Preprocessing only: Steps 1--3

~~~bash
CUDA_VISIBLE_DEVICES=3,4 python3 run_longmemeval.py \
  --dataset "$DATASET" \
  --original-dir "$(pwd)" \
  --output-dir "$OUT" \
  --question-type knowledge-update \
  --start 0 \
  --limit 11 \
  --stop-after index \
  --partition-workers 4 \
  --partition-batch-size 4 \
  --index-workers 12 \
  --index-batch-size 4 \
  --provider openai \
  --base-url https://api.openai.com/v1 \
  --model gpt-4.1-mini-2025-04-14 \
  --timeout 900 \
  --gpt-base-url https://api.openai.com/v1 \
  --gpt-model gpt-4.1-mini-2025-04-14 \
  --gpt-timeout 900 \
  --embedding-model "$EMBEDDING_MODEL"
~~~

--stop-after index stops after Step 3. A complete query directory should
contain:

~~~text
01_sessions.json
02_theme_episodes.json
03_entity_index.json
~~~

## 7. Reuse preprocessing: Steps 4--6

When Step 1--3 artifacts already exist, use:

~~~bash
CUDA_VISIBLE_DEVICES=3,4 python3 run_online_longmemeval.py \
  --dataset "$DATASET" \
  --original-dir "$(pwd)" \
  --run-dir "$OUT" \
  --question-type knowledge-update \
  --gpt-base-url https://api.openai.com/v1 \
  --gpt-model gpt-4.1-mini-2025-04-14 \
  --gpt-timeout 900 \
  --embedding-model "$EMBEDDING_MODEL" \
  --embedding-batch-size 32 \
  --max-reasoning-episodes 20 \
  --force
~~~

To rerun selected existing queries, repeat --question-id:

~~~bash
  --question-id 0a995998 \
  --question-id 6d550036
~~~

The online runner ends after Step 6. Use the standalone evaluator for Step 7.

## 8. Judge-only evaluation

07_evaluate_longmemeval.py reads saved 06_prediction.json files. It does not
rerun preprocessing, planning, retrieval, or generation.

GPT-4o-mini judge:

~~~bash
python3 07_evaluate_longmemeval.py \
  --dataset "$DATASET" \
  --run-dir "$OUT" \
  --base-url https://api.openai.com/v1 \
  --model gpt-4o-mini-2024-07-18 \
  --timeout 900 \
  --force
~~~

Qwen3-32B judge through local vLLM on port 8000:

~~~bash
python3 07_evaluate_longmemeval.py \
  --dataset "$DATASET" \
  --run-dir "$OUT" \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen3-32B \
  --api-key EMPTY \
  --timeout 900 \
  --force
~~~

For Qwen3 judging, memora_evaluation.py disables hidden thinking for the
yes/no checker and uses a sufficient output budget. Without this, a ten-token
request can be consumed before Qwen3 emits its final decision, making correct
answers appear false.

--force overwrites 07_judge.json and updates evaluation metadata in
06_prediction.json. It does not change Steps 1--6.

## 9. Aggregate all six categories

After judging the intended final run directories:

~~~bash
RUN_ROOT=/path/to/LongMemEval/runs
OLD_ROOT=/path/to/other/EnSI/runs

python3 aggregate_longmemeval_results.py \
  --dataset "$DATASET" \
  --run-root "$RUN_ROOT" \
  --run-root "$OLD_ROOT" \
  --judge-model qwen3-32B \
  --output "$RUN_ROOT/all_categories_qwen3_summary.json"
~~~

Use --judge-model gpt-4o-mini-2024-07-18 for GPT judge results. The script
reports correct / total and accuracy for every category and overall. Duplicate
question IDs are removed; v2 runs are preferred over v1 runs.

## 10. Output artifacts

Each query is stored under output-dir/queries/question_id/:

~~~text
01_sessions.json
01_questions.json
01_longmemeval_metadata.json
02_theme_episodes.json
02_theme_episode_embeddings.npz
03_entity_index.json
03_entity_index_embeddings.npz
04_query_hop_plan.json
05_retrieval.json
06_generation_trace.json
06_prediction.json
07_judge.json
~~~

Run-level files include:

~~~text
07_results_and_context_summary.json
predictions.jsonl
predictions_with_metadata.json
run_manifest.json
~~~

run_manifest.json records dataset, model, selection, and workflow metadata and
should be archived with final results.

## 11. Reproducibility notes

1. Use the exact same cleaned dataset and embedding checkpoint.
2. Record exact model names, endpoints, and date-specific model IDs.
3. Keep selection arguments, exclusions, concurrency, retrieval thresholds, and
   max-reasoning-episodes unchanged.
4. Keep GPT and Qwen judge results separate; they are alternative evaluation
   models and should not be averaged into one score.
5. Temperature is zero, but API and model-server revisions can still produce
   small differences. Archive run_manifest.json and generated artifacts.
6. gold_dialogs_retrieved is a retrieval diagnostic, not the Step-7 correctness
   decision.

## 12. Troubleshooting

### original EnSI common module not found

Run from this directory and pass:

~~~bash
--original-dir "$(pwd)"
~~~

The shared _common.py, llm.py, online_efficiency.py, prompts.py, and Step 2--5
modules must be present there.

### ModuleNotFoundError: longmemeval_retrieval

Make sure longmemeval_retrieval.py is present and run from the program
directory.

### answer-only requires existing Step 4/5 artifacts

The exact output directory must contain Step 4 and Step 5 files. If only
Step 1--3 exist, use run_online_longmemeval.py to create Steps 4--6 first.

### Remote end closed connection or IncompleteRead

Reuse the same output directory so completed query artifacts remain available,
then retry with lower worker counts or smaller batch sizes. Do not delete
completed query directories unless a fresh run is intended.

### Qwen judge returns zero for every query

Inspect 07_judge.json and raw_evaluator_response. Ensure the current
memora_evaluation.py is installed and Qwen3 hidden thinking is disabled.
Rerun only Step 7 with 07_evaluate_longmemeval.py --force.

## 13. Excluded files

Historical backups, one-off shell scripts, old runners, logs, __pycache__,
model weights, API keys, and generated runs/ directories are not required in a
clean source checkout. Archive them separately when provenance is needed.

