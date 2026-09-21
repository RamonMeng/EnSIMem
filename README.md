[README.md](https://github.com/user-attachments/files/32451247/README.md)
# EnSIMem# EnSIMem: Reproducibility Package

This package contains the executable code used for the EnSIMem experiments on
LoCoMo and LongMemEval.  EnSIMem is an entity-structured long-term memory
system designed to turn a long-context memory problem into a short-context
reasoning problem.  The code is organized as resumable stages so that a
reviewer can inspect the intermediate episodes, index records, query plans,
retrieved evidence, generated answers, and evaluation results.

The two benchmark directories are self-contained program bundles:

```text
EnSI-Memory/
├── README.md                 # this overview
├── LoCoMo/                   # LoCoMo pipeline and detailed instructions
│   └── README.md
└── LongMemEval/              # LongMemEval pipeline and detailed instructions
    └── README.md
```

The detailed benchmark-specific READMEs contain the complete command lines and
all options.  This file explains the common design and how the two programs
fit together.

## 1. EnSIMem in one paragraph

EnSIMem preserves the original interaction evidence while adding a structured
addressing layer.  During offline construction, each conversation is divided
into contiguous, theme-coherent episodes.  An entity--property index is then
built from the dialogue, with records conceptually organized as
`[entity][entity type][property:value]` and linked back to the source episode
and dialogue turns.  At query time, a planner converts a user request into
evidence requirements, including the answer target, property, reasoning type,
and retrieval scope.  Structured entity/type/property matching retrieves
candidate episodes; lexical and dense fallback paths protect recall when the
wording differs from the index.  Retrieved episodes are expanded to their
original source text before answer generation, so the answer model reasons over
grounded evidence rather than over a lossy summary or an anonymous text chunk.

The implementation therefore has three separable concerns:

1. **Memory construction:** sessions -> theme episodes -> entity--property
   records.
2. **Evidence access:** query planning -> structured matching -> fallback
   retrieval -> episode deduplication and evidence budgeting.
3. **Grounded evaluation:** answer generation from the selected source
   episodes, followed by deterministic diagnostics and a benchmark-compatible
   evaluation-model score.

## 2. Common pipeline

The numbered scripts in both directories follow the same high-level order:

```text
raw benchmark data
  -> 01 prepare/adapt sessions and questions
  -> 02 partition sessions into theme-coherent episodes
  -> 03 build the entity--property index
  -> 04 plan query hops/evidence requirements
  -> 05 retrieve and deduplicate source episodes
  -> 06 generate an answer from the retrieved episodes
  -> 07 compute benchmark diagnostics or evaluate answers
```

The exact file names differ slightly between benchmarks.  A new run should use
a fresh output directory, because the intermediate JSON files record the
dataset selection, model names, retrieval settings, and schema version used to
create them.  Reusing a directory is appropriate only when resuming the same
configuration.

## 3. LoCoMo

### Benchmark

LoCoMo evaluates long-term conversational memory over multi-session, often
multimodal interactions.  The released examples contain ordered sessions with
speaker turns, timestamps, dialogue identifiers, and optional image-related
fields, together with questions and gold answers/evidence.  The questions
stress several memory behaviors, including multi-hop reasoning, temporal
grounding, open-domain recall, focused single-hop recall, and aggregation or
unanswerable cases handled by the benchmark protocol.

The LoCoMo implementation keeps the source dialogue and provenance available
throughout the pipeline.  The headline binary accuracy is produced by the
Memora-compatible evaluation script (`08_memora_llm_judge.py`); `07_score.py`
also reports token-level answer F1 and evidence coverage for debugging.  The
evaluation-model output is not used to construct the index or retrieve
evidence.

### Program stages

The main files are:

```text
00_run_remaining_conversations.py       run Steps 1--8 for selected conversations
01_prepare_sessions.py                  select questions and prepare sessions
02_partition_theme_episodes.py          construct theme-coherent episodes
03_build_entity_index.py                extract entity--property records
04_plan_query_hops.py                   plan evidence requirements
05_retrieve_hops.py                     structured + lexical/dense retrieval
05_retrieve_hops_true_version.py        alternate retrieval implementation
06_answer_from_hops.py                  generate grounded answers
07_score.py                             deterministic diagnostics
08_memora_llm_judge.py                  Memora-compatible evaluation-model score
09_build_episode_granularity_ablation.py
10_build_property_granularity_ablation.py
11_retrieve_dense_only.py               dense-only retrieval ablation
online_efficiency.py                    optional latency/token accounting
```

The standard paper configuration uses theme-coherent episodes, the broad
intermediate property prompt, five structured candidates per hop, lexical and
dense recall guards, and adaptive stopping based on the planned question type.
The controlled ablations change one design choice at a time:

* per-turn or per-session episodes;
* fine, broad-plus, broad, or deliberately very-broad properties;
* dense-only retrieval;
* fixed top-k values of 1, 3, 5, 8, 15, or 20 instead of adaptive budgets.

### Minimal command pattern

Set the paths and endpoints first (use your own dataset and model locations):

```bash
export ROOT=/path/to/EnSI-Memory
export STEPS="$ROOT/LoCoMo"
export LOCOMO_JSON=/path/to/locomo10.json
export RUN="$ROOT/runs/locomo_reproduction"
export EMBEDDING_MODEL=/path/to/qwen3-embedding-8B
export OPENAI_API_KEY='read-from-your-secret-manager'
export OPENAI_BASE_URL='https://api.openai.com/v1'
```

For the paper's offline partitioning configuration, an OpenAI-compatible Qwen
server may be started on a selected GPU:

```bash
CUDA_VISIBLE_DEVICES=3 vllm serve /path/to/qwen3-32B \
  --served-model-name qwen3-32B --host 127.0.0.1 --port 8000
```

Then run the numbered scripts from `$STEPS` using the complete commands in
[`LoCoMo/README.md`](LoCoMo/README.md).  The orchestration helper can run all
selected conversations:

```bash
python "$STEPS/00_run_remaining_conversations.py" \
  --steps "$STEPS" \
  --locomo "$LOCOMO_JSON" \
  --output-root "$ROOT/runs/locomo_all" \
  --exclude-conversation-ids '' \
  --judge-script "$STEPS/08_memora_llm_judge.py" \
  --embedding-model "$EMBEDDING_MODEL"
```

For a single run, the expected artifacts are:

```text
01_sessions.json
01_questions.json
01_gold_DO_NOT_USE_BEFORE_STEP_7.json
02_theme_episodes.json
03_entity_index.json
04_query_hop_plans.json
05_entity_structured_retrievals.json
06_entity_structured_predictions.json
07_entity_structured_scores.json
08_memora_judge.json
```

The gold file is isolated for scoring and audit.  It must not be supplied to
Steps 1--6 when producing a benchmark result.

## 4. LongMemEval

### Benchmark

LongMemEval evaluates interactive long-term memory over a conversation history
and a later user question.  Its categories distinguish different memory
requirements rather than treating every query as generic retrieval:

```text
single-session-user
single-session-assistant
single-session-preference
multi-session
knowledge-update
temporal-reasoning
```

The adapter converts each LongMemEval item to the EnSIMem session/question
format while retaining the original question ID, sessions, timestamps, gold
answer, and benchmark metadata.  Queries are processed independently and
sequentially by the main runner.  Gold answers and gold evidence labels are
used only after answer generation by the evaluation stage.

### Program stages

```text
run_longmemeval.py             complete Steps 1--7 for selected queries
run_online_longmemeval.py      reuse Steps 1--3 and run only Steps 4--6
01_prepare_longmemeval.py      adapt one benchmark item
02_partition_theme_episodes.py construct episodes
03_build_entity_index.py      construct the entity--property index
04_plan_query_hops.py          plan evidence requirements
05_retrieve_hops.py            retrieve source episodes
06_answer_longmemeval.py       generate an answer
07_evaluate_longmemeval.py     evaluation-model scoring
aggregate_longmemeval_results.py
                                aggregate category/run summaries
```

### Minimal command pattern

```bash
export ROOT=/path/to/EnSI-Memory
export STEPS="$ROOT/LongMemEval"
export DATASET=/path/to/longmemeval_s_cleaned.json
export RUN="$ROOT/runs/longmemeval_reproduction"
export EMBEDDING_MODEL=/path/to/qwen3-embedding-8B
export OPENAI_API_KEY='read-from-your-secret-manager'

cd "$STEPS"
python run_longmemeval.py \
  --dataset "$DATASET" \
  --original-dir "$STEPS" \
  --output-dir "$RUN" \
  --stop-after judge \
  --provider openai \
  --base-url https://api.openai.com/v1 \
  --model gpt-4.1-mini-2025-04-14 \
  --gpt-base-url https://api.openai.com/v1 \
  --gpt-model gpt-4.1-mini-2025-04-14 \
  --judge-base-url https://api.openai.com/v1 \
  --judge-model gpt-4o-mini-2024-07-18 \
  --embedding-model "$EMBEDDING_MODEL"
```

To use local Qwen3 for partitioning, change the partitioning provider and
endpoint to `vllm` while leaving the GPT-compatible options for Steps 3--6.
For a rerun in which Steps 1--3 already exist, use
`run_online_longmemeval.py`.  To score existing predictions only, use
`07_evaluate_longmemeval.py`; this does not regenerate answers.

Per-query artifacts are stored under `queries/<question_id>/`:

```text
01_sessions.json
01_questions.json
01_longmemeval_metadata.json
02_theme_episodes.json
03_entity_index.json
04_query_hop_plan.json
05_retrieval.json
06_generation_trace.json
06_prediction.json
07_judge.json
```

The run-level `run_manifest.json`, `predictions.jsonl`, and summary JSON files
should be archived with the final result.  Use
[`LongMemEval/README.md`](LongMemEval/README.md) for category selection,
judge-only evaluation, and aggregation commands.

## 5. Requirements and configuration

* Python 3.10 or newer.
* PyTorch and `sentence-transformers` for the local embedding model; a CUDA
  GPU is strongly recommended.
* An OpenAI-compatible endpoint for answer generation and evaluation-model
  scoring.  A local vLLM endpoint can be used for Qwen3 stages.
* The exact benchmark data files and embedding checkpoint used for the target
  experiment.  These are intentionally not bundled in the source directory.

Install the common Python dependencies in a clean environment:

```bash
python -m pip install --upgrade pip
python -m pip install 'numpy>=1.24' 'sentence-transformers>=3.4.1' torch
```

The clients use Python's standard HTTP library; installing the OpenAI Python
SDK is not required.  `tiktoken` is optional and is used only for detailed
token accounting in the LongMemEval efficiency utilities.

Generation and evaluation requests use zero temperature in the supplied
configuration.  Exact model IDs, endpoints, selection manifests, retrieval
thresholds, embedding paths, and output manifests should be recorded because
changing any of them can change the result.  Keep the GPT-4o-mini and Qwen3
evaluation-model outputs in separate files; they are alternative evaluation
models, not values to average into a single headline score.

## 6. Reproducibility and safety notes

1. Use the official, unmodified benchmark files and the same question IDs for
   the baseline and all ablations.
2. Start every new configuration in a fresh run directory; resume only a run
   with matching metadata.
3. Do not pass gold evidence to preprocessing, planning, retrieval, or answer
   generation.  Gold data is isolated for evaluation.
4. If `CUDA_VISIBLE_DEVICES=3` is used, the first visible device is addressed
   inside Python as `cuda:0`.
5. Rebuild embedding caches whenever the embedding checkpoint, maximum length,
   or rendered episode text changes.
6. Never commit API keys, private endpoints, model weights, generated runs,
   `__pycache__`, `.pyc`, or `.DS_Store` files.

For full argument lists, checkpoint rules, ablation commands, and troubleshooting,
see [`LoCoMo/README.md`](LoCoMo/README.md) and
[`LongMemEval/README.md`](LongMemEval/README.md).
