# LoCoMo reproduction instructions for EnSIMem

This directory contains the LoCoMo implementation used for the EnSIMem experiments.  The
pipeline is intentionally split into resumable stages so that a reviewer can inspect every
intermediate artifact (episodes, entity--property records, query plans, retrieved evidence,
answers, and evaluation results).

The commands below use only relative/generic paths.  Replace `/path/to/EnSI-Memory` with the
directory containing this `LoCoMo/` folder; do not hard-code a machine-specific path in a
submitted run script.

## 1. What is evaluated

The standard pipeline is:

1. prepare the raw LoCoMo conversation and questions;
2. partition each session into contiguous, theme-coherent episodes;
3. extract dialogue-grounded records of the form
   `[entity][entity_type][property:value][condition_property:condition_value]`;
4. decompose each question into evidence-seeking hops;
5. retrieve and deduplicate complete source episodes using structured matching plus lexical and
   dense recall guards;
6. generate an answer from the retrieved source episodes;
7. compute deterministic token-F1/evidence-coverage diagnostics; and
8. apply the Memora-compatible binary evaluation protocol to obtain the reported accuracy.

The answer model, extraction model, planner, retriever, and evaluation model are separate
configuration choices.  Changing any of them changes the experiment and should be recorded in
the output directory name.

## 2. Expected directory layout

The scripts do not require this exact layout when absolute paths are supplied, but the default
arguments assume:

```text
EnSI-Memory/
├── LoCoMo/                         # this directory
├── data/locomo10.json              # official LoCoMo JSON release
├── litsearch/qwen3-embedding-8B/   # local Sentence-Transformers checkpoint
└── runs/                           # created by the commands below
```

`locomo10.json` must be a JSON list.  Each item must contain a `sample_id`, a `conversation`
object with `session_1`, `session_2`, ... and corresponding timestamps, and a `qa` list with
question, answer, category, and evidence fields.  Use the official LoCoMo release; do not alter
the conversation text, dialogue IDs, timestamps, image-caption fields, or QA annotations.

For exact reproduction of a reported subset, preserve the same question IDs and order.  The
optional `--selection-file` argument accepts a JSON object with a `queries` list and is the safest
way to record a fixed subset.  If no selection manifest is used, set
`--questions-per-category 0` to evaluate every question in the selected conversation.

## 3. Software and hardware requirements

* Python 3.10 or newer (the scripts use modern type annotations and
  `argparse.BooleanOptionalAction`).
* A CUDA-capable GPU is recommended for the local embedding model.  CPU execution is possible
  but substantially slower.
* A local OpenAI-compatible vLLM server for the Qwen partitioning stage.  The default endpoint
  is `http://127.0.0.1:8000/v1` and the default model name is `qwen3-32B`.
* An OpenAI-compatible endpoint for Steps 3--6.  The paper configuration uses
  `gpt-4.1-mini-2025-04-14` at `https://api.openai.com/v1`.
* An OpenAI-compatible endpoint for Step 8.  The paper configuration uses `gpt-4o-mini`.
* Python packages:

  ```bash
  python -m pip install --upgrade pip
  python -m pip install 'numpy>=1.24' 'sentence-transformers>=3.4.1'
  ```

  Install a CUDA-compatible PyTorch build before `sentence-transformers` if the environment does
  not already provide one.  The LLM clients use Python's standard `urllib`; the `openai` Python
  package is not required by these scripts.

The local embedding checkpoint is not downloaded by the scripts.  Place the exact
`qwen3-embedding-8B` Sentence-Transformers directory at the path supplied to
`--embedding-model`.  Embedding vectors are normalized before cosine scoring.

## 4. Start the local Qwen server

Start a vLLM OpenAI-compatible server before Step 2.  The exact model path depends on the
reviewer's model installation; the endpoint must expose `/v1/models` and
`/v1/chat/completions`.

```bash
export ROOT=/path/to/EnSI-Memory
export STEPS="$ROOT/LoCoMo"
export QWEN_MODEL=/path/to/qwen3-32B

# Example: reserve GPU 3 for vLLM.
CUDA_VISIBLE_DEVICES=3 \
  vllm serve "$QWEN_MODEL" \
    --served-model-name qwen3-32B \
    --host 127.0.0.1 --port 8000
```

If a remote or already-running OpenAI-compatible endpoint is used, set
`QWEN_BASE_URL` to its `/v1` URL and pass `--base-url "$QWEN_BASE_URL"` to Step 2.  The local
client sends `temperature=0`, `top_p=1`, `seed=0`, and disables Qwen thinking blocks; keep those
settings unchanged for the paper configuration.

## 5. Configure the OpenAI-compatible endpoint

Never put a real key in a command committed to the supplementary package.  Export it in the
shell or provide it through the execution environment:

```bash
export OPENAI_API_KEY='replace-with-your-key'
export OPENAI_BASE_URL='https://api.openai.com/v1'
export ANSWER_MODEL='gpt-4.1-mini-2025-04-14'
export EVAL_MODEL='gpt-4o-mini'
```

The scripts read `OPENAI_API_KEY` when `--provider openai` or Step 8 is used.  A compatible
endpoint may be substituted, but the model name, tokenizer, prompt, temperature, seed support,
and response-format behavior should be recorded because they can change the result.

## 6. Select a conversation and create a fresh run directory

Find the stable conversation ID and its zero-based index instead of guessing the index:

```bash
python - <<'PY'
import json, os
path = os.environ.get("LOCOMO_JSON", "data/locomo10.json")
with open(path, encoding="utf-8") as f:
    data = json.load(f)
for index, sample in enumerate(data):
    print(index, sample["sample_id"])
PY
```

Set the selected index explicitly.  The example below evaluates all questions for one
conversation; replace `CONVERSATION_INDEX` with the index printed above.

```bash
export LOCOMO_JSON="$ROOT/data/locomo10.json"
export CONVERSATION_INDEX=0
export RUN="$ROOT/runs/locomo_reproduction"
export EMBEDDING_MODEL="$ROOT/litsearch/qwen3-embedding-8B"

mkdir -p "$ROOT/runs"
test ! -e "$RUN" || test -z "$(find "$RUN" -mindepth 1 -maxdepth 1 -print -quit)" \
  || { echo "RUN must be new or empty"; exit 1; }
```

For a fixed paper subset, create a manifest outside the code directory and pass
`--selection-file "$SELECTION_MANIFEST"` in Step 1.  A selection manifest must contain stable
`query_id` values; never recreate the subset by relying on incidental list positions after
editing the dataset.

## 7. Reproduce the standard pipeline (Steps 1--8)

Run the following commands from the `LoCoMo/` directory.  Every command is resumable except
Step 1, which deliberately requires a fresh output directory.

### Step 1: prepare sessions, questions, and isolated gold metadata

```bash
cd "$STEPS"
python 01_prepare_sessions.py \
  --locomo "$LOCOMO_JSON" \
  --out-dir "$RUN" \
  --conversation-index "$CONVERSATION_INDEX" \
  --questions-per-category 0 \
  --query-start 1 \
  --query-count 0 \
  --include-image-captions \
  --extraction-context-radius 2
```

This creates `01_sessions.json`, `01_questions.json`, and
`01_gold_DO_NOT_USE_BEFORE_STEP_7.json`.  The gold file is retained for audit/scoring only.  It
is not inserted into the Step 6 generation prompt; do not use
`--oracle-relevant-sessions-only` for a benchmark score because that option intentionally uses
gold evidence to form a diagnostic corpus subset.

If the paper run used a fixed manifest, replace the question-selection arguments with:

```bash
  --selection-file "$SELECTION_MANIFEST"
```

and keep the same manifest for all ablations.

### Step 2: partition sessions into theme-coherent episodes

```bash
python 02_partition_theme_episodes.py \
  --run-dir "$RUN" \
  --batch-size 4 \
  --provider vllm \
  --base-url 'http://127.0.0.1:8000/v1' \
  --model qwen3-32B \
  --timeout 900
```

The model must return JSON partitions covering every source dialogue ID exactly once, in order,
without gaps or overlaps.  Invalid model JSON is repaired once; an unrecoverable request falls
back to a complete-session episode so that the run remains auditable.  The output is
`02_theme_episodes.json`.

### Step 3: build the entity--property index

The standard EnSIMem configuration uses the broad/intermediate property prompt.  It preserves
answer-bearing values and relations while allowing explicit action variants to share a stable
`activity` predicate.

```bash
python 03_build_entity_index.py \
  --run-dir "$RUN" \
  --batch-size 8 \
  --property-granularity broad \
  --provider openai \
  --base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --model "$ANSWER_MODEL" \
  --timeout 900
```

The output is `03_entity_index.json`.  Each record retains the source episode and evidence DIA
IDs; the index is a navigation structure, not a replacement answer corpus.

### Step 4: plan evidence-seeking query hops

```bash
python 04_plan_query_hops.py \
  --run-dir "$RUN" \
  --questions-file 01_questions.json \
  --index-file 03_entity_index.json \
  --output-file 04_query_hop_plans.json \
  --property-granularity broad \
  --provider openai \
  --base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --model "$ANSWER_MODEL" \
  --timeout 900
```

The planner reads the observed property vocabulary from Step 3 and emits point, temporal,
compositional, or exhaustive/aggregation plans.  Use the same property granularity as Step 3;
mixing `fine` and `broad` invalidates the controlled comparison.

### Step 5: structured retrieval with dense recall fallback

The paper setting uses five structured candidates per hop, a lexical fallback of twelve episodes,
and an additive dense fallback of eight candidates.  It does not replace structured matching with
dense retrieval: structured entity/type/property matching is primary, while the dense path embeds
the question plus planner anchor against overlapping chunks of complete source episodes.
The OpenAI-compatible client is also used here for bridge resolution when a multi-hop plan needs
an intermediate value; this is why Step 5 still receives the same endpoint/model arguments.

```bash
python 05_retrieve_hops.py \
  --run-dir "$RUN" \
  --plan-file 04_query_hop_plans.json \
  --output-file 05_entity_structured_retrievals.json \
  --top-k-per-hop 5 \
  --lexical-fallback-top-k 12 \
  --dense-fallback-top-k 8 \
  --minimum-score 0.35 \
  --property-semantic-threshold 0.78 \
  --value-semantic-threshold 0.86 \
  --embedding-model "$EMBEDDING_MODEL" \
  --embedding-device cuda:0 \
  --embedding-batch-size 32 \
  --embedding-max-length 256 \
  --provider openai \
  --base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --model "$ANSWER_MODEL" \
  --timeout 900
```

If `CUDA_VISIBLE_DEVICES=3` is set, `cuda:0` above means the first visible device (physical GPU
3).  Do not pass `cuda:3` after remapping unless four physical GPUs are visible.  This command
creates embedding caches in the run directory; use `--rebuild-embeddings` only after changing the
embedding model, maximum length, or rendered corpus.

### Step 6: answer from complete retrieved episodes

```bash
python 06_answer_from_hops.py \
  --run-dir "$RUN" \
  --retrieval-file 05_entity_structured_retrievals.json \
  --output-file 06_entity_structured_predictions.json \
  --gold-file 01_gold_DO_NOT_USE_BEFORE_STEP_7.json \
  --provider openai \
  --base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --model "$ANSWER_MODEL" \
  --timeout 900
```

Answer generation uses `temperature=0` and a 256-token output budget.  The answer prompt receives
the original question, the compact plan, and complete retrieved episodes (including source
turns and enabled image metadata), not isolated index records.  Gold answers are appended only to
the output audit fields after generation.

### Step 7: deterministic diagnostics

```bash
python 07_score.py \
  --run-dir "$RUN" \
  --gold-file 01_gold_DO_NOT_USE_BEFORE_STEP_7.json \
  --retrieval-file 05_entity_structured_retrievals.json \
  --predictions-file 06_entity_structured_predictions.json \
  --output-file 07_entity_structured_scores.json
```

This reports token-level answer F1 and evidence coverage.  These diagnostics are useful for
debugging retrieval, but the paper's LoCoMo headline accuracy is the Step 8 evaluation-model
accuracy.

### Step 8: Memora-compatible evaluation-model accuracy

```bash
python 08_memora_llm_judge.py \
  --run-dir "$RUN" \
  --input-file 06_entity_structured_predictions.json \
  --output-file 08_memora_judge.json \
  --base-url "$OPENAI_BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --model "$EVAL_MODEL" \
  --timeout 900 \
  --retries 3
```

The judge prompt follows the Memora/Mem0 binary protocol: each generated answer is labeled
`CORRECT` or `WRONG` against the question and gold answer.  The request uses `temperature=0` and
`seed=42`; categories 1--4 are scored and category 5 is handled according to the Memora policy
and is not included in the reported binary accuracy.  The JSON output contains `count`,
`correct`, `llm_judge_accuracy`, and per-category summaries.  Report the exact model and endpoint
used for this step; another evaluation model can change the number even when the generated
answers are identical.

To evaluate the same fixed predictions with a second evaluation model, keep the input file
unchanged and write to a different output file.  For a local OpenAI-compatible Qwen endpoint, for
example:

```bash
python 08_memora_llm_judge.py \
  --run-dir "$RUN" \
  --input-file 06_entity_structured_predictions.json \
  --output-file 08_memora_judge_qwen3.json \
  --base-url 'http://127.0.0.1:8000/v1' \
  --api-key EMPTY \
  --model qwen3-32B \
  --timeout 900 \
  --retries 3
```

## 8. Run all conversations

`00_run_remaining_conversations.py` is an orchestration helper.  It discovers conversation IDs
from `locomo10.json`, creates one independent run directory per conversation, and runs Steps 1--8
with the paper defaults.  Its historical default exclusion list is
`conv-26,conv-30,conv-41,conv-42,conv-49`; do not rely on that default when reproducing a table.
The following command clears that exclusion list and runs every conversation:

```bash
cd "$STEPS"
CUDA_VISIBLE_DEVICES=3 \
python 00_run_remaining_conversations.py \
  --steps "$STEPS" \
  --locomo "$LOCOMO_JSON" \
  --judge-script "$STEPS/08_memora_llm_judge.py" \
  --exclude-conversation-ids '' \
  --output-root "$ROOT/runs/locomo_all" \
  --qwen-base-url 'http://127.0.0.1:8000/v1' \
  --qwen-model qwen3-32B \
  --openai-base-url "$OPENAI_BASE_URL" \
  --openai-model "$ANSWER_MODEL" \
  --judge-model "$EVAL_MODEL" \
  --embedding-model "$EMBEDDING_MODEL" \
  --embedding-device cuda:0 \
  --cuda-visible-devices 3 \
  --timeout 900 \
  --step2-batch-size 4 \
  --step3-batch-size 8
```

The helper writes `00_remaining_conversations_summary.json` and does not mix checkpoints between
conversations.  If one conversation fails, rerun only that ID with a new or completed run
directory; do not merge files produced with different question selections or model settings.
To run only a named conversation, replace `--exclude-conversation-ids ''` with, for example,
`--conversation-ids 'conv-26'`.  The helper intentionally evaluates all questions in each
selected conversation; use the explicit Steps 1--8 commands above when a paper table uses a
published fixed-selection manifest.

For a multi-conversation headline score, aggregate the eligible judge rows rather than averaging
per-conversation percentages (this keeps the denominator correct):

```bash
python - <<'PY'
import glob, json, os
root = os.environ.get("RUN_ROOT", "runs/locomo_all")
files = sorted(glob.glob(os.path.join(root, "*", "08_memora_judge_*.json")))
correct = count = 0
for path in files:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    summary = payload.get("summary", payload)
    correct += int(summary["correct"])
    count += int(summary["count"])
print({"files": len(files), "count": count, "correct": correct,
       "llm_judge_accuracy": round(correct / count, 4) if count else None})
PY
```

## 9. Ablations

The following scripts reuse the same questions, episodes, plans, answer model, and evaluation
protocol.  Each output directory must be new or empty.

### Episode granularity

The ablation remaps the same entity records to deterministic per-turn or per-session episodes;
it does not call an extraction model.

```bash
python 09_build_episode_granularity_ablation.py \
  --source-run-dir "$RUN" \
  --output-run-dir "$ROOT/runs/ablation_per_turn" \
  --granularity per-turn \
  --plan-file 04_query_hop_plans.json
```

Run `05_retrieve_hops.py`, `06_answer_from_hops.py`, and `08_memora_llm_judge.py` on the new
directory, changing only the run/input/output paths.  Repeat with `--granularity per-session`.

### Property granularity

```bash
python 10_build_property_granularity_ablation.py \
  --source-run-dir "$RUN" \
  --output-run-dir "$ROOT/runs/ablation_property_fine" \
  --granularity fine \
  --plan-file 04_query_hop_plans.json
```

Repeat with `--granularity broad`.  This transformation keeps the source episodes fixed and
changes only property labels and the corresponding planner fields.  Use the same retrieval and
answer commands afterward; do not add `--property-granularity` to `03_build_entity_index.py` for
this no-LLM controlled ablation unless intentionally rebuilding a different experiment.

Two additional stress tests are available in the same script:

```bash
# Slightly broader than the paper's broad setting, but still an intermediate vocabulary.
python 10_build_property_granularity_ablation.py \
  --source-run-dir "$RUN" \
  --output-run-dir "$ROOT/runs/ablation_property_broad_plus" \
  --granularity broad_plus \
  --plan-file 04_query_hop_plans.json

# Deliberately over-broad stress test: every non-empty property becomes `fact`.
python 10_build_property_granularity_ablation.py \
  --source-run-dir "$RUN" \
  --output-run-dir "$ROOT/runs/ablation_property_very_broad" \
  --granularity very_broad \
  --plan-file 04_query_hop_plans.json
```

`broad_plus` preserves the current broad relation families and additionally groups a few
action-like variants (such as buying, learning, studying, teaching, and celebrating) as
`activity`.  `very_broad` intentionally removes all predicate distinctions while leaving the
entity, value, evidence, and provenance fields untouched.  Run the same Step 5, Step 6, and
Step 8 commands on both output directories.  These modes are controlled post-processing
ablations: they do not rebuild episodes or call an extraction model, so any accuracy difference
isolates the property-label granularity.

### Structured matching versus dense-only retrieval

```bash
python 11_retrieve_dense_only.py \
  --source-run-dir "$RUN" \
  --output-run-dir "$ROOT/runs/ablation_dense_only" \
  --plan-file 04_query_hop_plans.json \
  --output-file 05_dense_only_retrievals.json \
  --top-k-per-hop 5 \
  --dense-all-matching-top-k 50 \
  --turns-per-chunk 4 \
  --embedding-model "$EMBEDDING_MODEL" \
  --embedding-device cuda:0 \
  --embedding-batch-size 32 \
  --embedding-max-length 256
```

Then run Step 6 with `--retrieval-file 05_dense_only_retrievals.json`, followed by Step 8.  This
script intentionally does not read `03_entity_index.json`; it embeds overlapping chunks of the
complete episode text and ranks episodes by the question plus natural-language hop purpose.

### Fixed top-k retrieval ablation

To compare fixed budgets with the adaptive policy, rerun Step 5 for each
`k in {1,3,5,8,15,20}` using a distinct output file and add
`--fixed-top-k-ablation --top-k-per-hop "$k"`.  Keep the plans, answer model, evaluation model,
embedding checkpoint, and question IDs fixed.  The adaptive paper configuration omits
`--fixed-top-k-ablation` and uses the query-type-aware stopping rules.

## 10. Checkpoints, reproducibility, and common failure modes

* **Fresh Step 1 directory.** `01_prepare_sessions.py` refuses a directory containing existing
  `0*.json` files.  Use a new directory for a new question selection.
* **Do not mix checkpoints.** Step 2, Step 3, Step 4, Step 6, and Step 8 record model/schema
  metadata and reject incompatible files.  A model change requires a new run directory; Step 3
  supports `--force-rebuild` and Step 4 supports `--force-replan` only when intentionally
  regenerating that stage.
* **Stable question set.** Question IDs and order determine the output rows.  Use one manifest or
  one immutable `01_questions.json` for the baseline and all ablations.
* **Image evidence.** Keep `--include-image-captions` enabled for the paper setting.  Image URLs
  are provenance fields only; the code never performs an outside image lookup.
* **GPU remapping.** With `CUDA_VISIBLE_DEVICES=3`, use `--embedding-device cuda:0`.
* **Embedding cache.** If the local embedding path, max length, or rendered episode text changes,
  delete the relevant cache or pass `--rebuild-embeddings`; stale vectors must not be reused.
* **API failures.** The clients retry transient endpoint errors, and successful stages are
  checkpointed.  A failed request can therefore be resumed, but rerun with the same model,
  endpoint, and run directory to preserve comparability.
* **Gold isolation.** Never use the gold file to select sessions, episodes, or retrieval
  candidates for a reported score.  The only allowed use is post-generation audit and Step 8
  evaluation.
* **Judge output.** Use `llm_judge_accuracy = correct / count` from the Step 8 JSON, not Step 7
  token-F1, when reproducing the paper's headline LoCoMo accuracy.

## 11. Output checklist

Before reporting a result, verify that the run directory contains at least:

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

Keep the JSON artifacts and the exact command-line configuration together.  The most useful files
for review are `01_questions.json` (the evaluated IDs), `03_entity_index.json` (grounded records),
`04_query_hop_plans.json` (adaptive budgets), `05_*` (retrieval audit), and
`08_memora_judge.json` (the headline accuracy).
