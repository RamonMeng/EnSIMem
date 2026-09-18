# EnSI-Memory: LoCoMo Method Ported to LongMemEval

This package validates the **successful LoCoMo EnSI-Memory method** on LongMemEval while keeping its one-shot retrieval workflow intact.

## Method boundary

The benchmark pipeline is:

```text
LongMemEval item
  -> adapter only
  -> Step 2: LoCoMo theme partitioning               [UNCHANGED METHOD]
  -> Step 3: LoCoMo entity-structured indexing      [UNCHANGED METHOD]
  -> Step 4: LoCoMo one-shot query/hop planning     [UNCHANGED METHOD]
  -> Step 5: LoCoMo one-shot multi-hop retrieval    [core method + additive recall guards]
  -> Step 6: LongMemEval-specific evidence reasoning
  -> Step 7: Memora-style LongMemEval judge
```

There is **no iterative retrieve -> sufficiency check -> retrieve again loop**. Step 5 closes retrieval. The current port keeps the LoCoMo structured retrieval/ranking path and adds bounded, additive recall guards for sparse LongMemEval representations (lexical text, dense episode chunks, and explicit temporal date windows). Step 6 can audit, aggregate, reconcile updates, reason over dates, and verify an answer, but it cannot retrieve any new memory.

The following files provide the supplied successful LoCoMo `entity_structured_steps.zip` method and its LongMemEval-safe additive recall guards:

- `_common.py`
- `llm.py`
- `online_efficiency.py`
- `prompts.py`
- `02_partition_theme_episodes.py`
- `03_build_entity_index.py`
- `04_plan_query_hops.py`
- `05_retrieve_hops.py` (LoCoMo retrieval path plus bounded recall guards)

The baseline SHA-256 hashes are recorded in `LOCOMO_SOURCE_SHA256.txt`; the current
`05_retrieve_hops.py` hash intentionally differs because it includes the additive LongMemEval
recall guards described above.

LongMemEval-specific files are deliberately separated:

- `longmemeval_adapter.py`: converts LongMemEval `role/content` haystacks into EnSI session/turn format; gold answers are not passed to Steps 2-6.
- `01_prepare_longmemeval.py`: writes the exact file shapes consumed by LoCoMo Steps 2-5.
- `longmemeval_generation.py`: evidence-first LongMemEval reasoning logic. Temporal questions use
  a bounded-batch, quote-grounded date ledger before final answering; preference and assistant
  paths retain their existing logic.
- `06_answer_longmemeval.py`: runs that reasoning only on the closed Step-5 evidence set.
- `memora_evaluation.py` + `07_evaluate_longmemeval.py`: LongMemEval correctness judge and post-generation gold diagnostics.
- `run_longmemeval.py`: orchestration only; it does not replace the LoCoMo retrieval algorithms.

## Important default: fresh preprocessing

`run_longmemeval.py` defaults to:

```text
--fresh-preprocess
```

For every selected query it deletes that query's existing artifact directory and reruns Steps 1-6 from scratch. This means Step 2 partitioning and Step 3 indexing are **not reused** from the earlier ~90% LongMemEval method.

To resume an interrupted run without deleting completed per-query preprocessing, explicitly use:

```bash
--no-fresh-preprocess
```

## Model layout

The defaults follow the current experimental setup requested for this port:

- Step 2 partition: GPT API (`gpt-4.1-mini-2025-04-14`)
- Step 3 entity index: GPT API (`gpt-4.1-mini-2025-04-14`)
- Step 4 query planner: local vLLM (`qwen3-32B`, `http://127.0.0.1:8000/v1`)
- Step 5 bridge calls: GPT API; retrieval embeddings remain local
- Step 6 LongMemEval answer reasoning: GPT API (`gpt-4.1-mini-2025-04-14`)
- Step 7 judge: `gpt-4o-mini-2024-07-18`

Every endpoint/model can be overridden from the command line.

### Temporal generation behavior

For `temporal-reasoning` questions, Step 6 still scans every Step-5 retrieved episode, but it
does so in bounded extraction batches. Each batch produces User-only facts with exact quotes and
absolute event dates. Step 5 additionally appends episodes observed on an explicitly requested
relative-date window such as `last Friday`, `a week ago`, or `past weekend`. A final GPT pass answers
from the merged ledger; explicit two-endpoint interval questions use deterministic calendar arithmetic.
If the final answer omits grounded citations, a citation-repair pass runs before the closed-book
evidence gate. `--max-reasoning-episodes` does not cap temporal evidence; it remains applicable
to the other reranked answer paths.

## 1. Test the exact previous queries

Create a text file containing one prior LongMemEval question ID per line, for example:

```text
<question_id_1>
<question_id_2>
<question_id_3>
```

Then run:

```bash
cd ensi_memory_locomo_to_longmemeval_v1
export OPENAI_API_KEY="YOUR_KEY"

python run_longmemeval.py \
  --dataset /shared/data3/xuanyum2/LongMemEval/data/longmemeval_s_cleaned.json \
  --output-dir /shared/data3/xuanyum2/LongMemEval/runs/locomo_method_lme_test \
  --question-id-file previous_query_ids.txt \
  --fresh-preprocess \
  --planner-base-url http://127.0.0.1:8000/v1 \
  --planner-model qwen3-32B \
  --embedding-model ../litsearch/qwen3-embedding-8B \
  --embedding-device cuda:2
```

You can also specify IDs directly:

```bash
python run_longmemeval.py \
  --dataset /shared/data3/xuanyum2/LongMemEval/data/longmemeval_s_cleaned.json \
  --output-dir /shared/data3/xuanyum2/LongMemEval/runs/locomo_method_lme_test \
  --question-id ID_1 \
  --question-id ID_2 \
  --question-id ID_3 \
  --fresh-preprocess \
  --embedding-model ../litsearch/qwen3-embedding-8B \
  --embedding-device cuda:2
```

`--question-id-file` accepts plain text, a JSON list, or a JSON manifest containing `question_ids` / `selected_question_ids`.

## 2. Small smoke experiment by dataset position

```bash
python run_longmemeval.py \
  --dataset /shared/data3/xuanyum2/LongMemEval/data/longmemeval_s_cleaned.json \
  --output-dir /shared/data3/xuanyum2/LongMemEval/runs/locomo_method_lme_smoke \
  --start 0 \
  --limit 5 \
  --fresh-preprocess \
  --embedding-model ../litsearch/qwen3-embedding-8B \
  --embedding-device cuda:2
```

## 3. Balanced category experiment

For seven queries from each of the six base LongMemEval categories:

```bash
python run_longmemeval.py \
  --dataset /shared/data3/xuanyum2/LongMemEval/data/longmemeval_s_cleaned.json \
  --output-dir /shared/data3/xuanyum2/LongMemEval/runs/locomo_method_lme_42 \
  --per-category 7 \
  --fresh-preprocess \
  --embedding-model ../litsearch/qwen3-embedding-8B \
  --embedding-device cuda:2
```

This deterministic convenience mode takes the first N eligible examples per category. For a strict A/B against an old experiment, use an explicit question-ID file instead.

## 4. Stop after a stage for debugging

Useful values:

```text
--stop-after prepare
--stop-after partition
--stop-after index
--stop-after plan
--stop-after retrieve
--stop-after answer
--stop-after judge
```

Example: inspect retrieval before spending answer/judge calls:

```bash
python run_longmemeval.py ... --stop-after retrieve
```

Then inspect each query directory:

```text
01_sessions.json
01_questions.json
01_longmemeval_metadata.json
02_theme_episodes.json
03_entity_index.json
04_query_hop_plans_v2.json
05_hop_retrievals.json
```

## 5. Run only the judge later

```bash
python 07_evaluate_longmemeval.py \
  --dataset /shared/data3/xuanyum2/LongMemEval/data/longmemeval_s_cleaned.json \
  --run-dir /shared/data3/xuanyum2/LongMemEval/runs/locomo_method_lme_test \
  --force
```

The aggregate result is written to:

```text
07_results_and_context_summary.json
```

## What to inspect first

For the first few previous queries, compare these three things before scaling up:

1. `04_query_hop_plans_v2.json`: Did the one-shot planner choose the right hop structure and `retrieval_scope`?
2. `05_hop_retrievals.json`: Did the same LoCoMo retrieval method actually include the needed LongMemEval evidence episodes?
3. `06_prediction.json`: If retrieval was good but the answer was wrong, inspect `generation_trace` to determine whether the failure is temporal anchoring, state reconciliation, aggregation semantics, preference reasoning, or final verification.

That separation is intentional: LongMemEval should validate the **same retrieval method**, while permitting benchmark-specific answer reasoning after retrieval is complete.
