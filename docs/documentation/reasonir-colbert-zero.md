# ReasonIR ColBERT-Zero Fine-Tuning

This note captures the available workflows for fine-tuning `lightonai/ColBERT-Zero` on ReasonIR data, including both the local synthetic triplets path and the official HQ dataset path.

For the end-to-end official HQ tuning workflow, including the exact run order for sweeps and BRIGHT evals, see:

- [`REASONIR_HQ_SWEEP_RUNBOOK.md`](/home/rbw/repo/pylate/REASONIR_HQ_SWEEP_RUNBOOK.md)

## Goal

Replicate the general structure of [`examples/train/reason_moderncolbert.py`](/home/rbw/repo/pylate/examples/train/reason_moderncolbert.py) for `ColBERT-Zero`, while preserving the prompt behavior required by the base checkpoint. The original local workflow here uses synthetic data generated in the ReasonIR repository, and the official HQ replication path is documented separately below.

The critical constraint is prompt alignment:

- queries must keep the `search_query: ` prefix
- documents and negatives must keep the `search_document: ` prefix

This is required by the base checkpoint and is preserved in [`examples/train/ColBERT-zero/reasonir.py`](/home/rbw/repo/pylate/examples/train/ColBERT-zero/reasonir.py).

The training scripts in this repo also default to Weights & Biases logging with:

- project: `ColBERT-Zero`
- entity: `rbw`

Those defaults are baked into:

- [`examples/train/ColBERT-zero/reasonir.py`](/home/rbw/repo/pylate/examples/train/ColBERT-zero/reasonir.py)
- [`examples/train/ColBERT-zero/reason_moderncolbert.py`](/home/rbw/repo/pylate/examples/train/ColBERT-zero/reason_moderncolbert.py)

You can override them with `--wandb-project`, `--wandb-entity`, or disable trainer integrations with `--report-to none`.

The BRIGHT evaluator also supports opt-in W&B logging with the same defaults:

- project: `ColBERT-Zero`
- entity: `rbw`

Use `--report-to wandb` on [`bright_reasonir.py`](/home/rbw/repo/pylate/examples/evaluation/bright_reasonir.py) to log per-task metrics plus the final summary into W&B. Disable it with `--report-to none`.

For quick project inspection from the terminal, use:

- [`scripts/wandb_project_runs.py`](/home/rbw/repo/pylate/scripts/wandb_project_runs.py)

Example:

```bash
uv run python scripts/wandb_project_runs.py \
  --entity rbw \
  --project ColBERT-Zero \
  --limit 10 \
  --history-tail 5
```

## Local Inputs

- Base model used in this run: `/mnt/ml_models/lightonai/ColBERT-Zero`
- ReasonIR HQ data used in this run: `/home/rbw/repo/ReasonIR/synthetic_data_generation/synthetic_data/hq/hq_gen/gemini-3-flash-preview/final_train_data.jsonl`
- ReasonIR VL data used in this run: `/home/rbw/repo/ReasonIR/synthetic_data_generation/synthetic_data/vl/hq_gen/gemini-3-flash-preview/final_train_data.jsonl`

Those are local machine paths, not hard requirements. On another machine, override them with:

- `--model /path/to/ColBERT-Zero`
- `--data-root /path/to/ReasonIR/synthetic_data_generation/synthetic_data`
- `--output-dir /path/to/output`
- `--dataset-cache-dir /path/to/writable/hf-cache`

Current dataset sizes:

- HQ: `14979` triplets
- VL: `12174` triplets
- Total: `27153` triplets

## Training Script

The dedicated training entry point is:

- [`examples/train/ColBERT-zero/reasonir.py`](/home/rbw/repo/pylate/examples/train/ColBERT-zero/reasonir.py)

Key behavior:

- loads local HQ/VL ReasonIR JSONL triplets
- filters to rows with at least one positive and the requested number of negatives
- maps rows into `query`, `document`, `negative_0..negative_n`
- preserves `search_query:` and `search_document:` during training
- uses a prompt-aligned triplet evaluator when validation is enabled
- supports a writable HF datasets cache via `--dataset-cache-dir`

## Official HQ Replication Path

If you want to stay closer to [`examples/train/reason_moderncolbert.py`](/home/rbw/repo/pylate/examples/train/reason_moderncolbert.py), there is now a dedicated ColBERT-Zero entry point for the official ReasonIR HQ dataset:

- [`examples/train/ColBERT-zero/reason_moderncolbert.py`](/home/rbw/repo/pylate/examples/train/ColBERT-zero/reason_moderncolbert.py)

This script follows the dataset-card guidance for HQ data:

- loads `reasonir/reasonir-data` with config `hq`
- loads `xlangai/BRIGHT` with config `documents`
- reconstructs positive document text by resolving the positive document identifier against the BRIGHT document store
- keeps the original HQ negative text as provided by ReasonIR
- preserves `search_query: ` and `search_document: ` during training and validation

Example command:

```bash
uv run python examples/train/ColBERT-zero/reason_moderncolbert.py \
  --model /mnt/ml_models/lightonai/ColBERT-Zero \
  --reasonir-split train \
  --validation-size 0 \
  --epochs 3 \
  --lr 1e-5 \
  --bs 16 \
  --eval-bs 16 \
  --mini-batch-size 4 \
  --fp16 \
  --num-workers 4 \
  --save-steps 1000 \
  --logging-steps 10 \
  --save-total-limit 2 \
  --run-name reasonir-hq-colbert-zero \
  --dataset-cache-dir /tmp/pylate-hf-cache \
  --output-dir /home/rbw/repo/pylate/output/reasonir-hq-colbert-zero
```

Use this path when the goal is parity with the public ReasonIR HQ training recipe rather than the local synthetic HQ/VL workflow described below.

### Pilot Matrix

Use a three-stage pilot sweep so the results are easier to interpret.

Stage 1 is a fairer batch-size comparison with roughly equal examples seen and learning rate scaled with batch size:

- [`scripts/reasonir_hq_batch_sweep.sh`](/home/rbw/repo/pylate/scripts/reasonir_hq_batch_sweep.sh)

This script keeps shared settings fixed:

- `temp=1.0`
- `validation_size=0.01`
- `warmup_ratio=0.1`
- `mini_batch_size=32`
- `save_steps=25`
- `eval_steps=25`
- `save_total_limit=20`

It runs:

- `hq-batch-bs256-lr1e5-temp1` with `max_steps=400`
- `hq-batch-bs512-lr2e5-temp1` with `max_steps=200`
- `hq-batch-bs1024-lr4e5-temp1` with `max_steps=100`
- `hq-batch-bs2048-lr8e5-temp1` with `max_steps=50`

Stage 2 is a learning-rate sweep at the winning batch size:

- [`scripts/reasonir_hq_lr_sweep.sh`](/home/rbw/repo/pylate/scripts/reasonir_hq_lr_sweep.sh)

Set `BEST_BATCH_SIZE` before running it. By default it uses `1024`.

It runs:

- `hq-lr-bs<BEST_BATCH_SIZE>-lr1e6-temp1`
- `hq-lr-bs<BEST_BATCH_SIZE>-lr5e6-temp1`
- `hq-lr-bs<BEST_BATCH_SIZE>-lr1e5-temp1`
- `hq-lr-bs<BEST_BATCH_SIZE>-lr5e5-temp1`

Stage 3 is a temperature sweep at the winning batch size and learning rate:

- [`scripts/reasonir_hq_temp_sweep.sh`](/home/rbw/repo/pylate/scripts/reasonir_hq_temp_sweep.sh)

Set both `BEST_BATCH_SIZE` and `BEST_LR` before running it.

It runs:

- `temp=0.02`
- `temp=0.05`
- `temp=0.1`

For a BRIGHT subset comparison over produced sweep outputs, use:

- [`scripts/reasonir_hq_bright_subset_eval.sh`](/home/rbw/repo/pylate/scripts/reasonir_hq_bright_subset_eval.sh)

By default this evaluates the subset:

- `biology`
- `economics`
- `robotics`
- `pony`

It also defaults to logging each eval run to W&B as:

- project: `ColBERT-Zero`
- entity: `rbw`
- group: `reasonir-hq-bright-subset`

Override those with `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_GROUP`, or disable eval logging with `REPORT_TO=none`.

Use `STAGE=batch`, `STAGE=lr`, `STAGE=temp`, or `STAGE=all` to choose which sweep stage to evaluate when you are not passing explicit run names.

This eval wrapper also defaults to cleaning up the model-specific BRIGHT document embedding cache after each run, since sweep comparisons do not reuse document shards across different model paths. Disable that with `CLEANUP_DOCUMENT_CACHE=0` if you explicitly want cache reuse for reruns or resumes.

The eval wrapper no longer forces Hugging Face offline mode. It will populate the BRIGHT cache on demand. If you want cached-only reruns, prefix the command with:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CLEANUP_DOCUMENT_CACHE=0
```

To evaluate retained `checkpoint-*` directories with the same script, set `INCLUDE_CHECKPOINTS=1`. If you already ran the final-model pass and do not want duplicate `final` evals, also set `INCLUDE_FINAL=0`. Checkpoint-mode evals default to the W&B group `reasonir-hq-bright-checkpoints`.

Example:

```bash
STAGE=temp BEST_BATCH_SIZE=1024 BEST_LR=1e-5 INCLUDE_CHECKPOINTS=1 INCLUDE_FINAL=0 ./scripts/reasonir_hq_bright_subset_eval.sh
```

## Commands Used

### GPU smoke test

This verifies end-to-end GPU training with just `2` examples from each split:

```bash
uv run python examples/train/ColBERT-zero/reasonir.py \
  --datasets hq,vl \
  --max-examples-per-split 2 \
  --validation-size 0 \
  --epochs 1 \
  --bs 2 \
  --eval-bs 2 \
  --mini-batch-size 1 \
  --fp16 \
  --num-workers 0 \
  --save-steps 9999 \
  --logging-steps 1 \
  --save-total-limit 1 \
  --run-name reasonir-smoke-gpu \
  --dataset-cache-dir /tmp/pylate-hf-cache \
  --output-dir /home/rbw/repo/pylate/output/reasonir-smoke-gpu
```

### GPU pilot run

This was used to confirm stable settings on a larger subset:

```bash
uv run python examples/train/ColBERT-zero/reasonir.py \
  --datasets hq,vl \
  --max-examples-per-split 64 \
  --validation-size 0 \
  --epochs 1 \
  --bs 16 \
  --eval-bs 16 \
  --mini-batch-size 4 \
  --fp16 \
  --num-workers 0 \
  --save-steps 9999 \
  --logging-steps 1 \
  --save-total-limit 1 \
  --run-name reasonir-pilot-bs16 \
  --dataset-cache-dir /tmp/pylate-hf-cache \
  --output-dir /home/rbw/repo/pylate/output/reasonir-pilot-bs16
```

### Full training run

This is the full HQ+VL run currently used for the main checkpoint:

```bash
uv run python -u examples/train/ColBERT-zero/reasonir.py \
  --datasets hq,vl \
  --validation-size 0 \
  --epochs 3 \
  --lr 8e-6 \
  --bs 16 \
  --eval-bs 16 \
  --mini-batch-size 4 \
  --fp16 \
  --num-workers 4 \
  --save-steps 1000 \
  --logging-steps 10 \
  --save-total-limit 2 \
  --run-name reasonir-full-gpu \
  --dataset-cache-dir /tmp/pylate-hf-cache \
  --output-dir /home/rbw/repo/pylate/output/reasonir-full-gpu
```

## Selected Training Parameters

The final full run used these settings:

- datasets: `hq,vl`
- train rows: `27153`
- validation split: `0`
- epochs: `3`
- learning rate: `8e-6`
- per-device train batch size: `16`
- per-device eval batch size: `16`
- cached contrastive mini-batch size: `4`
- precision: `fp16`
- dataloader workers: `4`
- checkpoint save interval: `1000` steps
- logging interval: `10` steps
- checkpoint retention: `2`
- dataset cache dir: `/tmp/pylate-hf-cache`
- output dir: `/home/rbw/repo/pylate/output/reasonir-full-gpu`

Observed training summary:

- runtime: `5437.17s`
- steps: `5091`
- train loss: `0.3463`
- train steps per second: `0.936`

## Operational Notes

- The Codex sandbox did not expose `/dev/nvidia*`, so GPU runs had to be launched outside the sandbox.
- The default HF datasets cache root was not writable in this environment, so the script was extended with `--dataset-cache-dir`.
- The smoke test originally failed because `sentence-transformers` reserves the `dataset_name` column name; the training script was updated to stop injecting it into the plain concatenated dataset.
- The JSON loading step is cached under `/tmp/pylate-hf-cache`, but the script does not currently write a fully materialized preprocessed dataset via `save_to_disk()`.

## Evaluation Review

There are already evaluation entry points in this repository:

- [`examples/evaluation/beir_dataset.py`](/home/rbw/repo/pylate/examples/evaluation/beir_dataset.py) for BEIR
- [`examples/evaluation/custom_dataset.py`](/home/rbw/repo/pylate/examples/evaluation/custom_dataset.py) for local corpus/query/qrels datasets
- [`examples/evaluation/longembed_dataset.py`](/home/rbw/repo/pylate/examples/evaluation/longembed_dataset.py) for LongEmbed tasks
- [`pylate/evaluation/nano_beir_evaluator.py`](/home/rbw/repo/pylate/pylate/evaluation/nano_beir_evaluator.py) for quick NanoBEIR checks

Prompt-sensitive models such as `ColBERT-Zero` require prompt-aligned evaluation. The example evaluation scripts were updated to automatically use `prompt_name="query"` and `prompt_name="document"` whenever the checkpoint exposes those prompts in `config_sentence_transformers.json`.

### Recommended eval commands

Quick BEIR check on the fine-tuned checkpoint:

```bash
uv run python examples/evaluation/beir_dataset.py \
  --model_name_or_path /home/rbw/repo/pylate/output/reasonir-full-gpu/final \
  --dataset_name scifact \
  --document_batch_size 128 \
  --query_batch_size 64
```

Custom dataset evaluation:

```bash
uv run python examples/evaluation/custom_dataset.py \
  --model_name_or_path /home/rbw/repo/pylate/output/reasonir-full-gpu/final \
  --dataset_path path/to/custom_dataset \
  --split dev
```

For very fast retrieval sanity checks during training, prefer `NanoBEIREvaluator` from PyLate. For BRIGHT or ReasonIR's original evaluation harness, use the separate ReasonIR repository under `/home/rbw/repo/ReasonIR/evaluation`.

### BRIGHT evaluation

To reproduce the evaluation style used by `Reason-ModernColBERT`, this repository now includes a PyLate-native BRIGHT runner:

- [`examples/evaluation/bright_reasonir.py`](/home/rbw/repo/pylate/examples/evaluation/bright_reasonir.py)

Key behavior:

- loads the official `xlangai/BRIGHT` tasks directly
- supports both raw BRIGHT queries and `gpt4_reason`
- applies BRIGHT `excluded_ids` before scoring
- uses exact PyLate MaxSim instead of approximate retrieval
- caches document embeddings per model and task under `--cache_dir`
- reuses cached document embeddings across different `--corpus_chunk_size` values via `--document_cache_chunk_size`
- resumes from `--output_json` and skips already completed tasks

Example raw-query BRIGHT run:

```bash
uv run python examples/evaluation/bright_reasonir.py \
  --model_name_or_path /home/rbw/repo/pylate/output/reasonir-full-gpu/final \
  --reasoning none \
  --query_batch_size 16 \
  --query_encode_batch_size 32 \
  --document_batch_size 128 \
  --corpus_chunk_size 1024 \
  --top_k 1000 \
  --cache_dir /tmp/pylate-bright-cache \
  --output_json /home/rbw/repo/pylate/output/bright-reasonir-full-gpu-raw.json
```

Example GPT-4 reasoning-trace BRIGHT run, using the per-task query lengths listed in the `Reason-ModernColBERT` model card:

```bash
uv run python examples/evaluation/bright_reasonir.py \
  --model_name_or_path /home/rbw/repo/pylate/output/reasonir-full-gpu/final \
  --reasoning gpt4 \
  --use_reason_moderncolbert_gpt4_lengths \
  --query_batch_size 16 \
  --query_encode_batch_size 32 \
  --document_batch_size 128 \
  --corpus_chunk_size 1024 \
  --top_k 1000 \
  --cache_dir /tmp/pylate-bright-cache \
  --output_json /home/rbw/repo/pylate/output/bright-reasonir-full-gpu-gpt4.json
```

Operational note: exhaustive BRIGHT MaxSim scoring is expensive. On the 3090, the largest tasks such as `earth_science` and `stackoverflow` were slow enough that moving the sweep to a 5090 was judged worthwhile.

### NanoBEIR sweep

To replicate the style of the released `ColBERT-Zero` model-card evaluation, use:

```bash
uv run python examples/evaluation/nanobeir_sweep.py \
  --model_name_or_path /mnt/ml_models/lightonai/ColBERT-Zero \
  --batch_size 64 \
  --corpus_chunk_size 2000 \
  --output_json /home/rbw/repo/pylate/output/nanobeir-base-colbert-zero.json
```

Then run the same sweep on the fine-tuned checkpoint:

```bash
uv run python examples/evaluation/nanobeir_sweep.py \
  --model_name_or_path /home/rbw/repo/pylate/output/reasonir-full-gpu/final \
  --batch_size 64 \
  --corpus_chunk_size 2000 \
  --output_json /home/rbw/repo/pylate/output/nanobeir-reasonir-full-gpu.json
```

Current observed NanoBEIR mean comparison:

- base `ColBERT-Zero` `ndcg@10`: `0.6824`
- fine-tuned ReasonIR checkpoint `ndcg@10`: `0.5818`
- delta: `-0.1007`

- base `ColBERT-Zero` `mrr@10`: `0.7518`
- fine-tuned ReasonIR checkpoint `mrr@10`: `0.6469`
- delta: `-0.1049`

- base `ColBERT-Zero` `map@100`: `0.6017`
- fine-tuned ReasonIR checkpoint `map@100`: `0.5020`
- delta: `-0.0996`

The largest `ndcg@10` drops in the current run were on `NanoFiQA2018`, `NanoClimateFEVER`, and `NanoTouche2020`. No NanoBEIR dataset improved over the base checkpoint in this comparison.

## BRIGHT Partial Results

The BRIGHT runner was validated on both checkpoints and partially run on the raw-query setting before being handed off to a stronger GPU.

Observed raw-query BRIGHT `NDCG@10` values:

- base `ColBERT-Zero` `biology`: `16.20`
- base `ColBERT-Zero` `earth_science`: `21.73`
- base `ColBERT-Zero` `economics`: `15.38`
- base `ColBERT-Zero` `psychology`: `18.01`
- base `ColBERT-Zero` `robotics`: `15.31`
- base `ColBERT-Zero` `pony`: `15.20`
- fine-tuned checkpoint `pony`: `4.74`

That is only a partial BRIGHT comparison, but the early signal was already unfavorable for the current fine-tune.
