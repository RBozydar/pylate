# ReasonIR ColBERT-Zero Fine-Tuning

This note captures the available workflows for fine-tuning `lightonai/ColBERT-Zero` on ReasonIR data, including the local synthetic triplets path, the new balanced mixed-dataset path, and the official HQ dataset path.

For the future structured W&B sweep workflow for the official HQ path, see:

- [`REASONIR_HQ_SWEEP_RUNBOOK.md`](/home/rbw/repo/pylate/REASONIR_HQ_SWEEP_RUNBOOK.md)

For the staged balanced mixed-dataset workflow, see:

- [`REASONIR_MIXED_RUNBOOK.md`](/home/rbw/repo/pylate/REASONIR_MIXED_RUNBOOK.md)

For completed mixed-sweep comparisons, see:

- [`reasonir-mixed-lr-sweep-results.md`](/home/rbw/repo/pylate/docs/documentation/reasonir-mixed-lr-sweep-results.md)
- [`reasonir-mixed-temp-sweep-results.md`](/home/rbw/repo/pylate/docs/documentation/reasonir-mixed-temp-sweep-results.md)

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
- preferred synthetic root on this machine: `/mnt/ml_models/datasets/ReasonIR/synthetic_data`
- regenerated HQ source: `/mnt/ml_models/datasets/ReasonIR/synthetic_data/hq/hq_gen/gemini-3-flash-preview/final_train_data.jsonl`
- regenerated VL source: `/mnt/ml_models/datasets/ReasonIR/synthetic_data/vl/hq_gen/gemini-3-flash-preview/final_train_data.jsonl`
- balanced mixed dataset root: `/mnt/ml_models/datasets/ReasonIR/synthetic_data/mixed/hq_gen/balanced-v1`

Those are local machine paths, not hard requirements. On another machine, override them with:

- `--model /path/to/ColBERT-Zero`
- `--data-root /path/to/ReasonIR/synthetic_data_generation/synthetic_data`
- `--output-dir /path/to/output`
- `--dataset-cache-dir /path/to/writable/hf-cache`

Current regenerated HQ/VL sizes:

- HQ: `14979` triplets
- VL: `12174` triplets
- Total: `27153` triplets

## Balanced Mixed Dataset

The mixed dataset builder is:

- [`scripts/prepare_reasonir_mixed_dataset.py`](/home/rbw/repo/pylate/scripts/prepare_reasonir_mixed_dataset.py)

It stages a balanced local training mix directly under `/mnt/ml_models/datasets/ReasonIR/synthetic_data` so the existing local training script can consume it without any special loader changes.

Default output:

- `/mnt/ml_models/datasets/ReasonIR/synthetic_data/mixed/hq_gen/balanced-v1/final_train_data.jsonl`
- `/mnt/ml_models/datasets/ReasonIR/synthetic_data/mixed/hq_gen/balanced-v1/manifest.json`
- `/mnt/ml_models/datasets/ReasonIR/synthetic_data/mixed/hq_gen/balanced-v1/sources/*.jsonl`

Default composition in the current build:

- regenerated HQ: `10000`
- regenerated VL: `10000`
- Nomic general pairs: `10000`
- 2WikiMultiHopQA: `5000`
- QASC: `5000`
- HoVer: `4000`
- StrategyQA: `1603` (all train rows)
- total: `45603`

Realized per-source shares:

- `hq`: `21.93%`
- `vl`: `21.93%`
- `nomic_general`: `21.93%`
- `2wiki`: `10.96%`
- `qasc`: `10.96%`
- `hover`: `8.77%`
- `strategyqa`: `3.52%`

That keeps HQ + VL as the largest block, preserves a Nomic general-retrieval anchor, and keeps every single dataset well below the `30%` dominance threshold.

Important caveat:

- the referenced `Dzeniks/hover` mirror does not expose hop count, so HoVer falls back to label-balanced sampling instead of explicit `2/3/4`-hop stratification

The builder currently pulls and stores external dataset files under:

- `/mnt/ml_models/datasets/_raw`

Regenerate the staged mixed dataset with:

```bash
uv run python scripts/prepare_reasonir_mixed_dataset.py --force
```

Train on the mixed dataset with:

```bash
uv run python examples/train/ColBERT-zero/reasonir.py \
  --data-root /mnt/ml_models/datasets/ReasonIR/synthetic_data \
  --datasets mixed \
  --prompt-id hq_gen \
  --generator balanced-v1 \
  --max-negatives 1
```

Operational notes for the mixed path:

- the completed mixed baseline `reasonir-mixed-bs2048-lr8e5` reached `27.12` on full BRIGHT with GPT-4 reasoning traces vs base `26.51`
- the lower-LR mixed sweep at `bs=2048` is complete; its best run was `reasonir-mixed-lr-bs2048-lr5e-5` at `26.89`, which did not beat the `lr=8e-5` baseline
- the temperature sweep at `bs=2048 lr=8e-5` is also complete; its winner was `reasonir-mixed-temp-bs2048-lr8e-5-temp05` at `27.72`
- the current recommended checkpoint for GPT-trace BRIGHT is `/home/rbw/repo/pylate/output/reasonir-mixed-temp-bs2048-lr8e-5-temp05/final`
- the mixed path has dedicated launchers for both sweeps:
  [`scripts/reasonir_mixed_lr_sweep.sh`](/home/rbw/repo/pylate/scripts/reasonir_mixed_lr_sweep.sh) and
  [`scripts/reasonir_mixed_temp_sweep.sh`](/home/rbw/repo/pylate/scripts/reasonir_mixed_temp_sweep.sh)
- run that job on the stronger machine, not the local exploratory box
- launch it in `tmux`, `screen`, `nohup`, or equivalent; do not rely on an attached terminal for a long run
- the reusable full-BRIGHT GPT-trace wrapper is [`scripts/run_full_bright_gpt4_eval.sh`](/home/rbw/repo/pylate/scripts/run_full_bright_gpt4_eval.sh)
- that wrapper now supports `TASK_CONFIGS` for per-task batch/chunk settings on stronger GPUs
- the step-by-step launch and eval flow lives in [`REASONIR_MIXED_RUNBOOK.md`](/home/rbw/repo/pylate/REASONIR_MIXED_RUNBOOK.md)

## Training Script

The dedicated training entry point is:

- [`examples/train/ColBERT-zero/reasonir.py`](/home/rbw/repo/pylate/examples/train/ColBERT-zero/reasonir.py)

Key behavior:

- loads local HQ/VL ReasonIR JSONL triplets
- can also train from the staged `mixed` dataset group under `/mnt/ml_models/datasets/ReasonIR/synthetic_data`
- filters to rows with at least one positive and the requested number of negatives
- maps rows into `query`, `document`, `negative_0..negative_n`
- preserves `search_query:` and `search_document:` during training
- uses a prompt-aligned triplet evaluator when validation is enabled
- supports a writable HF datasets cache via `--dataset-cache-dir`
- now defaults `--data-root` to `/mnt/ml_models/datasets/ReasonIR/synthetic_data` when that path exists

Mixed-sweep runtime note:

- the mixed training entrypoint is epoch-based, not `max_steps`-based
- with the staged mixed dataset, train rows are currently `45146`
- at `bs=2048` with `dataloader_drop_last=True`, one epoch is `floor(45146 / 2048) = 22` steps
- so `epochs=3` produces `66` optimizer steps
- this differs from the HQ sweep wrappers, which use explicit `max_steps=100` for tighter hyperparameter comparisons

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

For the current first pass, keep using the shell-script workflow so the run order stays explicit while we are still actively comparing intermediate results.

The current training wrappers are:

- [`scripts/reasonir_hq_batch_sweep.sh`](/home/rbw/repo/pylate/scripts/reasonir_hq_batch_sweep.sh)
- [`scripts/reasonir_hq_lr_sweep.sh`](/home/rbw/repo/pylate/scripts/reasonir_hq_lr_sweep.sh)
- [`scripts/reasonir_hq_temp_sweep.sh`](/home/rbw/repo/pylate/scripts/reasonir_hq_temp_sweep.sh)

Shared defaults across the current shell workflow:

- `validation_size=0.01`
- `warmup_ratio=0.1`
- `mini_batch_size=32`
- `save_steps=25`
- `eval_steps=25`
- `save_total_limit=20`
- BRIGHT subset eval target: `biology,economics,robotics,pony`

### Stage 1

Stage 1 is a fairer batch-size comparison with roughly equal examples seen and learning rate scaled with batch size.

Run:

```bash
./scripts/reasonir_hq_batch_sweep.sh
```

The 4 Stage 1 configs are:

- `hq-batch-bs256-lr1e5-temp1` with `max_steps=400`
- `hq-batch-bs512-lr2e5-temp1` with `max_steps=200`
- `hq-batch-bs1024-lr4e5-temp1` with `max_steps=100`
- `hq-batch-bs2048-lr8e5-temp1` with `max_steps=50`

Current status from the first pass:

- best original Stage 1 sweep final: `hq-batch-bs2048-lr8e5-temp1` with BRIGHT subset `full_mean=8.91`
- exploratory `bs4096` extension: `hq-batch-bs4096-lr1e4-temp1`
- best observed checkpoint so far: `bs4096 checkpoint-5` with `full_mean=12.21`
- `bs4096 final` is only `8.95`, so the gain over `bs2048 final` is marginal and highly checkpoint-sensitive

That means the default next step is still the `bs2048` LR sweep. Treat `bs4096` as a special short-schedule experiment unless you also tighten checkpoint cadence for later stages.

The current Stage 2 sweep is intentionally centered higher than before so it covers the LR region that actually worked in Stage 1 at large batch sizes.

Updated reasoning-trace takeaway:

- the earlier 4-task GPT-4 gate correctly showed that raw-query Stage 1 ranking does not transfer cleanly
- the full 12-task GPT-4 BRIGHT comparison still ranks base first, but the gap is now small enough to justify continuing tuning
- full-BRIGHT GPT-4 ranking:
  - base `ColBERT-Zero`: `26.51`
  - `bs2048 checkpoint-50`: `26.01`
  - `bs4096 checkpoint-5`: `25.96`

Reviews:

- preliminary 4-task gate: [`reasonir-hq-stage1-gpt4-gate-review.md`](/home/rbw/repo/pylate/output/reasonir-hq-stage1-gpt4-gate-review.md)
- full 12-task comparison: [`reasonir-hq-full-gpt4-review.md`](/home/rbw/repo/pylate/output/reasonir-hq-full-gpt4-review.md)

So:

- if your target regime is raw queries, continue with the HQ Stage 2 sweep below
- if your target regime is GPT-4 reasoning traces, the mixed baseline is now the best result and the next step is the mixed `bs2048` lower-LR sweep

### Stage 2

Stage 2 is a learning-rate sweep at the winning batch size. Once Stage 1 BRIGHT results are reviewed, set `BEST_BATCH_SIZE` and run:

```bash
BEST_BATCH_SIZE=<WINNER_BATCH_SIZE> ./scripts/reasonir_hq_lr_sweep.sh
```

The Stage 2 LR values are:

- `2e-5`
- `5e-5`
- `8e-5`
- `1e-4`

### Stage 3

Stage 3 is a temperature sweep at the winning batch size and learning rate. Once Stage 2 BRIGHT results are reviewed, set both and run:

```bash
BEST_BATCH_SIZE=<WINNER_BATCH_SIZE> BEST_LR=<WINNER_LR> ./scripts/reasonir_hq_temp_sweep.sh
```

The Stage 3 temperatures are:

- `0.02`
- `0.05`
- `0.1`

### Post-hoc BRIGHT Eval

If you want additional BRIGHT subset comparisons after the pilot runs finish, use:

- [`scripts/reasonir_hq_bright_subset_eval.sh`](/home/rbw/repo/pylate/scripts/reasonir_hq_bright_subset_eval.sh) for retained `final` directories
- the same wrapper with `INCLUDE_CHECKPOINTS=1` for retained `checkpoint-*` plus `final`

These wrappers default to W&B logging with:

- project: `ColBERT-Zero`
- entity: `rbw`

They no longer force Hugging Face offline mode. For cached-only reruns, prefix them with:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
```

### Future Structured Path

Once the shell-script workflow is no longer the main path, the repo also has a W&B sweep-based version of the same three-stage process:

- runner: [`scripts/reasonir_hq_sweep_runner.py`](/home/rbw/repo/pylate/scripts/reasonir_hq_sweep_runner.py)
- Stage 1 config: [`sweeps/reasonir_hq_stage1_batch.yaml`](/home/rbw/repo/pylate/sweeps/reasonir_hq_stage1_batch.yaml)
- Stage 2 config: [`sweeps/reasonir_hq_stage2_lr.yaml`](/home/rbw/repo/pylate/sweeps/reasonir_hq_stage2_lr.yaml)
- Stage 3 config: [`sweeps/reasonir_hq_stage3_temp.yaml`](/home/rbw/repo/pylate/sweeps/reasonir_hq_stage3_temp.yaml)

Each sweep run trains with the official HQ script, evaluates the resulting `final` checkpoint on the BRIGHT subset, and logs `bright/summary/full_mean` into the same W&B run so Sweeps can optimize directly on the retrieval metric.

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
  --cache_dir /mnt/ml_models/cache/pylate-bright-cache \
  --cleanup-document-cache \
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
  --cache_dir /mnt/ml_models/cache/pylate-bright-cache \
  --cleanup-document-cache \
  --output_json /home/rbw/repo/pylate/output/bright-reasonir-full-gpu-gpt4.json
```

Operational note: exhaustive BRIGHT MaxSim scoring is expensive. On the 3090, the largest tasks such as `earth_science` and `stackoverflow` were slow enough that moving the sweep to a 5090 was judged worthwhile.
For large runs on this host, prefer `/mnt/ml_models/cache/pylate-bright-cache` over `/tmp` and keep `--cleanup-document-cache` enabled so model-specific BRIGHT shards are removed even after an interrupted run.

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
