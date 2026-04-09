# ReasonIR HQ Sweep Runbook

This runbook is the step-by-step workflow for tuning `ColBERT-Zero` on the official ReasonIR HQ dataset and selecting the best configuration with BRIGHT subset evals.

## Scope

This covers:

- Stage 1 batch-size sweep
- Stage 1 BRIGHT subset evals on `final`
- Stage 1 BRIGHT subset evals on retained `checkpoint-*`
- Stage 2 learning-rate sweep
- Stage 2 BRIGHT subset evals
- Stage 3 temperature sweep
- Stage 3 BRIGHT subset evals
- W&B review and run inspection

It assumes the repo-local defaults already in use on this machine:

- model: `/mnt/ml_models/lightonai/ColBERT-Zero`
- dataset cache: `/tmp/pylate-hf-cache`
- BRIGHT cache: `/tmp/pylate-bright-cache`
- W&B project: `ColBERT-Zero`
- W&B entity: `rbw`

## Before You Start

Make sure:

- `uv run wandb login --verify` succeeds
- the BRIGHT cache location has enough free disk
- the Hugging Face cache location is writable

Useful inspection command:

```bash
uv run python scripts/wandb_project_runs.py --entity rbw --project ColBERT-Zero --limit 10
```

## Step 1: Run The Batch Sweep

Run:

```bash
./scripts/reasonir_hq_batch_sweep.sh
```

What this does:

- trains 4 runs
- uses `validation_size=0.01`
- saves every `25` steps
- runs internal validation every `25` steps
- keeps up to `20` checkpoints

The 4 runs are:

- `hq-batch-bs256-lr1e5-temp1`
- `hq-batch-bs512-lr2e5-temp1`
- `hq-batch-bs1024-lr4e5-temp1`
- `hq-batch-bs2048-lr8e5-temp1`

## Step 2: Evaluate Batch Sweep Finals

After Stage 1 finishes, run:

```bash
STAGE=batch ./scripts/reasonir_hq_bright_subset_eval.sh
```

For Stage 1, do not set `BEST_BATCH_SIZE` or `BEST_LR`. The script will:

- evaluate the 4 Stage 1 `final` models

By default this evaluates:

- `biology`
- `economics`
- `robotics`
- `pony`

## Step 3: Evaluate Batch Sweep Checkpoints

Run:

```bash
STAGE=batch INCLUDE_CHECKPOINTS=1 INCLUDE_FINAL=0 ./scripts/reasonir_hq_bright_subset_eval.sh
```

This evaluates every retained `checkpoint-*` for those 4 runs without duplicating `final`.

## Step 4: Review Stage 1 Results

Use three sources:

1. W&B training runs
2. W&B BRIGHT eval runs
3. JSON outputs under `/home/rbw/repo/pylate/output`

Useful W&B queries:

```bash
uv run python scripts/wandb_project_runs.py \
  --entity rbw \
  --project ColBERT-Zero \
  --name-contains hq-batch \
  --limit 10
```

```bash
uv run python scripts/wandb_project_runs.py \
  --entity rbw \
  --project ColBERT-Zero \
  --group reasonir-hq-bright-subset \
  --limit 20
```

```bash
uv run python scripts/wandb_project_runs.py \
  --entity rbw \
  --project ColBERT-Zero \
  --group reasonir-hq-bright-checkpoints \
  --limit 20
```

Selection rule:

- choose by BRIGHT subset quality first
- use training stability and throughput as tie-breakers

## Step 5: Pick The Best Batch Size

Once Stage 1 is reviewed, choose:

```bash
export BEST_BATCH_SIZE=<winner>
```

Example:

```bash
export BEST_BATCH_SIZE=1024
```

## Step 6: Run The LR Sweep

Run:

```bash
BEST_BATCH_SIZE=$BEST_BATCH_SIZE ./scripts/reasonir_hq_lr_sweep.sh
```

This produces:

- `hq-lr-bs<BEST_BATCH_SIZE>-lr1e6-temp1`
- `hq-lr-bs<BEST_BATCH_SIZE>-lr5e6-temp1`
- `hq-lr-bs<BEST_BATCH_SIZE>-lr1e5-temp1`
- `hq-lr-bs<BEST_BATCH_SIZE>-lr5e5-temp1`

## Step 7: Evaluate LR Sweep Finals

Run:

```bash
STAGE=lr BEST_BATCH_SIZE=$BEST_BATCH_SIZE ./scripts/reasonir_hq_bright_subset_eval.sh
```

At this point the script will evaluate the Stage 2 LR runs for the chosen batch size.

## Step 8: Evaluate LR Sweep Checkpoints

Run:

```bash
STAGE=lr BEST_BATCH_SIZE=$BEST_BATCH_SIZE INCLUDE_CHECKPOINTS=1 INCLUDE_FINAL=0 ./scripts/reasonir_hq_bright_subset_eval.sh
```

## Step 9: Pick The Best LR

Choose:

```bash
export BEST_LR=<winner>
```

Example:

```bash
export BEST_LR=1e-5
```

## Step 10: Run The Temperature Sweep

Run:

```bash
BEST_BATCH_SIZE=$BEST_BATCH_SIZE BEST_LR=$BEST_LR ./scripts/reasonir_hq_temp_sweep.sh
```

This produces:

- `hq-temp-bs<BEST_BATCH_SIZE>-lr${BEST_LR//./}-temp002`
- `hq-temp-bs<BEST_BATCH_SIZE>-lr${BEST_LR//./}-temp005`
- `hq-temp-bs<BEST_BATCH_SIZE>-lr${BEST_LR//./}-temp01`

## Step 11: Evaluate Temperature Sweep Finals

Run:

```bash
STAGE=temp BEST_BATCH_SIZE=$BEST_BATCH_SIZE BEST_LR=$BEST_LR ./scripts/reasonir_hq_bright_subset_eval.sh
```

Now the script will evaluate the Stage 3 temp runs for the chosen batch size and LR.

## Step 12: Evaluate Temperature Sweep Checkpoints

Run:

```bash
STAGE=temp BEST_BATCH_SIZE=$BEST_BATCH_SIZE BEST_LR=$BEST_LR INCLUDE_CHECKPOINTS=1 INCLUDE_FINAL=0 ./scripts/reasonir_hq_bright_subset_eval.sh
```

## Step 13: Pick The Winner

Final selection should be based on:

- BRIGHT subset `final` performance
- checkpoint trajectory if a run peaked early
- training stability
- runtime and throughput

Do not choose purely from training loss.

## Optional Flags

Disable W&B during eval:

```bash
REPORT_TO=none ./scripts/reasonir_hq_bright_subset_eval.sh
```

Keep BRIGHT document caches between eval runs:

```bash
CLEANUP_DOCUMENT_CACHE=0 ./scripts/reasonir_hq_bright_subset_eval.sh
```

Force cached-only BRIGHT reruns:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CLEANUP_DOCUMENT_CACHE=0 ./scripts/reasonir_hq_bright_subset_eval.sh
```

## Related Files

- [`examples/train/ColBERT-zero/reason_moderncolbert.py`](/home/rbw/repo/pylate/examples/train/ColBERT-zero/reason_moderncolbert.py)
- [`scripts/reasonir_hq_batch_sweep.sh`](/home/rbw/repo/pylate/scripts/reasonir_hq_batch_sweep.sh)
- [`scripts/reasonir_hq_lr_sweep.sh`](/home/rbw/repo/pylate/scripts/reasonir_hq_lr_sweep.sh)
- [`scripts/reasonir_hq_temp_sweep.sh`](/home/rbw/repo/pylate/scripts/reasonir_hq_temp_sweep.sh)
- [`scripts/reasonir_hq_bright_subset_eval.sh`](/home/rbw/repo/pylate/scripts/reasonir_hq_bright_subset_eval.sh)
- [`scripts/wandb_project_runs.py`](/home/rbw/repo/pylate/scripts/wandb_project_runs.py)
- [`docs/documentation/reasonir-colbert-zero.md`](/home/rbw/repo/pylate/docs/documentation/reasonir-colbert-zero.md)
- [`BRIGHT_SUBSET_RUNBOOK.md`](/home/rbw/repo/pylate/BRIGHT_SUBSET_RUNBOOK.md)
