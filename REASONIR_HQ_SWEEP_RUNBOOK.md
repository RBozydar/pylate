# ReasonIR HQ Sweep Runbook

This runbook is the future structured workflow for tuning `ColBERT-Zero` on the official ReasonIR HQ dataset with W&B Sweeps and selecting the best configuration from BRIGHT subset results.

The current first-pass workflow in this repo is still the shell-script path documented in [`docs/documentation/reasonir-colbert-zero.md`](/home/rbw/repo/pylate/docs/documentation/reasonir-colbert-zero.md). Use this runbook when you explicitly want to move that process into W&B Sweeps rather than replacing the in-flight shell workflow.

## Scope

This covers:

- Stage 1 batch sweep
- Stage 2 learning-rate sweep
- Stage 3 temperature sweep
- optional post-hoc BRIGHT subset evals on retained `checkpoint-*` plus `final`
- W&B review and run inspection

It assumes the repo-local defaults already in use on this machine:

- model: `/mnt/ml_models/lightonai/ColBERT-Zero`
- dataset cache: `/tmp/pylate-hf-cache`
- BRIGHT cache: `/tmp/pylate-bright-cache`
- W&B project: `ColBERT-Zero`
- W&B entity: `rbw`

## Files

- sweep runner:
  - [`scripts/reasonir_hq_sweep_runner.py`](/home/rbw/repo/pylate/scripts/reasonir_hq_sweep_runner.py)
- sweep configs:
  - [`sweeps/reasonir_hq_stage1_batch.yaml`](/home/rbw/repo/pylate/sweeps/reasonir_hq_stage1_batch.yaml)
  - [`sweeps/reasonir_hq_stage2_lr.yaml`](/home/rbw/repo/pylate/sweeps/reasonir_hq_stage2_lr.yaml)
  - [`sweeps/reasonir_hq_stage3_temp.yaml`](/home/rbw/repo/pylate/sweeps/reasonir_hq_stage3_temp.yaml)
- current shell-script workflow:
  - [`scripts/reasonir_hq_batch_sweep.sh`](/home/rbw/repo/pylate/scripts/reasonir_hq_batch_sweep.sh)
  - [`scripts/reasonir_hq_lr_sweep.sh`](/home/rbw/repo/pylate/scripts/reasonir_hq_lr_sweep.sh)
  - [`scripts/reasonir_hq_temp_sweep.sh`](/home/rbw/repo/pylate/scripts/reasonir_hq_temp_sweep.sh)
- post-hoc BRIGHT eval wrappers:
  - [`scripts/reasonir_hq_bright_subset_eval.sh`](/home/rbw/repo/pylate/scripts/reasonir_hq_bright_subset_eval.sh)
  - the same wrapper with `INCLUDE_CHECKPOINTS=1`
- W&B inspection helper:
  - [`scripts/wandb_project_runs.py`](/home/rbw/repo/pylate/scripts/wandb_project_runs.py)

## Before You Start

Make sure:

- `uv run wandb login --verify` succeeds
- the BRIGHT cache location has enough free disk
- the Hugging Face cache location is writable

Useful inspection command:

```bash
uv run python scripts/wandb_project_runs.py --entity rbw --project ColBERT-Zero --limit 10
```

## How The Sweep Runner Works

Each W&B sweep run:

- trains with the official HQ script
- saves checkpoints every `25` steps
- runs internal HQ validation every `25` steps
- retains up to `20` checkpoints
- evaluates the resulting `final` checkpoint on the BRIGHT subset
- logs `bright/summary/full_mean` into the same W&B run

That means the sweep selection metric already lives on the sweep runs themselves. You do not need a second wrapper just to compare final checkpoints.

## Current Manual First Pass

If you are continuing the current in-flight workflow, do not start here with `wandb sweep`.

Use the shell scripts instead:

```bash
./scripts/reasonir_hq_batch_sweep.sh
BEST_BATCH_SIZE=<WINNER_BATCH_SIZE> ./scripts/reasonir_hq_lr_sweep.sh
BEST_BATCH_SIZE=<WINNER_BATCH_SIZE> BEST_LR=<WINNER_LR> ./scripts/reasonir_hq_temp_sweep.sh
./scripts/reasonir_hq_bright_subset_eval.sh
INCLUDE_CHECKPOINTS=1 ./scripts/reasonir_hq_bright_subset_eval.sh
```

Current Stage 1 status in this repo:

- best original sweep final: `hq-batch-bs2048-lr8e5-temp1` at `8.91`
- exploratory `bs4096` extension best checkpoint: `hq-batch-bs4096-lr1e4-temp1 checkpoint-5` at `12.21`
- exploratory `bs4096` final: `8.95`

That makes `bs2048` the safer default for the scripted LR sweep, unless you intentionally redesign the later stages for `bs4096` with shorter runs and denser checkpointing.

The numbered steps below are only for the future W&B Sweeps workflow.

## Sweep Step 1: Run The Batch Sweep

Initialize the Stage 1 sweep:

```bash
uv run wandb sweep --project ColBERT-Zero sweeps/reasonir_hq_stage1_batch.yaml
```

Then start one or more agents:

```bash
uv run wandb agent rbw/ColBERT-Zero/<sweep-id>
```

The original Stage 1 configs are:

- `hq-batch-bs256-lr1e5-temp1`
- `hq-batch-bs512-lr2e5-temp1`
- `hq-batch-bs1024-lr4e5-temp1`
- `hq-batch-bs2048-lr8e5-temp1`

An exploratory extension, `hq-batch-bs4096-lr1e4-temp1`, was evaluated separately and peaked very early.

## Sweep Step 2: Review Stage 1 Results

Use:

1. the Stage 1 sweep runs themselves
2. optional checkpoint BRIGHT evals if you care about early peaks
3. local JSON outputs under `/home/rbw/repo/pylate/output`

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
  --group reasonir-hq-bright-checkpoints \
  --limit 20
```

Selection rule:

- choose by `bright/summary/full_mean` first
- use checkpoint trajectory, training stability, and throughput as tie-breakers

Specific caution for `bs4096`:

- the best observed result occurs at `checkpoint-5`, not at `final`
- do not carry `bs4096` into later stages with the default `MAX_STEPS=100` and `SAVE_STEPS=25` unchanged

## Optional: Evaluate Stage 1 Checkpoints

If you want BRIGHT subset scores for retained checkpoints as well as `final`, run:

```bash
INCLUDE_CHECKPOINTS=1 ./scripts/reasonir_hq_bright_subset_eval.sh \
  hq-batch-bs256-lr1e5-temp1 \
  hq-batch-bs512-lr2e5-temp1 \
  hq-batch-bs1024-lr4e5-temp1 \
  hq-batch-bs2048-lr8e5-temp1
```

Those evals log to W&B group `reasonir-hq-bright-checkpoints`.

## Sweep Step 3: Pick The Best Batch Size

Once Stage 1 is reviewed, update [`reasonir_hq_stage2_lr.yaml`](/home/rbw/repo/pylate/sweeps/reasonir_hq_stage2_lr.yaml):

- set `batch_size.value` to the winning batch size

## Sweep Step 4: Run The LR Sweep

Initialize the Stage 2 sweep:

```bash
uv run wandb sweep --project ColBERT-Zero sweeps/reasonir_hq_stage2_lr.yaml
```

Then start an agent:

```bash
uv run wandb agent rbw/ColBERT-Zero/<sweep-id>
```

The Stage 2 LR values are:

- `2e-5`
- `5e-5`
- `8e-5`
- `1e-4`

## Sweep Step 5: Review Stage 2 Results

Use the same pattern as Stage 1:

- sweep-run `bright/summary/full_mean` first
- optional checkpoint BRIGHT evals second
- training stability and throughput as tie-breakers

Optional checkpoint eval:

```bash
INCLUDE_CHECKPOINTS=1 ./scripts/reasonir_hq_bright_subset_eval.sh \
  hq-lr-bs<WINNER_BATCH>-lr1e6-temp1 \
  hq-lr-bs<WINNER_BATCH>-lr5e6-temp1 \
  hq-lr-bs<WINNER_BATCH>-lr1e5-temp1 \
  hq-lr-bs<WINNER_BATCH>-lr5e5-temp1
```

## Sweep Step 6: Pick The Best LR

Once Stage 2 is reviewed, update [`reasonir_hq_stage3_temp.yaml`](/home/rbw/repo/pylate/sweeps/reasonir_hq_stage3_temp.yaml):

- set `batch_size.value`
- set `learning_rate.value`

## Sweep Step 7: Run The Temperature Sweep

Initialize the Stage 3 sweep:

```bash
uv run wandb sweep --project ColBERT-Zero sweeps/reasonir_hq_stage3_temp.yaml
```

Then start an agent:

```bash
uv run wandb agent rbw/ColBERT-Zero/<sweep-id>
```

The Stage 3 temperatures are:

- `0.02`
- `0.05`
- `0.1`

## Sweep Step 8: Pick The Winner

Final selection should be based on:

- BRIGHT subset `final` performance from the sweep runs
- checkpoint trajectory if a run peaked early
- training stability
- runtime and throughput

Do not choose purely from training loss.

## Optional Flags

Disable W&B during post-hoc eval:

```bash
REPORT_TO=none ./scripts/reasonir_hq_bright_subset_eval.sh
```

Force cached-only BRIGHT reruns:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 ./scripts/reasonir_hq_bright_subset_eval.sh
```

If you are still in the first manual pass, keep using the shell scripts and treat this runbook as future infrastructure.
