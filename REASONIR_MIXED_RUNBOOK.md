# ReasonIR Mixed Runbook

This runbook is the recommended workflow for continuing the mixed `ReasonIR` + `ColBERT-Zero` path after the first balanced mixed baseline finished, the lower-LR sweep completed, and the temperature sweep produced a new best checkpoint on full BRIGHT with GPT-4 reasoning traces.

Use this path for the mixed local dataset under `/mnt/ml_models/datasets/ReasonIR/synthetic_data/mixed/hq_gen/balanced-v1`, not for the official HQ sweep workflow.

## Scope

This covers:

- rebuilding the staged mixed dataset
- the completed mixed-data baseline result
- the completed mixed-data LR sweep
- the completed mixed-data temperature sweep
- the current best checkpoint to use for future evals
- the next tuning step if more search is justified

## Files

- mixed dataset builder:
  - [`scripts/prepare_reasonir_mixed_dataset.py`](/home/rbw/repo/pylate/scripts/prepare_reasonir_mixed_dataset.py)
- local training entrypoint:
  - [`examples/train/ColBERT-zero/reasonir.py`](/home/rbw/repo/pylate/examples/train/ColBERT-zero/reasonir.py)
- BRIGHT evaluator:
  - [`examples/evaluation/bright_reasonir.py`](/home/rbw/repo/pylate/examples/evaluation/bright_reasonir.py)
- reusable full GPT-trace BRIGHT wrapper:
  - [`scripts/run_full_bright_gpt4_eval.sh`](/home/rbw/repo/pylate/scripts/run_full_bright_gpt4_eval.sh)
- mixed LR sweep launcher:
  - [`scripts/reasonir_mixed_lr_sweep.sh`](/home/rbw/repo/pylate/scripts/reasonir_mixed_lr_sweep.sh)
- mixed temperature sweep launcher:
  - [`scripts/reasonir_mixed_temp_sweep.sh`](/home/rbw/repo/pylate/scripts/reasonir_mixed_temp_sweep.sh)
- mixed temperature GPT-4 batch eval helper:
  - [`scripts/reasonir_mixed_temp_gpt4_eval_batch.sh`](/home/rbw/repo/pylate/scripts/reasonir_mixed_temp_gpt4_eval_batch.sh)
- W&B inspection helper:
  - [`scripts/wandb_project_runs.py`](/home/rbw/repo/pylate/scripts/wandb_project_runs.py)
- detailed mixed-data notes:
  - [`docs/documentation/reasonir-colbert-zero.md`](/home/rbw/repo/pylate/docs/documentation/reasonir-colbert-zero.md)
- lower-LR sweep result comparison:
  - [`docs/documentation/reasonir-mixed-lr-sweep-results.md`](/home/rbw/repo/pylate/docs/documentation/reasonir-mixed-lr-sweep-results.md)
- temperature sweep result comparison:
  - [`docs/documentation/reasonir-mixed-temp-sweep-results.md`](/home/rbw/repo/pylate/docs/documentation/reasonir-mixed-temp-sweep-results.md)

## Before You Start

Make sure:

- the stronger machine has the base checkpoint available, typically `/mnt/ml_models/lightonai/ColBERT-Zero`
- `/mnt/ml_models/datasets/ReasonIR/synthetic_data` is writable
- `/mnt/ml_models/cache/pylate-bright-cache` has enough space for BRIGHT caches
- `uv run wandb login --verify` succeeds

Important operational note:

- do not launch long mixed-data runs from an attached terminal session
- a prior attached run, `reasonir-mixed-bs2048-lr8e5`, crashed before the first saved step when the terminal/session disappeared
- use `tmux`, `screen`, `nohup`, or an equivalent detached launcher on the stronger machine

## Step 1: Rebuild The Mixed Dataset

Run:

```bash
uv run python scripts/prepare_reasonir_mixed_dataset.py --force
```

That stages:

- `/mnt/ml_models/datasets/ReasonIR/synthetic_data/mixed/hq_gen/balanced-v1/final_train_data.jsonl`
- `/mnt/ml_models/datasets/ReasonIR/synthetic_data/mixed/hq_gen/balanced-v1/manifest.json`
- `/mnt/ml_models/datasets/ReasonIR/synthetic_data/mixed/hq_gen/balanced-v1/README.md`

Current realized totals:

- total rows: `45603`
- HQ: `10000`
- VL: `10000`
- Nomic general: `10000`
- 2Wiki: `5000`
- QASC: `5000`
- HoVer: `4000`
- StrategyQA: `1603`

## Step 2: Mixed Baseline Result

The first mixed-data baseline is already complete:

- run: `reasonir-mixed-bs2048-lr8e5`
- dataset: `mixed/hq_gen/balanced-v1`
- `validation_size=0.01`
- `epochs=3`
- `lr=8e-5`
- `bs=2048`
- `eval_bs=2048`
- `mini_batch_size=32`
- `save_steps=5`
- `eval_steps=5`
- `logging_steps=1`
- `save_total_limit=20`

Full BRIGHT with GPT-4 reasoning traces:

- mixed baseline: `27.12`
- base `ColBERT-Zero`: `26.51`
- delta: `+0.61`

That baseline beat base `ColBERT-Zero`, but it is no longer the current best result after the completed temperature sweep.

## Step 3: Completed Mixed LR Sweep

The lower-LR sweep at the same `bs=2048` is complete:

- `5e-6`
- `1e-5`
- `3e-5`
- `5e-5`

Launcher:

- [`scripts/reasonir_mixed_lr_sweep.sh`](/home/rbw/repo/pylate/scripts/reasonir_mixed_lr_sweep.sh)

Full GPT-4 BRIGHT means:

- `lr=5e-6`: `26.60`
- `lr=1e-5`: `26.26`
- `lr=3e-5`: `26.81`
- `lr=5e-5`: `26.89`

Result:

- no lower-LR run beat the original mixed baseline `27.12`
- the best lower-LR run was `reasonir-mixed-lr-bs2048-lr5e-5` at `26.89`
- details live in [`docs/documentation/reasonir-mixed-lr-sweep-results.md`](/home/rbw/repo/pylate/docs/documentation/reasonir-mixed-lr-sweep-results.md)

## Step 4: Completed Mixed Temperature Sweep

The next sweep was temperature at the original winning `bs=2048 lr=8e-5` setting.

Launcher:

- [`scripts/reasonir_mixed_temp_sweep.sh`](/home/rbw/repo/pylate/scripts/reasonir_mixed_temp_sweep.sh)

Evaluated temperatures:

- `0.05`
- `0.1`
- `0.25`
- `0.5`

Full GPT-4 BRIGHT means:

- `temp=0.05`: `26.91`
- `temp=0.1`: `27.56`
- `temp=0.25`: `27.48`
- `temp=0.5`: `27.72`
- prior mixed baseline `temp=1.0`: `27.12`

Result:

- `temp=0.5` is the current best mixed-model checkpoint for the GPT-trace BRIGHT target regime
- `temp=0.5` improved the prior mixed baseline by `+0.60`
- `temp=0.1` and `temp=0.25` also beat the prior mixed baseline
- details live in [`docs/documentation/reasonir-mixed-temp-sweep-results.md`](/home/rbw/repo/pylate/docs/documentation/reasonir-mixed-temp-sweep-results.md)

The reusable batch eval helper for the completed temperature sweep is:

- [`scripts/reasonir_mixed_temp_gpt4_eval_batch.sh`](/home/rbw/repo/pylate/scripts/reasonir_mixed_temp_gpt4_eval_batch.sh)

## Step 5: Current Recommended Checkpoint

Use this checkpoint as the default mixed-model candidate for future BRIGHT evals:

- `/home/rbw/repo/pylate/output/reasonir-mixed-temp-bs2048-lr8e-5-temp05/final`

If you want the strongest current alternative that leans more toward coding and theorem-style tasks, keep this one nearby as well:

- `/home/rbw/repo/pylate/output/reasonir-mixed-temp-bs2048-lr8e-5-temp01/final`

Evaluate the current winner with the reusable wrapper:

```bash
MODEL_PATH=/home/rbw/repo/pylate/output/reasonir-mixed-temp-bs2048-lr8e-5-temp05/final \
OUTPUT_JSON=/home/rbw/repo/pylate/output/reasonir-mixed-temp-bs2048-lr8e-5-temp05-gpt4-full.json \
RUN_NAME=eval-reasonir-mixed-temp-bs2048-lr8e-5-temp05-gpt4-full \
WANDB_GROUP=reasonir-mixed-gpt4-full \
bash scripts/run_full_bright_gpt4_eval.sh
```

## Step 6: Compare Against Controls

Run the same full BRIGHT GPT-trace eval on base `ColBERT-Zero` if you need a fresh control on the stronger machine:

```bash
MODEL_PATH=/mnt/ml_models/lightonai/ColBERT-Zero \
OUTPUT_JSON=/home/rbw/repo/pylate/output/bright-base-colbert-zero-gpt4-full.json \
RUN_NAME=eval-colbert-zero-gpt4-full \
WANDB_GROUP=reasonir-mixed-gpt4-full \
bash scripts/run_full_bright_gpt4_eval.sh
```

Also keep these completed controls in view:

- mixed baseline `reasonir-mixed-bs2048-lr8e5`: `27.12`
- best lower-LR run `reasonir-mixed-lr-bs2048-lr5e-5`: `26.89`
- current winner `reasonir-mixed-temp-bs2048-lr8e-5-temp05`: `27.72`

## Step 7: Next Planned Work

The next move should not be another broad LR sweep.

Recommended order:

- treat `temp=0.5` as the promoted default mixed checkpoint
- if you want variance/stability signal, rerun `temp=0.5` once before opening another broad sweep
- if more tuning is still justified, refine temperature around the winning region instead of reopening LR; the obvious follow-up window is roughly `temp=0.25` to `0.75`
- if you need stricter artifact parity for reporting, produce a fresh local base-model full GPT-4 BRIGHT JSON on the same machine

## Notes

- the mixed builder keeps every single source below `30%` share so no one dataset dominates the mix
- HoVer is currently label-balanced, not hop-balanced, because the referenced HF mirror does not expose hop count
- the reusable GPT-trace eval wrapper supports task-specific configs through `TASK_CONFIGS`
- on this machine, prefer disk-backed temp/cache roots such as `${HOME}/.temp` over `/tmp` to avoid `tmpfs` exhaustion during BRIGHT evals
