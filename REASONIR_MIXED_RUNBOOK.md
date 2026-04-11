# ReasonIR Mixed Runbook

This runbook is the recommended workflow for continuing the mixed `ReasonIR` + `ColBERT-Zero` path after the first balanced mixed baseline finished and beat base `ColBERT-Zero` on full BRIGHT with GPT-4 reasoning traces.

Use this path for the mixed local dataset under `/mnt/ml_models/datasets/ReasonIR/synthetic_data/mixed/hq_gen/balanced-v1`, not for the official HQ sweep workflow.

## Scope

This covers:

- rebuilding the staged mixed dataset
- the completed mixed-data baseline result
- launching the mixed-data LR sweep
- running the job durably on a stronger machine
- monitoring the run in W&B
- running full BRIGHT GPT-trace evals after training

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
- W&B inspection helper:
  - [`scripts/wandb_project_runs.py`](/home/rbw/repo/pylate/scripts/wandb_project_runs.py)
- detailed mixed-data notes:
  - [`docs/documentation/reasonir-colbert-zero.md`](/home/rbw/repo/pylate/docs/documentation/reasonir-colbert-zero.md)

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

That makes the mixed baseline the current best result in this branch for the GPT-trace target regime.

## Step 3: Launch The Mixed LR Sweep

The next planned step is a lower-LR sweep at the same `bs=2048`:

- `5e-6`
- `1e-5`
- `3e-5`
- `5e-5`

The launcher is:

- [`scripts/reasonir_mixed_lr_sweep.sh`](/home/rbw/repo/pylate/scripts/reasonir_mixed_lr_sweep.sh)

It defaults to:

- dataset: `mixed/hq_gen/balanced-v1`
- `validation_size=0.01`
- `epochs=3`
- `bs=2048`
- `eval_bs=2048`
- `mini_batch_size=32`
- `save_steps=5`
- `eval_steps=5`
- `logging_steps=1`
- `save_total_limit=20`

### Recommended `tmux` Launch

Start a session:

```bash
tmux new -s reasonir-mixed-lr
```

Then run:

```bash
./scripts/reasonir_mixed_lr_sweep.sh
```

Detach with `Ctrl-b d`.

### Recommended `nohup` Launch

If you prefer a detached shell job:

```bash
mkdir -p /home/rbw/repo/pylate/output/logs
nohup ./scripts/reasonir_mixed_lr_sweep.sh \
  > /home/rbw/repo/pylate/output/logs/reasonir-mixed-lr-bs2048.log 2>&1 &
```

If you need to override the default LR list without editing the file:

```bash
LR_VALUES=5e-6,8e-6,1e-5,3e-5 ./scripts/reasonir_mixed_lr_sweep.sh
```

## Step 4: Monitor Training

Check W&B:

```bash
uv run python scripts/wandb_project_runs.py \
  --entity rbw \
  --project ColBERT-Zero \
  --name-contains reasonir-mixed-lr-bs2048 \
  --limit 5 \
  --history-tail 5
```

Expected W&B project:

- `rbw/ColBERT-Zero`

Expected output directories:

- `/home/rbw/repo/pylate/output/reasonir-mixed-lr-bs2048-lr5e-6`
- `/home/rbw/repo/pylate/output/reasonir-mixed-lr-bs2048-lr1e-5`
- `/home/rbw/repo/pylate/output/reasonir-mixed-lr-bs2048-lr3e-5`
- `/home/rbw/repo/pylate/output/reasonir-mixed-lr-bs2048-lr5e-5`

## Step 5: Run Full BRIGHT GPT-Trace Eval

After the sweep finishes, evaluate the most promising run or checkpoint with the reusable wrapper:

```bash
MODEL_PATH=/home/rbw/repo/pylate/output/reasonir-mixed-lr-bs2048-lr1e-5/final \
OUTPUT_JSON=/home/rbw/repo/pylate/output/reasonir-mixed-lr-bs2048-lr1e-5-gpt4-full.json \
RUN_NAME=eval-reasonir-mixed-lr-bs2048-lr1e-5-gpt4-full \
WANDB_GROUP=reasonir-mixed-gpt4-full \
bash scripts/run_full_bright_gpt4_eval.sh
```

If a single global config is leaving GPU throughput on the table, the wrapper also supports multiple task-specific configs through `TASK_CONFIGS`.

Format:

```text
tasks|query_batch|query_encode_batch|document_batch|corpus_chunk|document_cache_chunk
```

Separate multiple entries with `;`.

Example split tuned for a stronger GPU:

```bash
MODEL_PATH=/home/rbw/repo/pylate/output/reasonir-mixed-lr-bs2048-lr1e-5/final \
OUTPUT_JSON=/home/rbw/repo/pylate/output/reasonir-mixed-lr-bs2048-lr1e-5-gpt4-full.json \
RUN_NAME=eval-reasonir-mixed-lr-bs2048-lr1e-5-gpt4-full \
WANDB_GROUP=reasonir-mixed-gpt4-full \
TASK_CONFIGS='earth_science|32|128|256|256|256;biology,robotics,stackoverflow,aops,theoremqa_questions|64|128|256|512|512;economics,psychology,sustainable_living,leetcode,pony,theoremqa_theorems|64|128|256|1024|1024' \
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

Also keep the completed mixed baseline `reasonir-mixed-bs2048-lr8e5` as a direct control for the new LR runs.

## Notes

- the mixed builder keeps every single source below `30%` share so no one dataset dominates the mix
- HoVer is currently label-balanced, not hop-balanced, because the referenced HF mirror does not expose hop count
- the current next step is the lower-LR mixed sweep, not another rerun of the `8e-5` baseline
