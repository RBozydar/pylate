# ReasonIR Mixed Runbook

This runbook is the recommended workflow for training `ColBERT-Zero` on the staged balanced mixed ReasonIR dataset and then evaluating the resulting checkpoint on full BRIGHT with GPT-4 reasoning traces.

Use this path for the mixed local dataset under `/mnt/ml_models/datasets/ReasonIR/synthetic_data/mixed/hq_gen/balanced-v1`, not for the official HQ sweep workflow.

## Scope

This covers:

- rebuilding the staged mixed dataset
- launching the mixed-data fine-tune
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

## Step 2: Launch Training On The Stronger Machine

The current first mixed-data training config is:

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

### Recommended `tmux` Launch

Start a session:

```bash
tmux new -s reasonir-mixed
```

Then run:

```bash
uv run python -u examples/train/ColBERT-zero/reasonir.py \
  --data-root /mnt/ml_models/datasets/ReasonIR/synthetic_data \
  --datasets mixed \
  --prompt-id hq_gen \
  --generator balanced-v1 \
  --validation-size 0.01 \
  --epochs 3 \
  --lr 8e-5 \
  --bs 2048 \
  --eval-bs 2048 \
  --mini-batch-size 32 \
  --fp16 \
  --num-workers 4 \
  --save-steps 5 \
  --eval-steps 5 \
  --logging-steps 1 \
  --save-total-limit 20 \
  --run-name reasonir-mixed-bs2048-lr8e5 \
  --dataset-cache-dir /tmp/pylate-hf-cache \
  --output-dir /home/rbw/repo/pylate/output/reasonir-mixed-bs2048-lr8e5
```

Detach with `Ctrl-b d`.

### Recommended `nohup` Launch

If you prefer a detached shell job:

```bash
mkdir -p /home/rbw/repo/pylate/output/logs
nohup uv run python -u examples/train/ColBERT-zero/reasonir.py \
  --data-root /mnt/ml_models/datasets/ReasonIR/synthetic_data \
  --datasets mixed \
  --prompt-id hq_gen \
  --generator balanced-v1 \
  --validation-size 0.01 \
  --epochs 3 \
  --lr 8e-5 \
  --bs 2048 \
  --eval-bs 2048 \
  --mini-batch-size 32 \
  --fp16 \
  --num-workers 4 \
  --save-steps 5 \
  --eval-steps 5 \
  --logging-steps 1 \
  --save-total-limit 20 \
  --run-name reasonir-mixed-bs2048-lr8e5 \
  --dataset-cache-dir /tmp/pylate-hf-cache \
  --output-dir /home/rbw/repo/pylate/output/reasonir-mixed-bs2048-lr8e5 \
  > /home/rbw/repo/pylate/output/logs/reasonir-mixed-bs2048-lr8e5.log 2>&1 &
```

## Step 3: Monitor Training

Check W&B:

```bash
uv run python scripts/wandb_project_runs.py \
  --entity rbw \
  --project ColBERT-Zero \
  --name-contains reasonir-mixed-bs2048-lr8e5 \
  --limit 5 \
  --history-tail 5
```

Expected output path:

- `/home/rbw/repo/pylate/output/reasonir-mixed-bs2048-lr8e5`

Expected W&B project:

- `rbw/ColBERT-Zero`

## Step 4: Run Full BRIGHT GPT-Trace Eval

After training finishes, evaluate the final checkpoint with the reusable wrapper:

```bash
MODEL_PATH=/home/rbw/repo/pylate/output/reasonir-mixed-bs2048-lr8e5/final \
OUTPUT_JSON=/home/rbw/repo/pylate/output/reasonir-mixed-bs2048-lr8e5-gpt4-full.json \
RUN_NAME=eval-reasonir-mixed-bs2048-lr8e5-gpt4-full \
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
MODEL_PATH=/home/rbw/repo/pylate/output/reasonir-mixed-bs2048-lr8e5/final \
OUTPUT_JSON=/home/rbw/repo/pylate/output/reasonir-mixed-bs2048-lr8e5-gpt4-full.json \
RUN_NAME=eval-reasonir-mixed-bs2048-lr8e5-gpt4-full \
WANDB_GROUP=reasonir-mixed-gpt4-full \
TASK_CONFIGS='earth_science|32|128|256|256|256;biology,robotics,stackoverflow,aops,theoremqa_questions|64|128|256|512|512;economics,psychology,sustainable_living,leetcode,pony,theoremqa_theorems|64|128|256|1024|1024' \
bash scripts/run_full_bright_gpt4_eval.sh
```

## Step 5: Compare Against Base

Run the same full BRIGHT GPT-trace eval on base `ColBERT-Zero` if you need a fresh control on the stronger machine:

```bash
MODEL_PATH=/mnt/ml_models/lightonai/ColBERT-Zero \
OUTPUT_JSON=/home/rbw/repo/pylate/output/bright-base-colbert-zero-gpt4-full.json \
RUN_NAME=eval-colbert-zero-gpt4-full \
WANDB_GROUP=reasonir-mixed-gpt4-full \
bash scripts/run_full_bright_gpt4_eval.sh
```

## Notes

- the mixed builder keeps every single source below `30%` share so no one dataset dominates the mix
- HoVer is currently label-balanced, not hop-balanced, because the referenced HF mirror does not expose hop count
- `bs=2048` is the first planned mixed-data run, not a fully tuned optimum yet
