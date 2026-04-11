# Codex Handoff Note

This note is the fast handoff for continuing the mixed `ReasonIR` + `ColBERT-Zero` work from another Codex session.

## Current State

- repo branch: `reasonir-colbert-zero-handoff`
- latest pushed commits on this branch:
  - `0774f81` `Add mixed ReasonIR training workflow`
  - `b3d0980` `Document mixed training handoff`
  - `f7b87ff` `Support multi-config BRIGHT eval wrapper`
- current best mixed result:
  - run: `reasonir-mixed-bs2048-lr8e5`
  - full BRIGHT with GPT-4 reasoning traces: `27.12`
  - base `ColBERT-Zero`: `26.51`
  - delta: `+0.61`

So the mixed dataset path is currently the best-performing GPT-trace configuration in this branch.

## Primary Files

- mixed runbook:
  - [`REASONIR_MIXED_RUNBOOK.md`](/home/rbw/repo/pylate/REASONIR_MIXED_RUNBOOK.md)
- mixed dataset builder:
  - [`scripts/prepare_reasonir_mixed_dataset.py`](/home/rbw/repo/pylate/scripts/prepare_reasonir_mixed_dataset.py)
- local training entrypoint:
  - [`examples/train/ColBERT-zero/reasonir.py`](/home/rbw/repo/pylate/examples/train/ColBERT-zero/reasonir.py)
- reusable full GPT-trace BRIGHT eval wrapper:
  - [`scripts/run_full_bright_gpt4_eval.sh`](/home/rbw/repo/pylate/scripts/run_full_bright_gpt4_eval.sh)
- mixed LR sweep launcher:
  - [`scripts/reasonir_mixed_lr_sweep.sh`](/home/rbw/repo/pylate/scripts/reasonir_mixed_lr_sweep.sh)
- broader mixed-data notes:
  - [`docs/documentation/reasonir-colbert-zero.md`](/home/rbw/repo/pylate/docs/documentation/reasonir-colbert-zero.md)
- compact historical context:
  - [`HANDOFF.md`](/home/rbw/repo/pylate/HANDOFF.md)

If the next Codex instance only reads one file first, it should read:

- [`REASONIR_MIXED_RUNBOOK.md`](/home/rbw/repo/pylate/REASONIR_MIXED_RUNBOOK.md)

## What Was Added This Session

- built a balanced mixed dataset workflow combining:
  - regenerated HQ
  - regenerated VL
  - Nomic general pairs
  - 2WikiMultiHopQA
  - QASC
  - HoVer
  - StrategyQA
- staged that dataset under `/mnt`
- updated the local training script so `--data-root` prefers `/mnt/ml_models/datasets/ReasonIR/synthetic_data`
- documented the mixed training flow
- made the full BRIGHT GPT-trace wrapper reusable via env vars
- added the mixed `bs=2048` lower-LR sweep launcher

## Staged Dataset

The mixed dataset is already created at:

- [final_train_data.jsonl](/mnt/ml_models/datasets/ReasonIR/synthetic_data/mixed/hq_gen/balanced-v1/final_train_data.jsonl)
- [manifest.json](/mnt/ml_models/datasets/ReasonIR/synthetic_data/mixed/hq_gen/balanced-v1/manifest.json)
- [README.md](/mnt/ml_models/datasets/ReasonIR/synthetic_data/mixed/hq_gen/balanced-v1/README.md)

If the other machine has access to the same `/mnt`, it does **not** need to rebuild the dataset first.

Current realized mix:

- HQ: `10000`
- VL: `10000`
- Nomic general: `10000`
- 2Wiki: `5000`
- QASC: `5000`
- HoVer: `4000`
- StrategyQA: `1603`
- total: `45603`

Per-source shares:

- HQ: `21.93%`
- VL: `21.93%`
- Nomic general: `21.93%`
- 2Wiki: `10.96%`
- QASC: `10.96%`
- HoVer: `8.77%`
- StrategyQA: `3.52%`

Important caveat:

- the `Dzeniks/hover` dataset mirror does not expose hop count
- the builder therefore falls back to label-balanced HoVer sampling, not explicit `2/3/4`-hop stratification

## Next Planned Training Step

The next step is a lower-LR mixed sweep at `bs=2048`, using:

- `5e-6`
- `1e-5`
- `3e-5`
- `5e-5`

Launcher:

```bash
./scripts/reasonir_mixed_lr_sweep.sh
```

Defaults baked into that script:

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

If the LR list needs to change without editing the script:

```bash
LR_VALUES=5e-6,8e-6,1e-5,3e-5 ./scripts/reasonir_mixed_lr_sweep.sh
```

## Critical Operational Note

Do **not** launch the sweep or any rerun from an attached terminal.

What happened here:

- I started `reasonir-mixed-bs2048-lr8e5` interactively
- the session/terminal later disappeared overnight
- the W&B run exists and is marked crashed:
  - `http://192.168.1.19:8080/rbw/ColBERT-Zero/runs/5ked03dx`
- it crashed before first saved step
- no checkpoint or final model was written under:
  - [`output/reasonir-mixed-bs2048-lr8e5`](/home/rbw/repo/pylate/output/reasonir-mixed-bs2048-lr8e5)

Interpretation:

- this was an execution-mode failure, not evidence that the script/config is invalid
- the next machine should use `tmux`, `screen`, `nohup`, or a job scheduler

## Recommended Launch Mode

Preferred:

```bash
tmux new -s reasonir-mixed-lr
```

Then run:

```bash
./scripts/reasonir_mixed_lr_sweep.sh
```

Detach with `Ctrl-b d`.

Alternative:

```bash
mkdir -p /home/rbw/repo/pylate/output/logs
nohup ./scripts/reasonir_mixed_lr_sweep.sh \
  > /home/rbw/repo/pylate/output/logs/reasonir-mixed-lr-bs2048.log 2>&1 &
```

## W&B

Defaults baked into training/eval scripts:

- project: `ColBERT-Zero`
- entity: `rbw`
- base URL in this environment: `http://192.168.1.19:8080`

Quick inspection command:

```bash
uv run python scripts/wandb_project_runs.py \
  --entity rbw \
  --project ColBERT-Zero \
  --name-contains reasonir-mixed-lr-bs2048 \
  --limit 5 \
  --history-tail 5
```

## Full BRIGHT GPT-Trace Eval

The reusable wrapper is ready.

Use it for the completed mixed baseline final checkpoint:

```bash
MODEL_PATH=/home/rbw/repo/pylate/output/reasonir-mixed-bs2048-lr8e5/final \
OUTPUT_JSON=/home/rbw/repo/pylate/output/reasonir-mixed-bs2048-lr8e5-gpt4-full.json \
RUN_NAME=eval-reasonir-mixed-bs2048-lr8e5-gpt4-full \
WANDB_GROUP=reasonir-mixed-gpt4-full \
bash scripts/run_full_bright_gpt4_eval.sh
```

It also supports task-specific configs via `TASK_CONFIGS`, so the stronger machine does not have to use one global batch/chunk setting for every BRIGHT task.

Format:

```text
tasks|query_batch|query_encode_batch|document_batch|corpus_chunk|document_cache_chunk
```

Example multi-config run:

```bash
MODEL_PATH=/home/rbw/repo/pylate/output/reasonir-mixed-bs2048-lr8e5/final \
OUTPUT_JSON=/home/rbw/repo/pylate/output/reasonir-mixed-bs2048-lr8e5-gpt4-full.json \
RUN_NAME=eval-reasonir-mixed-bs2048-lr8e5-gpt4-full \
WANDB_GROUP=reasonir-mixed-gpt4-full \
TASK_CONFIGS='earth_science|32|128|256|256|256;biology,robotics,stackoverflow,aops,theoremqa_questions|64|128|256|512|512;economics,psychology,sustainable_living,leetcode,pony,theoremqa_theorems|64|128|256|1024|1024' \
bash scripts/run_full_bright_gpt4_eval.sh
```

Use it for a fresh base control:

```bash
MODEL_PATH=/mnt/ml_models/lightonai/ColBERT-Zero \
OUTPUT_JSON=/home/rbw/repo/pylate/output/bright-base-colbert-zero-gpt4-full.json \
RUN_NAME=eval-colbert-zero-gpt4-full \
WANDB_GROUP=reasonir-mixed-gpt4-full \
bash scripts/run_full_bright_gpt4_eval.sh
```

The wrapper uses the stable full-eval settings from the earlier BRIGHT work:

- `query_batch_size=32`
- `query_encode_batch_size=32`
- `document_batch_size=256`
- `corpus_chunk_size=256`
- `top_k=1000`
- `cache_dir=/mnt/ml_models/cache/pylate-bright-cache`

For LR sweep winners, swap in the selected output path, for example:

```bash
MODEL_PATH=/home/rbw/repo/pylate/output/reasonir-mixed-lr-bs2048-lr1e-5/final \
OUTPUT_JSON=/home/rbw/repo/pylate/output/reasonir-mixed-lr-bs2048-lr1e-5-gpt4-full.json \
RUN_NAME=eval-reasonir-mixed-lr-bs2048-lr1e-5-gpt4-full \
WANDB_GROUP=reasonir-mixed-gpt4-full \
bash scripts/run_full_bright_gpt4_eval.sh
```

## If The Next Codex Instance Needs To Verify Readiness

Suggested order:

1. Read [`REASONIR_MIXED_RUNBOOK.md`](/home/rbw/repo/pylate/REASONIR_MIXED_RUNBOOK.md)
2. Confirm the staged dataset exists under `/mnt`
3. Confirm [`scripts/reasonir_mixed_lr_sweep.sh`](/home/rbw/repo/pylate/scripts/reasonir_mixed_lr_sweep.sh) is present in that checkout
4. Launch the mixed LR sweep detached on the stronger machine
5. After training, run the reusable full BRIGHT GPT-trace eval wrapper for the sweep winner, the mixed baseline, and base

## Better Than This Note?

This note plus the runbook is enough.

If you want the next Codex instance to be maximally effective, point it to:

- [`codex_handoff_note.md`](/home/rbw/repo/pylate/codex_handoff_note.md)
- [`REASONIR_MIXED_RUNBOOK.md`](/home/rbw/repo/pylate/REASONIR_MIXED_RUNBOOK.md)

That is cleaner than trying to reconstruct this session from chat history.
