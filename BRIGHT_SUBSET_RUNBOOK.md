# BRIGHT Subset Runbook

This file captures the raw-query BRIGHT subset evaluation workflows used for:

- the local `ColBERT-Zero` base checkpoint
- the local fine-tuned `ReasonIR` checkpoint
- the official ReasonIR HQ pilot sweeps and checkpoint comparisons

## Scope

This runbook covers the 6-task subset:

- `biology`
- `earth_science`
- `economics`
- `psychology`
- `robotics`
- `pony`

Evaluation mode:

- BRIGHT raw queries
- exact MaxSim via `examples/evaluation/bright_reasonir.py`
- prompt-aware for `ColBERT-Zero`

## Model Paths

- Base checkpoint: `/mnt/ml_models/lightonai/ColBERT-Zero`
- Fine-tuned checkpoint: `/home/rbw/repo/pylate/output/reasonir-full-gpu/final`
- HQ pilot sweep outputs: `/home/rbw/repo/pylate/output/hq-*`

## Output Files

- Base subset result:
  - [`output/bright-base-colbert-zero-raw-partial.json`](/home/rbw/repo/pylate/output/bright-base-colbert-zero-raw-partial.json)
- Fine-tuned subset result:
  - [`output/bright-reasonir-full-gpu-raw-partial.json`](/home/rbw/repo/pylate/output/bright-reasonir-full-gpu-raw-partial.json)

The filenames still say `partial`, but both subset runs completed successfully.

For the official HQ pilot sweeps:

- final-model subset evals are written as:
  - `/home/rbw/repo/pylate/output/<run-name>-bright-subset.json`
- checkpoint subset evals are written as:
  - `/home/rbw/repo/pylate/output/<run-name>-checkpoint-<step>-bright-subset.json`
  - `/home/rbw/repo/pylate/output/<run-name>-final-bright-subset.json`

## Commands

Run from the repo root.

### Base `ColBERT-Zero`

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 uv run python examples/evaluation/bright_reasonir.py \
  --model_name_or_path /mnt/ml_models/lightonai/ColBERT-Zero \
  --tasks biology,earth_science,economics,psychology,robotics,pony \
  --reasoning none \
  --query_batch_size 16 \
  --query_encode_batch_size 32 \
  --document_batch_size 128 \
  --corpus_chunk_size 1024 \
  --top_k 1000 \
  --cache_dir /tmp/pylate-bright-cache \
  --output_json /home/rbw/repo/pylate/output/bright-base-colbert-zero-raw-partial.json
```

### Fine-tuned `ReasonIR` checkpoint

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 uv run python examples/evaluation/bright_reasonir.py \
  --model_name_or_path /home/rbw/repo/pylate/output/reasonir-full-gpu/final \
  --tasks biology,earth_science,economics,psychology,robotics,pony \
  --reasoning none \
  --query_batch_size 16 \
  --query_encode_batch_size 32 \
  --document_batch_size 128 \
  --corpus_chunk_size 1024 \
  --top_k 1000 \
  --cache_dir /tmp/pylate-bright-cache \
  --output_json /home/rbw/repo/pylate/output/bright-reasonir-full-gpu-raw-partial.json
```

### HQ pilot sweep outputs

Use the unified wrapper to evaluate retained `checkpoint-*` directories plus `final` for a given sweep stage:

```bash
STAGE=batch INCLUDE_CHECKPOINTS=1 ./scripts/reasonir_hq_bright_subset_eval.sh
```

By default this evaluates the 4-task subset:

- `biology`
- `economics`
- `robotics`
- `pony`

and logs evals to W&B with:

- project: `ColBERT-Zero`
- entity: `rbw`
- group: `reasonir-hq-bright-checkpoints`

To restrict evaluation to a single training run:

```bash
INCLUDE_CHECKPOINTS=1 ./scripts/reasonir_hq_bright_subset_eval.sh hq-batch-bs256-lr1e5-temp1
```

If you want finals only for a stage, omit `INCLUDE_CHECKPOINTS=1`:

```bash
STAGE=batch ./scripts/reasonir_hq_bright_subset_eval.sh
```

Disable eval logging with:

```bash
REPORT_TO=none ./scripts/reasonir_hq_bright_subset_eval.sh
```

The wrapper scripts do not force Hugging Face offline mode. They will fetch missing BRIGHT data into cache when needed. For cached-only reruns, prefix either wrapper with:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CLEANUP_DOCUMENT_CACHE=0
```

By default the wrapper scripts clean up the model-specific BRIGHT document cache after each eval run. That keeps one-off sweep comparisons from leaving behind tens of GiB of document shards per model. Set `CLEANUP_DOCUMENT_CACHE=0` if you want to retain those caches for reuse.

## Resume Behavior

The runner is resumable.

- If `--output_json` already exists, completed tasks are skipped.
- Example: the base rerun skipped `biology` and resumed with `earth_science`.

This behavior is implemented in:

- [`examples/evaluation/bright_reasonir.py`](/home/rbw/repo/pylate/examples/evaluation/bright_reasonir.py)

Wrapper script:

- [`scripts/reasonir_hq_bright_subset_eval.sh`](/home/rbw/repo/pylate/scripts/reasonir_hq_bright_subset_eval.sh)

## Cache Behavior

The BRIGHT runner stores per-model, per-task document embedding shards under:

- `/tmp/pylate-bright-cache/bright_doc_emb`

Important implications:

- the base and fine-tuned checkpoints do **not** share document embedding shards
- the first pass for a new model is slower because it builds its own cache
- reruns become much faster once the cache exists

## Operational Notes

### GPU

These runs were executed on the host GPU, not in the sandbox. The sandboxed environment did not have working NVIDIA driver access.

### Current HQ pilot cadence

The current HQ pilot sweep training setup keeps checkpoint analysis viable:

- `validation_size=0.01`
- `save_steps=25`
- `eval_steps=25`
- `save_total_limit=20`

That means BRIGHT subset evals can be run both on retained `checkpoint-*` directories and on `final` in a single pass.

### W&B

The BRIGHT evaluator can log eval metrics to W&B directly with:

- `--report-to wandb`
- `--wandb-project ColBERT-Zero`
- `--wandb-entity rbw`
- `--cleanup-document-cache` to remove model-specific doc shards after a direct evaluator run

Per-task BRIGHT metrics and the final summary are logged for each eval run.

### Disk usage

The BRIGHT cache is large. At one point:

- `/tmp/pylate-bright-cache` was about `107G`

The fine-tuned run initially failed because the filesystem filled up during a `torch.save(...)` call for a document embedding shard.

Observed failure mode:

- `RuntimeError: PytorchStreamWriter failed writing file data/0: file write failed`
- followed by a corrupted `.pt` cache shard

### Recovery from corrupted shard

When the disk filled, the fine-tuned run left behind a corrupt cache file:

- `/tmp/pylate-bright-cache/bright_doc_emb/home_rbw_repo_pylate_output_reasonir-full-gpu_final/earth_science/doclen_519/chunk_1024_bs_128/11264.pt`

After freeing disk space, deleting that single shard was sufficient:

```bash
rm /tmp/pylate-bright-cache/bright_doc_emb/home_rbw_repo_pylate_output_reasonir-full-gpu_final/earth_science/doclen_519/chunk_1024_bs_128/11264.pt
```

Then rerunning the same command resumed from the JSON output:

- `biology` was skipped
- `earth_science` was recomputed successfully
- the remaining tasks completed

## Results

### Base subset

Mean `NDCG@10`: `16.97`

- `biology`: `16.20`
- `earth_science`: `21.73`
- `economics`: `15.38`
- `psychology`: `18.01`
- `robotics`: `15.31`
- `pony`: `15.20`

### Fine-tuned subset

Mean `NDCG@10`: `5.21`

- `biology`: `5.77`
- `earth_science`: `6.72`
- `economics`: `4.03`
- `psychology`: `5.65`
- `robotics`: `4.34`
- `pony`: `4.74`

## Conclusion

On this raw-query 6-task BRIGHT subset, the fine-tuned `ReasonIR` checkpoint is substantially worse than the base `ColBERT-Zero` checkpoint.

No task improved in this subset comparison.

Given this result, expanding to the full 12-task raw-query BRIGHT run or the `gpt4` reasoning setup is not recommended unless there is a specific reason to do so.
