# ReasonIR ColBERT-Zero Fine-Tuning

This note captures the local workflow used to fine-tune `lightonai/ColBERT-Zero` on ReasonIR-style synthetic reasoning triplets.

## Goal

Replicate the general structure of [`examples/train/reason_moderncolbert.py`](/home/rbw/repo/pylate/examples/train/reason_moderncolbert.py) for `ColBERT-Zero`, but train from local synthetic data generated in the ReasonIR repository.

The critical constraint is prompt alignment:

- queries must keep the `search_query: ` prefix
- documents and negatives must keep the `search_document: ` prefix

This is required by the base checkpoint and is preserved in [`examples/train/ColBERT-zero/reasonir.py`](/home/rbw/repo/pylate/examples/train/ColBERT-zero/reasonir.py).

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
