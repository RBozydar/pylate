from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import evaluation, losses, models, utils
from pylate.scores import colbert_scores_pairwise

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("/mnt/ml_models/lightonai/ColBERT-Zero")
DEFAULT_DATA_ROOT = Path(
    "/home/rbw/repo/ReasonIR/synthetic_data_generation/synthetic_data"
)
DEFAULT_GENERATOR = "gemini-3-flash-preview"
DEFAULT_PROMPT_ID = "hq_gen"
EXPECTED_QUERY_PROMPT = "search_query: "
EXPECTED_DOCUMENT_PROMPT = "search_document: "


class PromptAlignedTripletEvaluator(evaluation.ColBERTTripletEvaluator):
    """Triplet evaluator that preserves ColBERT-Zero prompt alignment."""

    def __init__(
        self,
        *args,
        query_prompt: str,
        document_prompt: str,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.query_prompt = query_prompt
        self.document_prompt = document_prompt

    def __call__(
        self,
        model: models.ColBERT,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> dict[str, float]:
        LOGGER.info(
            "Evaluating %s at epoch=%s steps=%s with prompt-aligned encoding.",
            self.name or "validation",
            epoch,
            steps,
        )

        embeddings_anchors = model.encode(
            sentences=self.anchors,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=True,
            is_query=True,
            prompt=self.query_prompt,
        )
        embeddings_positives = model.encode(
            sentences=self.positives,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=False,
            is_query=False,
            prompt=self.document_prompt,
        )
        embeddings_negatives = model.encode(
            sentences=self.negatives,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=False,
            is_query=False,
            prompt=self.document_prompt,
        )

        positive_scores = colbert_scores_pairwise(
            queries_embeddings=embeddings_anchors,
            documents_embeddings=embeddings_positives,
        )
        negative_scores = colbert_scores_pairwise(
            queries_embeddings=embeddings_anchors,
            documents_embeddings=embeddings_negatives,
        )
        metrics = {
            "accuracy": (
                sum(positive_scores > negative_scores) / len(positive_scores)
            ).item()
        }

        for metric in self.metrics:
            LOGGER.info("%s: %.4f", metric, metrics[metric])

        self.store_metrics_in_model_card_data(model=model, metrics=metrics)
        return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune ColBERT-Zero on local ReasonIR-style synthetic triplets while "
            "preserving the required search_query:/search_document: prompt prefixes."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path or HF identifier for the ColBERT-Zero checkpoint.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing synthetic_data/{hq,vl}/<prompt-id>/<generator>/final_train_data.jsonl.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="hq,vl",
        help="Comma-separated dataset groups to use. Default: hq,vl",
    )
    parser.add_argument(
        "--prompt-id",
        type=str,
        default=DEFAULT_PROMPT_ID,
        help="Synthetic prompt template directory name.",
    )
    parser.add_argument(
        "--generator",
        type=str,
        default=DEFAULT_GENERATOR,
        help="Synthetic data generator subdirectory name.",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face datasets cache directory for local JSON loading.",
    )
    parser.add_argument(
        "--max-negatives",
        type=int,
        default=1,
        help="Number of mined negatives to keep per example. Default: 1.",
    )
    parser.add_argument(
        "--max-examples-per-split",
        type=int,
        default=None,
        help="Optional cap applied independently to each selected split after shuffling.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.01,
        help="Fraction (<1) or count (>=1) of each split reserved for validation. Set to 0 to disable.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=3.0,
        help="Number of training epochs. Default: 3.0.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Initial learning rate. Default: 1e-5.",
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=128,
        help="Per-device train batch size. Default: 128.",
    )
    parser.add_argument(
        "--eval-bs",
        type=int,
        default=128,
        help="Per-device evaluation batch size. Default: 128.",
    )
    parser.add_argument(
        "--mini-batch-size",
        type=int,
        default=32,
        help="Mini-batch size used inside CachedContrastive. Default: 32.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=1.0,
        help="Temperature for the contrastive loss. Default: 1.0.",
    )
    parser.add_argument(
        "--learnable-temperature",
        action="store_true",
        help="Learn the contrastive temperature instead of keeping it fixed.",
    )
    parser.add_argument(
        "--query-length",
        type=int,
        default=None,
        help="Optional query length override. By default the checkpoint config is used.",
    )
    parser.add_argument(
        "--document-length",
        type=int,
        default=None,
        help="Optional document length override. By default the checkpoint config is used.",
    )
    parser.add_argument(
        "--query-prompt",
        type=str,
        default=None,
        help="Prompt prepended to queries during training. Defaults to the model prompt config.",
    )
    parser.add_argument(
        "--document-prompt",
        type=str,
        default=None,
        help="Prompt prepended to positive and negative documents during training. Defaults to the model prompt config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to output/<model>/<run-name>.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name. Defaults to a name derived from model and dataset selection.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Checkpoint save interval in steps. Default: 500.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Validation interval in steps when validation is enabled. Default: 500.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Logging interval in steps. Default: 10.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to retain. Default: 2.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Dataloader worker count. Default: 8.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 mixed precision.",
    )
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable bf16 mixed precision.",
    )
    parser.add_argument(
        "--gather-across-devices",
        action="store_true",
        help="Gather document embeddings across devices for larger in-batch negatives.",
    )
    return parser.parse_args()


def parse_dataset_names(raw_value: str) -> list[str]:
    dataset_names = [value.strip() for value in raw_value.split(",") if value.strip()]
    if not dataset_names:
        raise ValueError("At least one dataset name must be provided via --datasets.")
    return dataset_names


def resolve_reasonir_file(
    data_root: Path,
    dataset_name: str,
    prompt_id: str,
    generator: str,
) -> Path:
    data_file = (
        data_root.expanduser()
        / dataset_name
        / prompt_id
        / generator
        / "final_train_data.jsonl"
    )
    if not data_file.is_file():
        raise FileNotFoundError(
            f"Could not find ReasonIR synthetic data file: {data_file}"
        )
    return data_file


def validation_size_for_dataset(
    requested_size: float,
    dataset_size: int,
) -> float | int:
    if requested_size <= 0 or dataset_size < 2:
        return 0
    if requested_size < 1:
        max_fraction = (dataset_size - 1) / dataset_size
        return min(requested_size, max_fraction)
    return min(int(requested_size), dataset_size - 1)


def map_reasonir_row(example: dict, max_negatives: int) -> dict[str, str]:
    mapped = {
        "query": example["query"].strip(),
        "document": example["pos"][0].strip(),
    }
    for index, negative in enumerate(example["neg"][:max_negatives]):
        mapped[f"negative_{index}"] = negative.strip()
    return mapped


def load_reasonir_split(
    *,
    data_file: Path,
    dataset_name: str,
    max_negatives: int,
    max_examples: int | None,
    validation_size: float,
    dataset_cache_dir: Path | None,
    seed: int,
) -> tuple[Dataset, Dataset | None]:
    dataset = load_dataset(
        "json",
        data_files=str(data_file),
        split="train",
        cache_dir=(
            str(dataset_cache_dir.expanduser()) if dataset_cache_dir is not None else None
        ),
    )
    dataset = dataset.filter(
        lambda row: (
            bool(row.get("query"))
            and len(row.get("pos") or []) >= 1
            and len(row.get("neg") or []) >= max_negatives
        )
    )
    dataset = dataset.shuffle(seed=seed)

    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    dataset = dataset.map(
        lambda row: map_reasonir_row(example=row, max_negatives=max_negatives),
        remove_columns=dataset.column_names,
    )
    ordered_columns = [
        "query",
        "document",
        *[f"negative_{index}" for index in range(max_negatives)],
    ]
    dataset = dataset.select_columns(ordered_columns)

    eval_size = validation_size_for_dataset(
        requested_size=validation_size,
        dataset_size=len(dataset),
    )
    if not eval_size:
        return dataset, None

    splits = dataset.train_test_split(test_size=eval_size, seed=seed)
    return splits["train"], splits["test"]


def concatenate_or_none(datasets_list: Sequence[Dataset]) -> Dataset | None:
    if not datasets_list:
        return None
    if len(datasets_list) == 1:
        return datasets_list[0]
    return concatenate_datasets(list(datasets_list))


def build_datasets(args: argparse.Namespace) -> tuple[Dataset, Dataset | None, list[str]]:
    selected_splits = parse_dataset_names(args.datasets)
    train_parts: list[Dataset] = []
    eval_parts: list[Dataset] = []

    for dataset_name in selected_splits:
        data_file = resolve_reasonir_file(
            data_root=args.data_root,
            dataset_name=dataset_name,
            prompt_id=args.prompt_id,
            generator=args.generator,
        )
        train_split, eval_split = load_reasonir_split(
            data_file=data_file,
            dataset_name=dataset_name,
            max_negatives=args.max_negatives,
            max_examples=args.max_examples_per_split,
            validation_size=args.validation_size,
            dataset_cache_dir=args.dataset_cache_dir,
            seed=args.seed,
        )
        LOGGER.info(
            "Loaded %s from %s with %d train rows%s.",
            dataset_name,
            data_file,
            len(train_split),
            (
                f" and {len(eval_split)} eval rows"
                if eval_split is not None
                else ""
            ),
        )
        train_parts.append(train_split)
        if eval_split is not None:
            eval_parts.append(eval_split)

    train_dataset = concatenate_or_none(train_parts)
    if train_dataset is None:
        raise ValueError("No training data was loaded.")

    train_dataset = train_dataset.shuffle(seed=args.seed)
    eval_dataset = concatenate_or_none(eval_parts)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.shuffle(seed=args.seed)

    return train_dataset, eval_dataset, selected_splits


def resolve_prompts(
    model: models.ColBERT,
    query_prompt_override: str | None,
    document_prompt_override: str | None,
) -> tuple[str, str]:
    configured_prompts = model.prompts or {}
    query_prompt = query_prompt_override or configured_prompts.get("query")
    document_prompt = document_prompt_override or configured_prompts.get("document")

    if query_prompt is None or document_prompt is None:
        raise ValueError(
            "ColBERT-Zero prompt alignment is required. Provide --query-prompt and "
            "--document-prompt or use a checkpoint that exposes `query` and "
            "`document` prompts in config_sentence_transformers.json."
        )

    if (
        query_prompt != EXPECTED_QUERY_PROMPT
        or document_prompt != EXPECTED_DOCUMENT_PROMPT
    ):
        LOGGER.warning(
            "Using prompts query=%r document=%r. The released ColBERT-Zero checkpoint "
            "expects %r and %r.",
            query_prompt,
            document_prompt,
            EXPECTED_QUERY_PROMPT,
            EXPECTED_DOCUMENT_PROMPT,
        )

    return query_prompt, document_prompt


def build_output_dir(
    *,
    output_dir: Path | None,
    run_name: str | None,
    model_path: Path,
    datasets_used: Sequence[str],
    generator: str,
) -> tuple[str, str]:
    model_short_name = model_path.name if model_path.name else str(model_path)
    resolved_run_name = (
        run_name
        if run_name is not None
        else f"{model_short_name}-ReasonIR-{generator}-{'-'.join(datasets_used)}"
    )
    resolved_output_dir = (
        output_dir
        if output_dir is not None
        else Path("output") / model_short_name / resolved_run_name
    )
    return resolved_run_name, str(resolved_output_dir)


def build_evaluator(
    eval_dataset: Dataset | None,
    eval_batch_size: int,
    query_prompt: str,
    document_prompt: str,
) -> PromptAlignedTripletEvaluator | None:
    if eval_dataset is None or len(eval_dataset) == 0:
        return None
    return PromptAlignedTripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["document"],
        negatives=eval_dataset["negative_0"],
        query_prompt=query_prompt,
        document_prompt=document_prompt,
        name="reasonir-validation",
        batch_size=eval_batch_size,
        show_progress_bar=False,
        write_csv=False,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    train_dataset, eval_dataset, datasets_used = build_datasets(args=args)

    temperature: float | torch.nn.Parameter
    if args.learnable_temperature:
        temperature = torch.nn.Parameter(torch.tensor(args.temp))
    else:
        temperature = args.temp

    model = models.ColBERT(
        model_name_or_path=str(args.model.expanduser()),
        local_files_only=args.model.expanduser().exists(),
        query_length=args.query_length,
        document_length=args.document_length,
    )
    query_prompt, document_prompt = resolve_prompts(
        model=model,
        query_prompt_override=args.query_prompt,
        document_prompt_override=args.document_prompt,
    )
    model.prompts = {
        **(model.prompts or {}),
        "query": query_prompt,
        "document": document_prompt,
    }

    run_name, output_dir = build_output_dir(
        output_dir=args.output_dir,
        run_name=args.run_name,
        model_path=args.model.expanduser(),
        datasets_used=datasets_used,
        generator=args.generator,
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train_loss = losses.CachedContrastive(
        model=model,
        mini_batch_size=args.mini_batch_size,
        gather_across_devices=args.gather_across_devices,
        temperature=temperature,
    )

    evaluator = build_evaluator(
        eval_dataset=eval_dataset,
        eval_batch_size=args.eval_bs,
        query_prompt=query_prompt,
        document_prompt=document_prompt,
    )
    use_bf16 = False if args.fp16 else not args.no_bf16
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.eval_bs,
        eval_strategy="steps" if evaluator is not None else "no",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        fp16=args.fp16,
        bf16=use_bf16,
        run_name=run_name,
        learning_rate=args.lr,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
        seed=args.seed,
    )

    negative_prompts = {
        f"negative_{index}": document_prompt for index in range(args.max_negatives)
    }
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=evaluator,
        data_collator=utils.ColBERTCollator(
            tokenize_fn=model.tokenize,
            prompts={
                "query": query_prompt,
                "document": document_prompt,
                **negative_prompts,
            },
        ),
    )

    trainer.train()
    model.save_pretrained(f"{output_dir}/final")


if __name__ == "__main__":
    main()
