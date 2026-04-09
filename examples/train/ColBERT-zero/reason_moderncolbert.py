from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Sequence

import torch
from datasets import Dataset, DatasetDict, load_dataset
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

from pylate import evaluation, losses, models, utils
from pylate.scores import colbert_scores_pairwise

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("/mnt/ml_models/lightonai/ColBERT-Zero")
DEFAULT_REASONIR_DATASET = "reasonir/reasonir-data"
DEFAULT_REASONIR_CONFIG = "hq"
DEFAULT_REASONIR_SPLIT = "train"
DEFAULT_BRIGHT_DATASET = "xlangai/BRIGHT"
DEFAULT_BRIGHT_CONFIG = "documents"
DEFAULT_WANDB_PROJECT = "ColBERT-Zero"
DEFAULT_WANDB_ENTITY = "rbw"
EXPECTED_QUERY_PROMPT = "search_query: "
EXPECTED_DOCUMENT_PROMPT = "search_document: "


class PromptAlignedTripletEvaluator(evaluation.ColBERTTripletEvaluator):
    """Triplet evaluator that preserves ColBERT-Zero prompt alignment."""

    def __init__(
        self,
        *args: Any,
        query_prompt: str,
        document_prompt: str,
        **kwargs: Any,
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
        del output_path
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
            "Fine-tune ColBERT-Zero on the official ReasonIR HQ dataset using the "
            "same BRIGHT document reconstruction described by the dataset card."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path or HF identifier for the ColBERT-Zero checkpoint.",
    )
    parser.add_argument(
        "--reasonir-dataset",
        type=str,
        default=DEFAULT_REASONIR_DATASET,
        help="Hugging Face dataset id for ReasonIR.",
    )
    parser.add_argument(
        "--reasonir-config",
        type=str,
        default=DEFAULT_REASONIR_CONFIG,
        help="ReasonIR dataset config to train on. Default: hq.",
    )
    parser.add_argument(
        "--reasonir-split",
        type=str,
        default=DEFAULT_REASONIR_SPLIT,
        help="ReasonIR dataset split. Default: train.",
    )
    parser.add_argument(
        "--bright-dataset",
        type=str,
        default=DEFAULT_BRIGHT_DATASET,
        help="Hugging Face dataset id for BRIGHT documents.",
    )
    parser.add_argument(
        "--bright-config",
        type=str,
        default=DEFAULT_BRIGHT_CONFIG,
        help="BRIGHT dataset config containing the documents. Default: documents.",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face datasets cache directory.",
    )
    parser.add_argument(
        "--max-negatives",
        type=int,
        default=1,
        help="Number of negative documents to keep per training example. Default: 1.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on the number of HQ examples after shuffling.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.01,
        help="Fraction (<1) or count (>=1) reserved for validation. Set to 0 to disable.",
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
        help="Prompt prepended to positives and negatives during training. Defaults to the model prompt config.",
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
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Override epochs with a fixed number of optimization steps. Default: -1.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.0,
        help="Learning-rate warmup ratio. Default: 0.0.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=DEFAULT_WANDB_PROJECT,
        help="Weights & Biases project name. Default: ColBERT-Zero.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=DEFAULT_WANDB_ENTITY,
        help="Weights & Biases entity name. Default: rbw.",
    )
    parser.add_argument(
        "--report-to",
        type=str,
        default="wandb",
        help="Comma-separated trainer integrations. Use 'none' to disable. Default: wandb.",
    )
    return parser.parse_args()


def dataset_cache_dir(cache_dir: Path | None) -> str | None:
    """Return an expanded Hugging Face cache dir string."""

    if cache_dir is None:
        return None
    return str(cache_dir.expanduser())


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


def configure_wandb(
    *,
    project: str | None,
    entity: str | None,
) -> None:
    """Configure W&B defaults for this training run."""

    if project:
        os.environ["WANDB_PROJECT"] = project
    if entity:
        os.environ["WANDB_ENTITY"] = entity


def resolve_report_to(raw_value: str) -> list[str]:
    """Parse report_to integrations from a comma-separated CLI value."""

    values = [value.strip() for value in raw_value.split(",") if value.strip()]
    if not values or values == ["none"]:
        return []
    return values


def join_text(parts: Sequence[str]) -> str:
    """Join non-empty text fragments with normalized whitespace."""

    return " ".join(part.strip() for part in parts if part and part.strip())


def normalize_query(raw_query: str | Sequence[str]) -> str:
    """Normalize HQ query payloads into a single string."""

    if isinstance(raw_query, str):
        return raw_query.strip()
    return join_text([str(part) for part in raw_query])


def get_doc_and_ids(doc_pairs: Sequence[dict[str, Any]]) -> tuple[list[str], list[str]]:
    """Extract BRIGHT document texts and ids from a task split."""

    doc_ids: list[str] = []
    documents: list[str] = []
    for doc_pair in doc_pairs:
        doc_ids.append(str(doc_pair["id"]))
        documents.append(str(doc_pair["content"]))
    return documents, doc_ids


def build_bright_document_lookup(bright_documents: DatasetDict) -> dict[str, str]:
    """Build a global BRIGHT document-id to document-text mapping."""

    id_to_document: dict[str, str] = {}
    collision_count = 0
    for task_name in bright_documents.keys():
        documents, document_ids = get_doc_and_ids(bright_documents[task_name])
        for document_text, document_id in zip(documents, document_ids, strict=True):
            existing = id_to_document.get(document_id)
            if existing is not None and existing != document_text:
                collision_count += 1
                raise ValueError(
                    "Found conflicting BRIGHT documents for id "
                    f"{document_id!r} while processing task {task_name!r}."
                )
            id_to_document[document_id] = document_text

    LOGGER.info(
        "Built BRIGHT document lookup with %d unique documents.",
        len(id_to_document),
    )
    if collision_count:
        LOGGER.warning("Detected %d conflicting BRIGHT document ids.", collision_count)
    return id_to_document


def row_has_required_fields(
    example: dict[str, Any],
    *,
    max_negatives: int,
    id_to_document: dict[str, str],
) -> bool:
    """Validate that an HQ row can be converted into a training triplet."""

    positives = example.get("pos") or []
    negatives = example.get("neg") or []
    if not normalize_query(example.get("query") or []):
        return False
    if len(positives) < 1 or len(negatives) < max_negatives:
        return False

    positive = positives[0]
    if len(positive) < 2:
        return False
    if str(positive[1]) not in id_to_document:
        return False

    for negative in negatives[:max_negatives]:
        if len(negative) < 2 or not join_text([str(negative[0]), str(negative[1])]):
            return False

    return True


def map_hq_row(
    example: dict[str, Any],
    *,
    max_negatives: int,
    id_to_document: dict[str, str],
) -> dict[str, str]:
    """Convert an HQ example into query/document/negative columns."""

    positives = example["pos"]
    negatives = example["neg"]
    positive_instruction, positive_id = positives[0][0], positives[0][1]

    mapped = {
        "query": normalize_query(example["query"]),
        "document": join_text(
            [str(positive_instruction), id_to_document[str(positive_id)]]
        ),
    }
    for index, negative in enumerate(negatives[:max_negatives]):
        mapped[f"negative_{index}"] = join_text(
            [str(negative[0]), str(negative[1])]
        )
    return mapped


def load_hq_dataset(args: argparse.Namespace) -> tuple[Dataset, Dataset | None]:
    """Load and transform the official ReasonIR HQ training set."""

    cache_dir = dataset_cache_dir(args.dataset_cache_dir)
    hq_dataset = load_dataset(
        args.reasonir_dataset,
        args.reasonir_config,
        split=args.reasonir_split,
        cache_dir=cache_dir,
    )
    bright_documents = load_dataset(
        args.bright_dataset,
        args.bright_config,
        cache_dir=cache_dir,
    )
    id_to_document = build_bright_document_lookup(bright_documents=bright_documents)

    original_size = len(hq_dataset)
    filtered_dataset = hq_dataset.filter(
        lambda row: row_has_required_fields(
            row,
            max_negatives=args.max_negatives,
            id_to_document=id_to_document,
        )
    )
    LOGGER.info(
        "Loaded %d/%d HQ examples after filtering rows without resolvable positives or enough negatives.",
        len(filtered_dataset),
        original_size,
    )
    if len(filtered_dataset) == 0:
        raise ValueError("No HQ training rows remain after filtering.")

    filtered_dataset = filtered_dataset.shuffle(seed=args.seed)
    if args.max_examples is not None:
        filtered_dataset = filtered_dataset.select(
            range(min(args.max_examples, len(filtered_dataset)))
        )

    mapped_dataset = filtered_dataset.map(
        lambda row: map_hq_row(
            row,
            max_negatives=args.max_negatives,
            id_to_document=id_to_document,
        ),
        remove_columns=filtered_dataset.column_names,
    )
    ordered_columns = [
        "query",
        "document",
        *[f"negative_{index}" for index in range(args.max_negatives)],
    ]
    mapped_dataset = mapped_dataset.select_columns(ordered_columns)

    eval_size = validation_size_for_dataset(
        requested_size=args.validation_size,
        dataset_size=len(mapped_dataset),
    )
    if not eval_size:
        return mapped_dataset, None

    splits = mapped_dataset.train_test_split(test_size=eval_size, seed=args.seed)
    return splits["train"], splits["test"]


def resolve_prompts(
    model: models.ColBERT,
    query_prompt_override: str | None,
    document_prompt_override: str | None,
) -> tuple[str, str]:
    """Resolve required ColBERT-Zero prompts from overrides or checkpoint config."""

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
    reasonir_config: str,
) -> tuple[str, str]:
    """Resolve run metadata and output directory."""

    model_short_name = model_path.name if model_path.name else str(model_path)
    resolved_run_name = (
        run_name
        if run_name is not None
        else f"{model_short_name}-ReasonIR-{reasonir_config}"
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
    """Build a prompt-aware triplet evaluator when validation is enabled."""

    if eval_dataset is None or len(eval_dataset) == 0:
        return None
    return PromptAlignedTripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["document"],
        negatives=eval_dataset["negative_0"],
        query_prompt=query_prompt,
        document_prompt=document_prompt,
        name="reasonir-hq-validation",
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
    configure_wandb(project=args.wandb_project, entity=args.wandb_entity)
    train_dataset, eval_dataset = load_hq_dataset(args=args)

    temperature: float | torch.nn.Parameter
    if args.learnable_temperature:
        temperature = torch.nn.Parameter(torch.tensor(args.temp))
    else:
        temperature = args.temp

    resolved_model_path = args.model.expanduser()
    model = models.ColBERT(
        model_name_or_path=str(resolved_model_path),
        local_files_only=resolved_model_path.exists(),
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
        model_path=resolved_model_path,
        reasonir_config=args.reasonir_config,
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
        max_steps=args.max_steps,
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
        warmup_ratio=args.warmup_ratio,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        ddp_find_unused_parameters=False,
        report_to=resolve_report_to(args.report_to),
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
