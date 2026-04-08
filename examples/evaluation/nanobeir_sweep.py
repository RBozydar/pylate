from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from datasets import load_dataset

from pylate import evaluation, models
from pylate.evaluation.nano_beir_evaluator import (
    MAPPING_DATASET_NAME_TO_HUMAN_READABLE,
    MAPPING_DATASET_NAME_TO_ID,
)

DEFAULT_DATASETS = [
    "climatefever",
    "dbpedia",
    "fever",
    "fiqa2018",
    "hotpotqa",
    "msmarco",
    "nfcorpus",
    "nq",
    "quoraretrieval",
    "scidocs",
    "arguana",
    "scifact",
    "touche2020",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a prompt-aligned NanoBEIR sweep for a ColBERT checkpoint. "
            "This evaluates each NanoBEIR dataset with "
            "PyLateInformationRetrievalEvaluator and also reports the "
            "NanoBEIR mean metrics."
        )
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Model path or HF identifier to evaluate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Shared query/document encoding batch size.",
    )
    parser.add_argument(
        "--corpus_chunk_size",
        type=int,
        default=2000,
        help="Corpus chunk size used during retrieval scoring.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated NanoBEIR dataset names.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional path to store the full result dictionary as JSON.",
    )
    return parser.parse_args()


def load_nanobeir_dataset(dataset_name: str) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    dataset_path = MAPPING_DATASET_NAME_TO_ID[dataset_name]
    corpus = load_dataset(dataset_path, "corpus", split="train")
    queries = load_dataset(dataset_path, "queries", split="train")
    qrels = load_dataset(dataset_path, "qrels", split="train")

    corpus_dict = {
        sample["_id"]: sample["text"]
        for sample in corpus
        if len(sample["text"]) > 0
    }
    queries_dict = {
        sample["_id"]: sample["text"]
        for sample in queries
        if len(sample["text"]) > 0
    }
    qrels_dict: dict[str, set[str]] = {}
    for sample in qrels:
        qrels_dict.setdefault(sample["query-id"], set()).add(sample["corpus-id"])

    return corpus_dict, queries_dict, qrels_dict


def to_jsonable(results: dict[str, object]) -> dict[str, float]:
    return {key: float(value) for key, value in results.items()}


def main() -> None:
    args = parse_args()
    dataset_names = [item.strip() for item in args.datasets.split(",") if item.strip()]

    model_path = Path(args.model_name_or_path)
    model = models.ColBERT(
        model_name_or_path=args.model_name_or_path,
        local_files_only=model_path.exists(),
    )
    prompts = model.prompts or {}
    query_prompt = prompts.get("query")
    document_prompt = prompts.get("document")

    all_results: dict[str, float] = {}
    per_metric: dict[str, list[float]] = {}

    for dataset_name in dataset_names:
        corpus, queries, relevant_docs = load_nanobeir_dataset(dataset_name=dataset_name)
        evaluator = evaluation.PyLateInformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            corpus_chunk_size=args.corpus_chunk_size,
            mrr_at_k=[10],
            ndcg_at_k=[10],
            accuracy_at_k=[1, 3, 5, 10],
            precision_recall_at_k=[1, 3, 5, 10],
            map_at_k=[100],
            show_progress_bar=True,
            batch_size=args.batch_size,
            write_csv=False,
            name=f"Nano{MAPPING_DATASET_NAME_TO_HUMAN_READABLE[dataset_name]}",
            query_prompt=query_prompt,
            corpus_prompt=document_prompt,
        )
        results = to_jsonable(results=evaluator(model))
        all_results.update(results)

        dataset_prefix = f"Nano{MAPPING_DATASET_NAME_TO_HUMAN_READABLE[dataset_name]}_"
        for key, value in results.items():
            metric_name = key.removeprefix(dataset_prefix)
            per_metric.setdefault(metric_name, []).append(value)

    for metric_name, values in per_metric.items():
        all_results[f"NanoBEIR_mean_{metric_name}"] = float(mean(values))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(all_results, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(all_results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
