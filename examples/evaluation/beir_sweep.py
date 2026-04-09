"""Run a prompt-aware BEIR sweep for a ColBERT checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path
from statistics import mean

import torch

from pylate import evaluation, indexes, models, retrieve

QUERY_LENGTHS = {
    "arguana": 64,
    "climate-fever": 64,
    "cqadupstack/android": 32,
    "cqadupstack/english": 32,
    "cqadupstack/gaming": 32,
    "cqadupstack/gis": 32,
    "cqadupstack/mathematica": 32,
    "cqadupstack/physics": 32,
    "cqadupstack/programmers": 32,
    "cqadupstack/stats": 32,
    "cqadupstack/tex": 32,
    "cqadupstack/unix": 32,
    "cqadupstack/webmasters": 32,
    "cqadupstack/wordpress": 32,
    "dbpedia-entity": 32,
    "fever": 32,
    "fiqa": 32,
    "hotpotqa": 32,
    "msmarco": 32,
    "nfcorpus": 32,
    "nq": 32,
    "quora": 32,
    "scidocs": 48,
    "scifact": 48,
    "trec-covid": 48,
    "webis-touche2020": 32,
}

CQADUPSTACK_SUBDATASETS = (
    "cqadupstack/android",
    "cqadupstack/english",
    "cqadupstack/gaming",
    "cqadupstack/gis",
    "cqadupstack/mathematica",
    "cqadupstack/physics",
    "cqadupstack/programmers",
    "cqadupstack/stats",
    "cqadupstack/tex",
    "cqadupstack/unix",
    "cqadupstack/webmasters",
    "cqadupstack/wordpress",
)

MODEL_CARD_DATASETS = (
    "fiqa",
    "nfcorpus",
    "trec-covid",
    "webis-touche2020",
    "arguana",
    "quora",
    "scidocs",
    "scifact",
    "nq",
    "climate-fever",
    "hotpotqa",
    "dbpedia-entity",
    "cqadupstack",
    "fever",
    "msmarco",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the prompt-aware ColBERT BEIR sweep used by the model-card "
            "style evaluation table."
        )
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Model path or HF identifier to evaluate.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(MODEL_CARD_DATASETS),
        help=(
            "Comma-separated dataset names. Use 'cqadupstack' to evaluate all "
            "12 CQADupstack subsets and report their average."
        ),
    )
    parser.add_argument(
        "--document_length",
        type=int,
        default=None,
        help="Optional document length override.",
    )
    parser.add_argument(
        "--query_length",
        type=int,
        default=None,
        help=(
            "Optional query length override. If omitted, the script uses the "
            "same per-dataset lengths as the existing single-dataset runner."
        ),
    )
    parser.add_argument(
        "--document_batch_size",
        type=int,
        default=128,
        help="Document encoding batch size.",
    )
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=32,
        help="Query encoding batch size.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Top-k documents to retrieve for evaluation.",
    )
    parser.add_argument(
        "--index_folder",
        type=Path,
        default=Path("/tmp/pylate-beir-indexes"),
        help="Folder used for the reusable PLAID index.",
    )
    parser.add_argument(
        "--index_name",
        type=str,
        default=None,
        help="Optional PLAID index name. Defaults to a sanitized model-based name.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional resumable JSON output path.",
    )
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    """Return a filesystem-safe identifier."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "beir_eval"


def resolve_prompt_name(model: models.ColBERT, role: str) -> str | None:
    """Return the prompt name for prompt-sensitive checkpoints when available."""
    prompts = model.prompts or {}
    return role if role in prompts else None


def expand_dataset_names(dataset_names: list[str]) -> list[str]:
    """Expand logical dataset groups into concrete datasets."""
    expanded: list[str] = []
    for dataset_name in dataset_names:
        normalized = dataset_name.strip()
        if not normalized:
            continue
        if normalized == "cqadupstack":
            expanded.extend(CQADUPSTACK_SUBDATASETS)
            continue
        expanded.append(normalized)
    return expanded


def load_dataset(dataset_name: str) -> tuple[list[dict[str, str]], dict[str, str], dict]:
    """Load a BEIR dataset or CQADupstack subset."""
    if dataset_name.startswith("cqadupstack/"):
        from beir import util

        util.download_and_unzip(
            url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip",
            out_dir="./evaluation_datasets/",
        )
        return evaluation.load_custom_dataset(
            path=f"evaluation_datasets/{dataset_name}",
            split="test",
        )

    split = "dev" if dataset_name == "msmarco" else "test"
    return evaluation.load_beir(dataset_name=dataset_name, split=split)


def remove_query_self_hits(
    queries: dict[str, str],
    scores: list[list[dict[str, float]]],
) -> None:
    """Remove accidental query-id self matches from the score lists."""
    for query_id, query_scores in zip(queries.keys(), scores):
        filtered_scores = [
            score for score in query_scores if str(score["id"]) != str(query_id)
        ]
        query_scores[:] = filtered_scores


def load_results(output_json: Path | None) -> dict[str, object]:
    """Load resumable output if present, otherwise initialize an empty structure."""
    if output_json is None or not output_json.exists():
        return {"datasets": {}, "aggregates": {}}

    loaded = json.loads(output_json.read_text(encoding="utf-8"))
    loaded.setdefault("datasets", {})
    loaded.setdefault("aggregates", {})
    return loaded


def save_results(output_json: Path | None, results: dict[str, object]) -> None:
    """Persist the resumable output JSON."""
    if output_json is None:
        return

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def update_aggregates(results: dict[str, object]) -> None:
    """Refresh CQADupstack and BEIR mean aggregates from the raw dataset scores."""
    datasets = results["datasets"]
    aggregates = results["aggregates"]

    cqadupstack_values = [
        datasets[subdataset]
        for subdataset in CQADUPSTACK_SUBDATASETS
        if subdataset in datasets
    ]
    if len(cqadupstack_values) == len(CQADUPSTACK_SUBDATASETS):
        aggregates["cqadupstack"] = float(mean(cqadupstack_values))
    else:
        aggregates.pop("cqadupstack", None)

    model_card_values: list[float] = []
    for dataset_name in MODEL_CARD_DATASETS:
        if dataset_name == "cqadupstack":
            if "cqadupstack" not in aggregates:
                aggregates.pop("beir_mean_ndcg@10", None)
                return
            model_card_values.append(aggregates["cqadupstack"])
            continue

        if dataset_name not in datasets:
            aggregates.pop("beir_mean_ndcg@10", None)
            return
        model_card_values.append(datasets[dataset_name])

    aggregates["beir_mean_ndcg@10"] = float(mean(model_card_values))


def evaluate_dataset(
    *,
    dataset_name: str,
    model_name_or_path: str,
    document_length: int | None,
    query_length: int | None,
    document_batch_size: int,
    query_batch_size: int,
    k: int,
    index_folder: Path,
    index_name: str,
) -> float:
    """Evaluate one BEIR dataset and return NDCG@10 in percentage points."""
    model_path = Path(model_name_or_path)
    model = models.ColBERT(
        model_name_or_path=model_name_or_path,
        document_length=document_length,
        query_length=query_length,
        local_files_only=model_path.exists(),
    )
    query_prompt_name = resolve_prompt_name(model=model, role="query")
    document_prompt_name = resolve_prompt_name(model=model, role="document")

    documents, queries, qrels = load_dataset(dataset_name=dataset_name)
    index = indexes.PLAID(
        index_folder=str(index_folder),
        index_name=index_name,
        override=True,
    )
    retriever = retrieve.ColBERT(index=index)

    documents_embeddings = model.encode(
        sentences=[document["text"] for document in documents],
        batch_size=document_batch_size,
        is_query=False,
        show_progress_bar=True,
        prompt_name=document_prompt_name,
    )
    index.add_documents(
        documents_ids=[document["id"] for document in documents],
        documents_embeddings=documents_embeddings,
    )

    queries_embeddings = model.encode(
        sentences=list(queries.values()),
        batch_size=query_batch_size,
        is_query=True,
        show_progress_bar=True,
        prompt_name=query_prompt_name,
    )
    scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=k)
    remove_query_self_hits(queries=queries, scores=scores)

    metrics = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=list(queries.keys()),
        metrics=["ndcg@10"],
    )

    del scores
    del queries_embeddings
    del documents_embeddings
    del retriever
    del index
    del qrels
    del queries
    del documents
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if isinstance(metrics, dict):
        ndcg_at_10 = float(metrics["ndcg@10"])
    else:
        ndcg_at_10 = float(metrics)

    return ndcg_at_10 * 100.0


def main() -> None:
    """Run the requested BEIR sweep."""
    args = parse_args()
    dataset_names = expand_dataset_names(
        dataset_names=[item.strip() for item in args.datasets.split(",")]
    )
    results = load_results(output_json=args.output_json)
    results["model_name_or_path"] = args.model_name_or_path
    results["k"] = args.k
    results["document_batch_size"] = args.document_batch_size
    results["query_batch_size"] = args.query_batch_size
    results["index_folder"] = str(args.index_folder)
    results["requested_datasets"] = dataset_names

    index_name = args.index_name or f"beir_{sanitize_name(args.model_name_or_path)}"
    args.index_folder.mkdir(parents=True, exist_ok=True)

    for dataset_name in dataset_names:
        if dataset_name in results["datasets"]:
            print(f"Skipping completed dataset: {dataset_name}")
            continue

        effective_query_length = args.query_length or QUERY_LENGTHS.get(dataset_name)
        print(
            f"Evaluating {dataset_name} "
            f"(query_length={effective_query_length}, k={args.k})"
        )
        ndcg_at_10 = evaluate_dataset(
            dataset_name=dataset_name,
            model_name_or_path=args.model_name_or_path,
            document_length=args.document_length,
            query_length=effective_query_length,
            document_batch_size=args.document_batch_size,
            query_batch_size=args.query_batch_size,
            k=args.k,
            index_folder=args.index_folder,
            index_name=index_name,
        )
        results["datasets"][dataset_name] = ndcg_at_10
        update_aggregates(results=results)
        save_results(output_json=args.output_json, results=results)
        print(f"{dataset_name}: ndcg@10={ndcg_at_10:.2f}")

    update_aggregates(results=results)
    save_results(output_json=args.output_json, results=results)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
