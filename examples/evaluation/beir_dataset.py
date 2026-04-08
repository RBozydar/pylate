"""Evaluation script for BEIR datasets with a PLAID index."""

from __future__ import annotations

import argparse
from pathlib import Path

from pylate import evaluation, indexes, models, retrieve


def resolve_prompt_name(model: models.ColBERT, role: str) -> str | None:
    """Return the prompt name for prompt-sensitive checkpoints when available."""
    prompts = model.prompts or {}
    return role if role in prompts else None


if __name__ == "__main__":
    query_len = {
        "quora": 32,
        "climate-fever": 64,
        "nq": 32,
        "msmarco": 32,
        "hotpotqa": 32,
        "nfcorpus": 32,
        "scifact": 48,
        "trec-covid": 48,
        "fiqa": 32,
        "arguana": 64,
        "scidocs": 48,
        "dbpedia-entity": 32,
        "webis-touche2020": 32,
        "fever": 32,
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
    }

    # Parse dataset_name from command line arguments
    parser = argparse.ArgumentParser(description="Dataset name")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="nfcorpus",
        help="Name of the dataset to evaluate on (default: 'fiqa')",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="lightonai/GTE-ModernColBERT-v1",
        help="Model path or HF identifier to evaluate.",
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
        help="Optional query length override.",
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
    args = parser.parse_args()
    dataset_name = args.dataset_name
    model_name = args.model_name_or_path
    model = models.ColBERT(
        model_name_or_path=model_name,
        document_length=args.document_length,
        query_length=args.query_length or query_len.get(dataset_name),
        local_files_only=Path(model_name).exists(),
    )
    query_prompt_name = resolve_prompt_name(model=model, role="query")
    document_prompt_name = resolve_prompt_name(model=model, role="document")

    if "cqadupstack" in dataset_name:
        # Download dataset if not already downloaded
        from beir import util

        data_path = util.download_and_unzip(
            url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip",
            out_dir="./evaluation_datasets/",
        )
        documents, queries, qrels = evaluation.load_custom_dataset(
            f"evaluation_datasets/{dataset_name}",
            split="test",
        )
        dataset_name = dataset_name.replace("/", "_")
    else:
        documents, queries, qrels = evaluation.load_beir(
            dataset_name=dataset_name,
            split="dev" if "msmarco" in dataset_name else "test",
        )

    index = indexes.PLAID(
        override=True,
        index_name=f"{dataset_name}_{model_name.split('/')[-1]}",
    )

    retriever = retrieve.ColBERT(index=index)

    documents_embeddings = model.encode(
        sentences=[document["text"] for document in documents],
        batch_size=args.document_batch_size,
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
        is_query=True,
        show_progress_bar=True,
        batch_size=args.query_batch_size,
        prompt_name=query_prompt_name,
    )

    scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=args.k)

    # Remove query_id from scores, needed for FiQA dataset
    for (query_id, query), query_scores in zip(queries.items(), scores):
        for score in query_scores:
            if score["id"] == query_id:
                # Remove the query_id from the score
                query_scores.remove(score)

    evaluation_scores = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=list(queries.keys()),
        # queries=queries,
        metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
    )

    print(evaluation_scores)
