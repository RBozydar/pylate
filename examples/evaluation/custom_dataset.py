"""Evaluation script for the custom dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from pylate import evaluation, indexes, models, retrieve


def resolve_prompt_name(model: models.ColBERT, role: str) -> str | None:
    """Return the prompt name for prompt-sensitive checkpoints when available."""
    prompts = model.prompts or {}
    return role if role in prompts else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on a custom dataset.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="lightonai/GTE-ModernColBERT-v1",
        help="Model path or HF identifier to evaluate.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/miracl_fr",
        help="Path to the custom dataset directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="Dataset split to evaluate.",
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
        "--k",
        type=int,
        default=100,
        help="Top-k documents to retrieve for evaluation.",
    )
    parser.add_argument(
        "--document_batch_size",
        type=int,
        default=32,
        help="Document encoding batch size.",
    )
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=32,
        help="Query encoding batch size.",
    )
    args = parser.parse_args()

    model = models.ColBERT(
        model_name_or_path=args.model_name_or_path,
        document_length=args.document_length,
        query_length=args.query_length,
        local_files_only=Path(args.model_name_or_path).exists(),
    )
    query_prompt_name = resolve_prompt_name(model=model, role="query")
    document_prompt_name = resolve_prompt_name(model=model, role="document")

    index = indexes.PLAID(override=True)
    retriever = retrieve.ColBERT(index=index)

    documents, queries, qrels = evaluation.load_custom_dataset(
        args.dataset_path,
        split=args.split,
    )

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
        batch_size=args.query_batch_size,
        is_query=True,
        show_progress_bar=True,
        prompt_name=query_prompt_name,
    )

    scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=args.k)

    evaluation_scores = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=list(queries.keys()),
        metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
    )

    print(evaluation_scores)
