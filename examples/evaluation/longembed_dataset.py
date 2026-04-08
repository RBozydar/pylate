"""Evaluation script for the LongEmbed task using the MTEB library."""

from __future__ import annotations

import argparse
from pathlib import Path

import mteb

from pylate import evaluation, indexes, models, retrieve


def resolve_prompt_name(model: models.ColBERT, role: str) -> str | None:
    """Return the prompt name for prompt-sensitive checkpoints when available."""
    prompts = model.prompts or {}
    return role if role in prompts else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on LongEmbed tasks.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="lightonai/GTE-ModernColBERT-v1",
        help="Model path or HF identifier to evaluate.",
    )
    parser.add_argument(
        "--document_batch_size",
        type=int,
        default=10,
        help="Document encoding batch size.",
    )
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=32,
        help="Query encoding batch size.",
    )
    args = parser.parse_args()

    tasks = mteb.get_tasks(
        tasks=[
            "LEMBNarrativeQARetrieval",
            "LEMBNeedleRetrieval",
            "LEMBPasskeyRetrieval",
            "LEMBQMSumRetrieval",
            "LEMBSummScreenFDRetrieval",
            "LEMBWikimQARetrieval",
        ]
    )
    for task in tasks:
        task.load_data()
        model_name = args.model_name_or_path
        model = models.ColBERT(
            model_name_or_path=model_name,
            document_length=16384,
            trust_remote_code=True,
            local_files_only=Path(model_name).exists(),
        )
        query_prompt_name = resolve_prompt_name(model=model, role="query")
        document_prompt_name = resolve_prompt_name(model=model, role="document")
        for eval_set in task.queries.keys():
            index = indexes.PLAID(
                override=True,
                nbits=4,
                index_name=f"{task.metadata.name}_{eval_set}_{model_name.split('/')[-1]}_4bits_ir",
            )

            retriever = retrieve.ColBERT(index=index)

            documents_embeddings = model.encode(
                sentences=list(task.corpus[eval_set].values()),
                batch_size=args.document_batch_size,
                is_query=False,
                show_progress_bar=True,
                prompt_name=document_prompt_name,
            )

            index.add_documents(
                documents_ids=list(task.corpus[eval_set].keys()),
                documents_embeddings=documents_embeddings,
            )
            queries_embeddings = model.encode(
                sentences=list(task.queries[eval_set].values()),
                is_query=True,
                show_progress_bar=True,
                batch_size=args.query_batch_size,
                prompt_name=query_prompt_name,
            )

            scores = retriever.retrieve(queries_embeddings=queries_embeddings)

            evaluation_scores = evaluation.evaluate(
                scores=scores,
                qrels=task.relevant_docs[eval_set],
                queries=list(task.queries[eval_set].keys()),
                metrics=[
                    "map",
                    "ndcg@1",
                    "ndcg@10",
                    "ndcg@100",
                    "recall@10",
                    "recall@100",
                ],
            )

            print(evaluation_scores)
