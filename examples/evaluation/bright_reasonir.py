from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
import re

import pytrec_eval
import torch
from datasets import load_dataset
from tqdm import trange

from pylate import models
from pylate.scores import colbert_scores

TASKS = [
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "leetcode",
    "pony",
    "aops",
    "theoremqa_questions",
    "theoremqa_theorems",
]

DISPLAY_NAMES = {
    "biology": "Biology",
    "earth_science": "Earth",
    "economics": "Economics",
    "psychology": "Psychology",
    "robotics": "Robotics",
    "stackoverflow": "Stackoverflow",
    "sustainable_living": "Sustainable",
    "leetcode": "Leetcode",
    "pony": "Pony",
    "aops": "AoPS",
    "theoremqa_questions": "Theorem - Q",
    "theoremqa_theorems": "Theorem - T",
}

STACKEXCHANGE_TASKS = [
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
]
CODING_TASKS = ["leetcode", "pony", "aops"]
THEOREM_TASKS = ["theoremqa_questions", "theoremqa_theorems"]

# Query lengths explicitly listed in the Reason-ModernColBERT model card for the
# GPT-4 reasoning-trace setup.
REASON_MODERNCOLBERT_GPT4_QUERY_LENGTHS = {
    "biology": 256,
    "earth_science": 1024,
    "economics": 128,
    "psychology": 128,
    "robotics": 256,
    "stackoverflow": 256,
    "sustainable_living": 128,
    "leetcode": 128,
    "pony": 128,
    "aops": 256,
    "theoremqa_questions": 256,
    "theoremqa_theorems": 128,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run BRIGHT evaluation with a PyLate ColBERT checkpoint using the same "
            "task splits and metrics as the ReasonIR evaluation harness."
        )
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Model path or HF identifier to evaluate.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(TASKS),
        help="Comma-separated BRIGHT tasks to evaluate.",
    )
    parser.add_argument(
        "--reasoning",
        type=str,
        default="none",
        choices=["none", "gpt4"],
        help="Query variant to evaluate: raw BRIGHT queries or gpt4_reason.",
    )
    parser.add_argument(
        "--long_context",
        action="store_true",
        help="Use BRIGHT long_documents and gold_ids_long.",
    )
    parser.add_argument(
        "--query_batch_size",
        type=int,
        default=8,
        help="Number of queries scored together in a MaxSim batch.",
    )
    parser.add_argument(
        "--document_batch_size",
        type=int,
        default=64,
        help="Batch size for document encoding.",
    )
    parser.add_argument(
        "--query_encode_batch_size",
        type=int,
        default=32,
        help="Batch size for query encoding.",
    )
    parser.add_argument(
        "--corpus_chunk_size",
        type=int,
        default=128,
        help="Number of documents scored together against a query batch.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1000,
        help="Number of retrieved documents to retain per query.",
    )
    parser.add_argument(
        "--query_length",
        type=int,
        default=None,
        help="Optional global query length override.",
    )
    parser.add_argument(
        "--document_length",
        type=int,
        default=None,
        help="Optional document length override.",
    )
    parser.add_argument(
        "--use_reason_moderncolbert_gpt4_lengths",
        action="store_true",
        help=(
            "For --reasoning gpt4, use the per-task query lengths listed in the "
            "Reason-ModernColBERT model card."
        ),
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/tmp/pylate-bright-cache",
        help="Hugging Face datasets cache directory.",
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional path to write the full BRIGHT result JSON.",
    )
    return parser.parse_args()


def calculate_retrieval_metrics(
    results: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    if k_values is None:
        k_values = [1, 5, 10, 25, 50, 100]

    ndcg = {f"NDCG@{k}": 0.0 for k in k_values}
    map_scores = {f"MAP@{k}": 0.0 for k in k_values}
    recall = {f"Recall@{k}": 0.0 for k in k_values}
    precision = {f"P@{k}": 0.0 for k in k_values}
    mrr = {"MRR": 0.0}

    map_string = "map_cut." + ",".join(str(k) for k in k_values)
    ndcg_string = "ndcg_cut." + ",".join(str(k) for k in k_values)
    recall_string = "recall." + ",".join(str(k) for k in k_values)
    precision_string = "P." + ",".join(str(k) for k in k_values)

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels,
        {map_string, ndcg_string, recall_string, precision_string, "recip_rank"},
    )
    scores = evaluator.evaluate(results)

    for query_id in scores:
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id][f"ndcg_cut_{k}"]
            map_scores[f"MAP@{k}"] += scores[query_id][f"map_cut_{k}"]
            recall[f"Recall@{k}"] += scores[query_id][f"recall_{k}"]
            precision[f"P@{k}"] += scores[query_id][f"P_{k}"]
        mrr["MRR"] += scores[query_id]["recip_rank"]

    denominator = len(scores)
    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / denominator, 5)
        map_scores[f"MAP@{k}"] = round(map_scores[f"MAP@{k}"] / denominator, 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / denominator, 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / denominator, 5)
    mrr["MRR"] = round(mrr["MRR"] / denominator, 5)

    return {**ndcg, **map_scores, **recall, **precision, **mrr}


def resolve_prompt_name(model: models.ColBERT, role: str) -> str | None:
    prompts = model.prompts or {}
    return role if role in prompts else None


def slugify(value: str) -> str:
    return re.sub(pattern=r"[^A-Za-z0-9._-]+", repl="_", string=value).strip("_")


def load_bright_task(
    task: str,
    cache_dir: str,
    reasoning: str,
    long_context: bool,
) -> tuple[list[str], list[str], list[set[str]], dict[str, dict[str, int]], list[str], list[str]]:
    dataset_source = "xlangai/BRIGHT"

    if reasoning == "none":
        examples = load_dataset(dataset_source, "examples", cache_dir=cache_dir)[task]
    else:
        examples = load_dataset(
            dataset_source, f"{reasoning}_reason", cache_dir=cache_dir
        )[task]

    documents_split = "long_documents" if long_context else "documents"
    doc_pairs = load_dataset(dataset_source, documents_split, cache_dir=cache_dir)[task]

    query_ids: list[str] = []
    queries: list[str] = []
    excluded_ids: list[set[str]] = []
    qrels: dict[str, dict[str, int]] = {}
    gold_key = "gold_ids_long" if long_context else "gold_ids"

    for example in examples:
        query_id = str(example["id"])
        query_ids.append(query_id)
        queries.append(example["query"])
        excluded_ids.append({doc_id for doc_id in example["excluded_ids"] if doc_id != "N/A"})
        qrels[query_id] = {str(doc_id): 1 for doc_id in example[gold_key]}

    corpus_ids = [str(sample["id"]) for sample in doc_pairs]
    corpus = [sample["content"] for sample in doc_pairs]
    return query_ids, queries, excluded_ids, qrels, corpus_ids, corpus


def get_query_length_for_task(args: argparse.Namespace, task: str, default: int | None) -> int | None:
    if args.query_length is not None:
        return args.query_length
    if args.reasoning == "gpt4" and args.use_reason_moderncolbert_gpt4_lengths:
        return REASON_MODERNCOLBERT_GPT4_QUERY_LENGTHS[task]
    return default


def evaluate_task(
    model: models.ColBERT,
    model_name_or_path: str,
    task: str,
    query_ids: list[str],
    queries: list[str],
    excluded_ids: list[set[str]],
    qrels: dict[str, dict[str, int]],
    corpus_ids: list[str],
    corpus: list[str],
    query_prompt_name: str | None,
    document_prompt_name: str | None,
    query_encode_batch_size: int,
    document_batch_size: int,
    query_batch_size: int,
    corpus_chunk_size: int,
    top_k: int,
    cache_dir: str,
) -> dict[str, float]:
    query_embeddings = model.encode(
        sentences=queries,
        is_query=True,
        batch_size=query_encode_batch_size,
        show_progress_bar=True,
        prompt_name=query_prompt_name,
        convert_to_numpy=False,
    )

    query_heaps: dict[str, list[tuple[float, str]]] = {query_id: [] for query_id in query_ids}

    for corpus_start_idx in trange(
        0,
        len(corpus),
        corpus_chunk_size,
        desc=f"{task}: corpus chunks",
    ):
        corpus_end_idx = min(corpus_start_idx + corpus_chunk_size, len(corpus))
        sub_corpus_ids = corpus_ids[corpus_start_idx:corpus_end_idx]

        cache_file = (
            Path(cache_dir)
            / "bright_doc_emb"
            / slugify(model_name_or_path)
            / task
            / f"doclen_{model.document_length}"
            / f"chunk_{corpus_chunk_size}_bs_{document_batch_size}"
            / f"{corpus_start_idx}.pt"
        )
        if cache_file.exists():
            sub_corpus_embeddings = torch.load(cache_file, map_location=model.device)
        else:
            sub_corpus_embeddings = torch.nn.utils.rnn.pad_sequence(
                model.encode(
                    sentences=corpus[corpus_start_idx:corpus_end_idx],
                    is_query=False,
                    batch_size=document_batch_size,
                    show_progress_bar=False,
                    prompt_name=document_prompt_name,
                    convert_to_numpy=False,
                ),
                batch_first=True,
                padding_value=0,
            )
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(sub_corpus_embeddings.cpu(), cache_file)
        local_doc_index = {
            corpus_id: local_idx for local_idx, corpus_id in enumerate(sub_corpus_ids)
        }

        for query_start_idx in range(0, len(query_ids), query_batch_size):
            query_end_idx = min(query_start_idx + query_batch_size, len(query_ids))
            sub_query_ids = query_ids[query_start_idx:query_end_idx]
            sub_query_embeddings = torch.nn.utils.rnn.pad_sequence(
                query_embeddings[query_start_idx:query_end_idx],
                batch_first=True,
                padding_value=0,
            )

            pair_scores = colbert_scores(
                queries_embeddings=sub_query_embeddings,
                documents_embeddings=sub_corpus_embeddings,
            )

            for query_offset, excluded_for_query in enumerate(
                excluded_ids[query_start_idx:query_end_idx]
            ):
                excluded_local_indices = [
                    local_doc_index[doc_id]
                    for doc_id in excluded_for_query
                    if doc_id in local_doc_index
                ]
                if excluded_local_indices:
                    pair_scores[query_offset, excluded_local_indices] = float("-inf")

            top_scores, top_indices = torch.topk(
                pair_scores,
                k=min(top_k, pair_scores.shape[1]),
                dim=1,
                largest=True,
                sorted=False,
            )

            for row_idx, query_id in enumerate(sub_query_ids):
                heap = query_heaps[query_id]
                for local_idx, score in zip(
                    top_indices[row_idx].tolist(),
                    top_scores[row_idx].tolist(),
                ):
                    if score == float("-inf"):
                        continue
                    candidate = (float(score), sub_corpus_ids[local_idx])
                    if len(heap) < top_k:
                        heapq.heappush(heap, candidate)
                    else:
                        heapq.heappushpop(heap, candidate)

        del sub_corpus_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results: dict[str, dict[str, float]] = {}
    for query_id in query_ids:
        ranked = sorted(query_heaps[query_id], key=lambda item: item[0], reverse=True)
        deduplicated: dict[str, float] = {}
        for score, doc_id in ranked:
            if doc_id not in deduplicated:
                deduplicated[doc_id] = score
        results[query_id] = deduplicated

    return calculate_retrieval_metrics(results=results, qrels=qrels)


def build_summary(task_metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    def mean_ndcg(tasks: list[str]) -> float:
        available_tasks = [task for task in tasks if task in task_metrics]
        if not available_tasks:
            return 0.0
        return round(
            sum(task_metrics[task]["NDCG@10"] for task in available_tasks)
            / len(available_tasks)
            * 100,
            2,
        )

    summary = {
        DISPLAY_NAMES[task]: round(metrics["NDCG@10"] * 100, 2)
        for task, metrics in task_metrics.items()
    }
    summary["Mean StackExchange"] = mean_ndcg(STACKEXCHANGE_TASKS)
    summary["Mean coding"] = mean_ndcg(CODING_TASKS)
    summary["Mean theorem"] = mean_ndcg(THEOREM_TASKS)
    summary["Full mean"] = round(
        sum(metrics["NDCG@10"] for metrics in task_metrics.values())
        / len(task_metrics)
        * 100,
        2,
    )
    return summary


def main() -> None:
    args = parse_args()
    task_names = [task.strip() for task in args.tasks.split(",") if task.strip()]
    model_path = Path(args.model_name_or_path)
    model = models.ColBERT(
        model_name_or_path=args.model_name_or_path,
        query_length=args.query_length,
        document_length=args.document_length,
        local_files_only=model_path.exists(),
    )
    query_prompt_name = resolve_prompt_name(model=model, role="query")
    document_prompt_name = resolve_prompt_name(model=model, role="document")

    task_metrics: dict[str, dict[str, float]] = {}
    query_lengths_used: dict[str, int | None] = {}

    if args.output_json is not None and args.output_json.exists():
        existing_output = json.loads(args.output_json.read_text())
        task_metrics = existing_output.get("task_metrics", {})
        query_lengths_used = existing_output.get("query_lengths_used", {})

    for task in task_names:
        if task in task_metrics:
            print(json.dumps({"task": task, "status": "skipped_existing"}))
            continue
        model.query_length = get_query_length_for_task(
            args=args, task=task, default=model.query_length
        )
        query_lengths_used[task] = model.query_length

        query_ids, queries, excluded_ids, qrels, corpus_ids, corpus = load_bright_task(
            task=task,
            cache_dir=args.cache_dir,
            reasoning=args.reasoning,
            long_context=args.long_context,
        )

        print(
            json.dumps(
                {
                    "task": task,
                    "queries": len(query_ids),
                    "documents": len(corpus_ids),
                    "query_length": model.query_length,
                    "document_length": model.document_length,
                    "reasoning": args.reasoning,
                }
            )
        )

        metrics = evaluate_task(
            model=model,
            model_name_or_path=args.model_name_or_path,
            task=task,
            query_ids=query_ids,
            queries=queries,
            excluded_ids=excluded_ids,
            qrels=qrels,
            corpus_ids=corpus_ids,
            corpus=corpus,
            query_prompt_name=query_prompt_name,
            document_prompt_name=document_prompt_name,
            query_encode_batch_size=args.query_encode_batch_size,
            document_batch_size=args.document_batch_size,
            query_batch_size=args.query_batch_size,
            corpus_chunk_size=args.corpus_chunk_size,
            top_k=args.top_k,
            cache_dir=args.cache_dir,
        )
        task_metrics[task] = metrics
        summary = build_summary(task_metrics=task_metrics)
        output = {
            "model_name_or_path": args.model_name_or_path,
            "reasoning": args.reasoning,
            "long_context": args.long_context,
            "query_lengths_used": query_lengths_used,
            "document_length": model.document_length,
            "task_metrics": task_metrics,
            "summary_ndcg10_percent": summary,
        }

        print(
            json.dumps(
                {
                    "task": task,
                    "NDCG@10": metrics["NDCG@10"],
                    "MRR": metrics["MRR"],
                    "MAP@100": metrics["MAP@100"],
                },
                sort_keys=True,
            )
        )

        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(output, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    summary = build_summary(task_metrics=task_metrics)
    output = {
        "model_name_or_path": args.model_name_or_path,
        "reasoning": args.reasoning,
        "long_context": args.long_context,
        "query_lengths_used": query_lengths_used,
        "document_length": model.document_length,
        "task_metrics": task_metrics,
        "summary_ndcg10_percent": summary,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
