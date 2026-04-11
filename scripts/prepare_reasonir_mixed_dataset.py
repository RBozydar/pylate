from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download, list_repo_files

LOGGER = logging.getLogger(__name__)

DEFAULT_REASONIR_ROOT = (
    Path("/mnt/ml_models/datasets/ReasonIR/synthetic_data")
    if Path("/mnt/ml_models/datasets/ReasonIR/synthetic_data").exists()
    else Path("/home/rbw/repo/ReasonIR/synthetic_data_generation/synthetic_data")
)
DEFAULT_HF_CACHE_DIR = Path("/mnt/ml_models/datasets")
DEFAULT_DATASET_NAME = "mixed"
DEFAULT_PROMPT_ID = "hq_gen"
DEFAULT_MIX_NAME = "balanced-v1"
DEFAULT_REASONIR_GENERATOR = "gemini-3-flash-preview"
DEFAULT_NOMIC_SPLITS = [
    "msmarco_distillation_simlm_rescored_reranked_min15",
    "reddit_triples",
    "nq_cocondensor_hn_mine_reranked_min15",
    "nli_simcse_50negs_fixed",
    "medi_sts_wiki_rephrasal",
    "medi_sts_stackexchange_dupe",
]
DEFAULT_COUNTS = {
    "hq": 10_000,
    "vl": 10_000,
    "nomic": 10_000,
    "2wiki": 5_000,
    "qasc": 5_000,
    "hover": 4_000,
    "strategyqa": -1,
}

HOVER_HOP_FIELDS = ("num_hops", "hops", "hop", "num_hop")
TWO_WIKI_BRIDGE_TYPES = {"bridge", "bridge_comparison"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a balanced mixed ReasonIR training dataset under "
            "/mnt/ml_models/datasets/ReasonIR/synthetic_data that can be consumed "
            "directly by examples/train/ColBERT-zero/reasonir.py."
        )
    )
    parser.add_argument(
        "--reasonir-root",
        type=Path,
        default=DEFAULT_REASONIR_ROOT,
        help=f"Root containing ReasonIR synthetic_data. Default: {DEFAULT_REASONIR_ROOT}.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=DEFAULT_HF_CACHE_DIR,
        help=f"Writable cache/download root for external datasets. Default: {DEFAULT_HF_CACHE_DIR}.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help=f"Dataset group name written under synthetic_data/. Default: {DEFAULT_DATASET_NAME}.",
    )
    parser.add_argument(
        "--prompt-id",
        type=str,
        default=DEFAULT_PROMPT_ID,
        help=f"Prompt directory name written under the dataset group. Default: {DEFAULT_PROMPT_ID}.",
    )
    parser.add_argument(
        "--mix-name",
        type=str,
        default=DEFAULT_MIX_NAME,
        help=f"Generator/mix directory name written under the prompt directory. Default: {DEFAULT_MIX_NAME}.",
    )
    parser.add_argument(
        "--reasonir-generator",
        type=str,
        default=DEFAULT_REASONIR_GENERATOR,
        help=(
            "Local HQ/VL generator directory to read from under the ReasonIR root. "
            f"Default: {DEFAULT_REASONIR_GENERATOR}."
        ),
    )
    parser.add_argument(
        "--hq-count",
        type=int,
        default=DEFAULT_COUNTS["hq"],
        help="Number of regenerated HQ rows to keep. Default: 10000.",
    )
    parser.add_argument(
        "--vl-count",
        type=int,
        default=DEFAULT_COUNTS["vl"],
        help="Number of regenerated VL rows to keep. Default: 10000.",
    )
    parser.add_argument(
        "--nomic-count",
        type=int,
        default=DEFAULT_COUNTS["nomic"],
        help="Number of Nomic general triplets to keep. Default: 10000.",
    )
    parser.add_argument(
        "--wiki-count",
        type=int,
        default=DEFAULT_COUNTS["2wiki"],
        help="Number of 2WikiMultiHopQA rows to keep. Default: 5000.",
    )
    parser.add_argument(
        "--qasc-count",
        type=int,
        default=DEFAULT_COUNTS["qasc"],
        help="Number of QASC rows to keep. Default: 5000.",
    )
    parser.add_argument(
        "--hover-count",
        type=int,
        default=DEFAULT_COUNTS["hover"],
        help="Number of HoVer rows to keep. Default: 4000.",
    )
    parser.add_argument(
        "--strategyqa-count",
        type=int,
        default=DEFAULT_COUNTS["strategyqa"],
        help="Number of StrategyQA rows to keep. Use -1 for all. Default: -1.",
    )
    parser.add_argument(
        "--nomic-splits",
        type=str,
        default=",".join(DEFAULT_NOMIC_SPLITS),
        help=(
            "Comma-separated Nomic splits used for the general-retrieval anchor. "
            f"Default: {','.join(DEFAULT_NOMIC_SPLITS)}."
        ),
    )
    parser.add_argument(
        "--nomic-shards-per-split",
        type=int,
        default=1,
        help=(
            "Number of random parquet shards to download per selected Nomic split "
            "before sampling. Default: 1."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for every sampling decision. Default: 42.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing mixed dataset directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned output and counts without writing files.",
    )
    return parser.parse_args()


def parse_csv(raw_value: str) -> list[str]:
    values = [value.strip() for value in raw_value.split(",") if value.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    return values


def normalize_text(value: Any) -> str:
    return " ".join(str(value).split()).strip()


def load_json_field(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def resolve_reasonir_file(
    *,
    reasonir_root: Path,
    dataset_name: str,
    prompt_id: str,
    generator: str,
) -> Path:
    data_file = (
        reasonir_root.expanduser()
        / dataset_name
        / prompt_id
        / generator
        / "final_train_data.jsonl"
    )
    if not data_file.is_file():
        raise FileNotFoundError(f"Missing local ReasonIR file: {data_file}")
    return data_file


def has_nonempty_text(values: Iterable[Any]) -> bool:
    return any(normalize_text(value) for value in values)


def load_reasonir_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def download_dataset_file(
    *,
    repo_id: str,
    filename: str,
    raw_root: Path,
) -> Path:
    target_root = raw_root / repo_id.replace("/", "___")
    target_root.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=str(target_root),
        )
    )


def load_single_file_dataset(
    *,
    repo_id: str,
    filename: str,
    file_format: str,
    raw_root: Path,
    cache_dir: Path,
) -> Dataset:
    local_file = download_dataset_file(repo_id=repo_id, filename=filename, raw_root=raw_root)
    return load_dataset(
        file_format,
        data_files={"train": str(local_file)},
        split="train",
        cache_dir=str(cache_dir),
    )


def sample_rows(
    rows: list[dict[str, Any]],
    *,
    target_count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if target_count < 0 or target_count >= len(rows):
        sampled = list(rows)
        rng.shuffle(sampled)
        return sampled
    return rng.sample(rows, target_count)


def allocate_evenly(
    capacities: dict[str, int],
    total_count: int,
) -> dict[str, int]:
    if total_count < 0 or total_count >= sum(capacities.values()):
        return capacities.copy()

    allocated = {key: 0 for key in capacities}
    active_keys = sorted(key for key, capacity in capacities.items() if capacity > 0)

    while sum(allocated.values()) < total_count and active_keys:
        next_active_keys: list[str] = []
        for key in active_keys:
            if sum(allocated.values()) >= total_count:
                break
            if allocated[key] < capacities[key]:
                allocated[key] += 1
            if allocated[key] < capacities[key]:
                next_active_keys.append(key)
        active_keys = next_active_keys

    return allocated


def sample_grouped_rows(
    rows: list[dict[str, Any]],
    *,
    key_name: str,
    target_count: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key_name])].append(row)

    allocations = allocate_evenly(
        {key: len(group_rows) for key, group_rows in groups.items()},
        total_count=target_count,
    )
    sampled_rows: list[dict[str, Any]] = []
    realized_counts: dict[str, int] = {}
    for key, count in allocations.items():
        group_rows = list(groups[key])
        rng.shuffle(group_rows)
        sampled = group_rows[:count]
        sampled_rows.extend(sampled)
        realized_counts[key] = len(sampled)

    rng.shuffle(sampled_rows)
    return sampled_rows, realized_counts


def rotate_negatives(
    positives: list[str],
    *,
    rng: random.Random,
) -> list[str]:
    if len(positives) < 2:
        raise ValueError("Need at least two positives to build random negatives.")

    indices = list(range(len(positives)))
    shuffled_indices = list(indices)
    rng.shuffle(shuffled_indices)
    for index, shuffled_index in enumerate(shuffled_indices):
        if index == shuffled_index:
            shuffled_indices = shuffled_indices[1:] + shuffled_indices[:1]
            break
    return [positives[index] for index in shuffled_indices]


def build_row(
    *,
    query: str,
    positive: str,
    negative: str,
    source: str,
    stratum: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "query": normalize_text(query),
        "pos": [normalize_text(positive)],
        "neg": [normalize_text(negative)],
        "source": source,
    }
    if stratum is not None:
        row["stratum"] = stratum
    if metadata:
        row.update(metadata)
    return row


def resolve_target_count(target_count: int, available: int) -> int:
    if target_count < 0:
        return available
    return min(target_count, available)


def take_first_text(values: Iterable[Any]) -> str:
    for value in values:
        text = normalize_text(value)
        if text:
            return text
    raise ValueError("Expected at least one non-empty text value.")


def prepare_reasonir_source(
    *,
    path: Path,
    source: str,
    target_count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows = load_reasonir_rows(path)
    valid_rows = [
        row
        for row in rows
        if normalize_text(row.get("query"))
        and row.get("pos")
        and row.get("neg")
        and has_nonempty_text(row["pos"])
        and has_nonempty_text(row["neg"])
    ]
    sampled_rows = sample_rows(
        valid_rows,
        target_count=resolve_target_count(target_count, len(valid_rows)),
        rng=rng,
    )
    prepared_rows: list[dict[str, Any]] = []
    for row in sampled_rows:
        prepared_rows.append(
            build_row(
                query=row["query"],
                positive=take_first_text(row["pos"]),
                negative=take_first_text(row["neg"]),
                source=source,
            )
        )
    return prepared_rows


def normalize_2wiki_type(raw_type: str) -> str:
    return "bridge" if raw_type in TWO_WIKI_BRIDGE_TYPES else raw_type


def format_context_page(title: str, sentences: list[str]) -> str:
    body = normalize_text(" ".join(sentences))
    if not body:
        return ""
    return f"{normalize_text(title)}\n{body}"


def prepare_2wiki(
    *,
    raw_root: Path,
    cache_dir: Path,
    target_count: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    dataset = load_single_file_dataset(
        repo_id="xanhho/2WikiMultihopQA",
        filename="train.parquet",
        file_format="parquet",
        raw_root=raw_root,
        cache_dir=cache_dir,
    )
    rows: list[dict[str, Any]] = []
    for row in dataset:
        row = dict(row)
        row["normalized_type"] = normalize_2wiki_type(str(row["type"]))
        rows.append(row)

    sampled_rows, counts = sample_grouped_rows(
        rows,
        key_name="normalized_type",
        target_count=resolve_target_count(target_count, len(rows)),
        rng=rng,
    )

    prepared_rows: list[dict[str, Any]] = []
    for row in sampled_rows:
        context = load_json_field(row["context"])
        supporting_facts = load_json_field(row["supporting_facts"])
        support_titles: list[str] = []
        for title, _ in supporting_facts:
            normalized_title = normalize_text(title)
            if normalized_title and normalized_title not in support_titles:
                support_titles.append(normalized_title)

        positive_pages: list[str] = []
        negative_pages: list[str] = []
        support_title_set = set(support_titles)
        for title, sentences in context:
            page_text = format_context_page(str(title), [str(sentence) for sentence in sentences])
            if not page_text:
                continue
            if normalize_text(title) in support_title_set:
                positive_pages.append(page_text)
            else:
                negative_pages.append(page_text)

        if not positive_pages or not negative_pages:
            continue

        prepared_rows.append(
            build_row(
                query=row["question"],
                positive="\n\n".join(positive_pages),
                negative=rng.choice(negative_pages),
                source="2wiki",
                stratum=row["normalized_type"],
                metadata={"source_id": row["_id"]},
            )
        )

    return prepared_rows, counts


def build_pair_rows_with_random_negatives(
    *,
    rows: list[dict[str, Any]],
    source: str,
    positive_key: str,
    query_key: str,
    rng: random.Random,
    stratum_key: str | None = None,
    metadata_builder: Any | None = None,
) -> list[dict[str, Any]]:
    positives = [normalize_text(row[positive_key]) for row in rows]
    negatives = rotate_negatives(positives, rng=rng)

    prepared_rows: list[dict[str, Any]] = []
    for row, negative in zip(rows, negatives, strict=True):
        metadata = metadata_builder(row) if metadata_builder is not None else None
        prepared_rows.append(
            build_row(
                query=row[query_key],
                positive=row[positive_key],
                negative=negative,
                source=source,
                stratum=str(row[stratum_key]) if stratum_key is not None else None,
                metadata=metadata,
            )
        )
    return prepared_rows


def prepare_qasc(
    *,
    raw_root: Path,
    cache_dir: Path,
    target_count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    dataset = load_single_file_dataset(
        repo_id="allenai/qasc",
        filename="data/train-00000-of-00001.parquet",
        file_format="parquet",
        raw_root=raw_root,
        cache_dir=cache_dir,
    )
    rows = sample_rows(
        [dict(row) for row in dataset],
        target_count=resolve_target_count(target_count, len(dataset)),
        rng=rng,
    )
    for row in rows:
        positive_parts = [row["fact1"], row["fact2"]]
        combined_fact = normalize_text(row.get("combinedfact"))
        if combined_fact and combined_fact not in normalize_text(" ".join(positive_parts)):
            positive_parts.append(combined_fact)
        row["positive_text"] = " ".join(
            part for part in (normalize_text(value) for value in positive_parts) if part
        )
    return build_pair_rows_with_random_negatives(
        rows=rows,
        source="qasc",
        positive_key="positive_text",
        query_key="question",
        rng=rng,
        metadata_builder=lambda row: {"source_id": row["id"]},
    )


def prepare_hover(
    *,
    raw_root: Path,
    cache_dir: Path,
    target_count: int,
    rng: random.Random,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], dict[str, int], str]:
    dataset = load_single_file_dataset(
        repo_id="Dzeniks/hover",
        filename="train.jsonl",
        file_format="json",
        raw_root=raw_root,
        cache_dir=cache_dir,
    )
    rows = [dict(row) for row in dataset if normalize_text(row.get("claim")) and normalize_text(row.get("evidence"))]

    hop_field = next((field for field in HOVER_HOP_FIELDS if field in dataset.column_names), None)
    if hop_field is not None:
        for row in rows:
            row["stratum"] = str(row[hop_field])
        strategy = "hop_count"
    else:
        warning = (
            "Dzeniks/hover does not expose a hop-count field, so HoVer sampling "
            "falls back to label-balanced sampling instead of 2/3/4-hop stratification."
        )
        warnings.append(warning)
        LOGGER.warning(warning)
        for row in rows:
            row["stratum"] = str(row["label"])
        strategy = "label_fallback"

    sampled_rows, counts = sample_grouped_rows(
        rows,
        key_name="stratum",
        target_count=resolve_target_count(target_count, len(rows)),
        rng=rng,
    )
    prepared_rows = build_pair_rows_with_random_negatives(
        rows=sampled_rows,
        source="hover",
        positive_key="evidence",
        query_key="claim",
        rng=rng,
        stratum_key="stratum",
        metadata_builder=lambda row: {"source_id": row["id"], "label": row["label"]},
    )
    return prepared_rows, counts, strategy


def prepare_strategyqa(
    *,
    raw_root: Path,
    cache_dir: Path,
    target_count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    dataset = load_single_file_dataset(
        repo_id="ChilleD/StrategyQA",
        filename="data/train-00000-of-00001-506370352f622815.parquet",
        file_format="parquet",
        raw_root=raw_root,
        cache_dir=cache_dir,
    )
    rows = [
        dict(row)
        for row in dataset
        if normalize_text(row.get("question")) and normalize_text(row.get("facts"))
    ]
    sampled_rows = sample_rows(
        rows,
        target_count=resolve_target_count(target_count, len(rows)),
        rng=rng,
    )
    return build_pair_rows_with_random_negatives(
        rows=sampled_rows,
        source="strategyqa",
        positive_key="facts",
        query_key="question",
        rng=rng,
        metadata_builder=lambda row: {"source_id": row["qid"], "answer": row["answer"]},
    )


def selected_nomic_shards(
    *,
    split_name: str,
    shard_count: int,
    seed: int,
) -> list[str]:
    repo_id = "nomic-ai/nomic-embed-supervised-data"
    prefix = f"data/{split_name}-"
    files = [
        file_name
        for file_name in list_repo_files(repo_id=repo_id, repo_type="dataset")
        if file_name.startswith(prefix) and file_name.endswith(".parquet")
    ]
    if not files:
        raise FileNotFoundError(f"No parquet shards found for Nomic split {split_name!r}.")
    rng = random.Random(f"{seed}:{split_name}")
    rng.shuffle(files)
    return files[: max(1, min(shard_count, len(files)))]


def prepare_nomic(
    *,
    raw_root: Path,
    cache_dir: Path,
    target_count: int,
    split_names: list[str],
    shards_per_split: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, list[str]]]:
    requested_counts = allocate_evenly(
        {split_name: 1_000_000_000 for split_name in split_names},
        total_count=target_count,
    )
    raw_files_by_split: dict[str, list[str]] = {}
    prepared_rows: list[dict[str, Any]] = []
    realized_counts: dict[str, int] = {}

    for split_name in split_names:
        shard_files = selected_nomic_shards(
            split_name=split_name,
            shard_count=shards_per_split,
            seed=rng.randint(0, 10_000_000),
        )
        raw_files_by_split[split_name] = shard_files
        local_files = [
            str(
                download_dataset_file(
                    repo_id="nomic-ai/nomic-embed-supervised-data",
                    filename=filename,
                    raw_root=raw_root,
                )
            )
            for filename in shard_files
        ]
        dataset = load_dataset(
            "parquet",
            data_files={"train": local_files},
            split="train",
            cache_dir=str(cache_dir),
        )
        rows = [dict(row) for row in dataset if row.get("query") and row.get("document") and row.get("negative")]
        sampled_rows = sample_rows(
            rows,
            target_count=resolve_target_count(requested_counts[split_name], len(rows)),
            rng=rng,
        )
        realized_counts[split_name] = len(sampled_rows)
        for row in sampled_rows:
            prepared_rows.append(
                build_row(
                    query=row["query"],
                    positive=row["document"],
                    negative=take_first_text(row["negative"]),
                    source="nomic_general",
                    stratum=split_name,
                    metadata={"nomic_dataset": row["dataset"]},
                )
            )

    rng.shuffle(prepared_rows)
    return prepared_rows, realized_counts, raw_files_by_split


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def ensure_output_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists: {path}. Pass --force to overwrite it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def format_percentage_table(percentages: dict[str, float]) -> str:
    return "\n".join(
        f"- `{source}`: `{value:.2f}%`" for source, value in sorted(percentages.items())
    )


def format_count_table(counts: dict[str, int]) -> str:
    return "\n".join(
        f"- `{source}`: `{value}`" for source, value in sorted(counts.items())
    )


def build_output_readme(manifest: dict[str, Any]) -> str:
    warning_lines = (
        "\n".join(f"- {warning}" for warning in manifest["warnings"])
        if manifest["warnings"]
        else "- none"
    )
    source_files = "\n".join(
        f"- `{source}`: `{path}`"
        for source, path in sorted(manifest["source_files"].items())
    )

    return (
        "# ReasonIR Mixed Dataset\n\n"
        "This directory was generated by `scripts/prepare_reasonir_mixed_dataset.py`.\n\n"
        "## Summary\n\n"
        f"- total rows: `{manifest['total_rows']}`\n"
        "- goal: keep HQ + VL as the largest block, keep a Nomic general anchor, "
        "and avoid any single dataset dominating the mix\n\n"
        "## Realized Counts\n\n"
        f"{format_count_table(manifest['realized_counts'])}\n\n"
        "## Per-source Share\n\n"
        f"{format_percentage_table(manifest['source_percentages'])}\n\n"
        "## Source Files\n\n"
        f"{source_files}\n\n"
        "## Warnings\n\n"
        f"{warning_lines}\n\n"
        "## Operational Notes\n\n"
        "- the first planned full mixed-data run uses `bs=2048`, `lr=8e-5`, "
        "`epochs=3`, and `validation_size=0.01`\n"
        "- run long mixed-data jobs in `tmux`, `screen`, `nohup`, or an equivalent "
        "detached launcher\n"
        "- do not rely on an attached terminal session for a long run\n"
        "- repo runbook: `/home/rbw/repo/pylate/REASONIR_MIXED_RUNBOOK.md`\n\n"
        "## Training Entrypoint\n\n"
        "```bash\n"
        f"{manifest['train_command']}\n"
        "```\n"
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    mix_root = (
        args.reasonir_root.expanduser()
        / args.dataset_name
        / args.prompt_id
        / args.mix_name
    )
    source_root = mix_root / "sources"
    raw_root = args.hf_cache_dir.expanduser() / "_raw"
    warnings: list[str] = []

    if args.dry_run:
        LOGGER.info("Dry run only. Planned output directory: %s", mix_root)
    else:
        ensure_output_dir(mix_root, force=args.force)
        source_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    hq_rows = prepare_reasonir_source(
        path=resolve_reasonir_file(
            reasonir_root=args.reasonir_root,
            dataset_name="hq",
            prompt_id=args.prompt_id,
            generator=args.reasonir_generator,
        ),
        source="reasonir_hq_regen",
        target_count=args.hq_count,
        rng=random.Random(args.seed + 1),
    )
    vl_rows = prepare_reasonir_source(
        path=resolve_reasonir_file(
            reasonir_root=args.reasonir_root,
            dataset_name="vl",
            prompt_id=args.prompt_id,
            generator=args.reasonir_generator,
        ),
        source="reasonir_vl_regen",
        target_count=args.vl_count,
        rng=random.Random(args.seed + 2),
    )
    wiki_rows, _wiki_counts = prepare_2wiki(
        raw_root=raw_root,
        cache_dir=args.hf_cache_dir.expanduser(),
        target_count=args.wiki_count,
        rng=random.Random(args.seed + 3),
    )
    qasc_rows = prepare_qasc(
        raw_root=raw_root,
        cache_dir=args.hf_cache_dir.expanduser(),
        target_count=args.qasc_count,
        rng=random.Random(args.seed + 4),
    )
    hover_rows, _hover_counts, hover_strategy = prepare_hover(
        raw_root=raw_root,
        cache_dir=args.hf_cache_dir.expanduser(),
        target_count=args.hover_count,
        rng=random.Random(args.seed + 5),
        warnings=warnings,
    )
    strategyqa_rows = prepare_strategyqa(
        raw_root=raw_root,
        cache_dir=args.hf_cache_dir.expanduser(),
        target_count=args.strategyqa_count,
        rng=random.Random(args.seed + 6),
    )
    nomic_rows, nomic_counts, nomic_files = prepare_nomic(
        raw_root=raw_root,
        cache_dir=args.hf_cache_dir.expanduser(),
        target_count=args.nomic_count,
        split_names=parse_csv(args.nomic_splits),
        shards_per_split=args.nomic_shards_per_split,
        rng=random.Random(args.seed + 7),
    )

    rows_by_source = {
        "hq": hq_rows,
        "vl": vl_rows,
        "nomic_general": nomic_rows,
        "2wiki": wiki_rows,
        "qasc": qasc_rows,
        "hover": hover_rows,
        "strategyqa": strategyqa_rows,
    }
    final_rows: list[dict[str, Any]] = []
    for source_rows in rows_by_source.values():
        final_rows.extend(source_rows)
    rng.shuffle(final_rows)

    manifest = {
        "mix_root": str(mix_root),
        "source_root": str(source_root),
        "seed": args.seed,
        "reasonir_generator": args.reasonir_generator,
        "requested_counts": {
            "hq": args.hq_count,
            "vl": args.vl_count,
            "nomic": args.nomic_count,
            "2wiki": args.wiki_count,
            "qasc": args.qasc_count,
            "hover": args.hover_count,
            "strategyqa": args.strategyqa_count,
        },
        "realized_counts": {key: len(value) for key, value in rows_by_source.items()},
        "total_rows": len(final_rows),
        "source_percentages": {
            key: round(len(value) / len(final_rows) * 100, 2) if final_rows else 0.0
            for key, value in rows_by_source.items()
        },
        "2wiki_type_counts": dict(Counter(row["stratum"] for row in wiki_rows)),
        "hover_sampling_strategy": hover_strategy,
        "hover_stratum_counts": dict(Counter(row["stratum"] for row in hover_rows)),
        "nomic_split_counts": nomic_counts,
        "nomic_selected_shards": nomic_files,
        "warnings": warnings,
        "prepare_command": (
            "uv run python scripts/prepare_reasonir_mixed_dataset.py "
            f"--reasonir-root {args.reasonir_root.expanduser()} "
            f"--hf-cache-dir {args.hf_cache_dir.expanduser()} "
            f"--dataset-name {args.dataset_name} "
            f"--prompt-id {args.prompt_id} "
            f"--mix-name {args.mix_name} "
            f"--reasonir-generator {args.reasonir_generator} "
            f"--hq-count {args.hq_count} "
            f"--vl-count {args.vl_count} "
            f"--nomic-count {args.nomic_count} "
            f"--wiki-count {args.wiki_count} "
            f"--qasc-count {args.qasc_count} "
            f"--hover-count {args.hover_count} "
            f"--strategyqa-count {args.strategyqa_count} "
            f"--nomic-splits {','.join(parse_csv(args.nomic_splits))} "
            f"--nomic-shards-per-split {args.nomic_shards_per_split} "
            f"--seed {args.seed} "
            "--force"
        ),
        "train_command": (
            "uv run python examples/train/ColBERT-zero/reasonir.py "
            f"--data-root {args.reasonir_root.expanduser()} "
            f"--datasets {args.dataset_name} "
            f"--prompt-id {args.prompt_id} "
            f"--generator {args.mix_name} "
            "--max-negatives 1"
        ),
        "source_files": {
            key: str(source_root / f"{key}.jsonl") for key in rows_by_source
        },
    }

    LOGGER.info("Prepared mixed dataset with %d total rows.", len(final_rows))
    for source_name, source_rows in rows_by_source.items():
        LOGGER.info("%s rows: %d", source_name, len(source_rows))
    LOGGER.info(
        "Per-source percentages: %s",
        {
            key: round(value, 2)
            for key, value in manifest["source_percentages"].items()
        },
    )

    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    for source_name, source_rows in rows_by_source.items():
        write_jsonl(source_root / f"{source_name}.jsonl", source_rows)
    write_jsonl(mix_root / "final_train_data.jsonl", final_rows)
    (mix_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (mix_root / "README.md").write_text(
        build_output_readme(manifest),
        encoding="utf-8",
    )

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
