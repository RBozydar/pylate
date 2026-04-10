from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import math
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch

import wandb

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("/mnt/ml_models/lightonai/ColBERT-Zero")
DEFAULT_TRAIN_SCRIPT = Path("examples/train/ColBERT-zero/reason_moderncolbert.py")
DEFAULT_EVAL_SCRIPT = Path("examples/evaluation/bright_reasonir.py")
DEFAULT_OUTPUT_ROOT = Path("/home/rbw/repo/pylate/output")
DEFAULT_DATASET_CACHE_DIR = Path("/tmp/pylate-hf-cache")
DEFAULT_BRIGHT_CACHE_DIR = (
    Path("/mnt/ml_models/cache/pylate-bright-cache")
    if Path("/mnt/ml_models/cache").is_dir()
    else Path("/tmp/pylate-bright-cache")
)
DEFAULT_TASKS = "biology,economics,robotics,pony"
DEFAULT_REASONING = "none"
DEFAULT_WANDB_PROJECT = "ColBERT-Zero"
DEFAULT_WANDB_ENTITY = "rbw"

BATCH_PRESETS: dict[str, dict[str, float | int]] = {
    "hq-batch-bs256-lr1e5-temp1": {
        "batch_size": 256,
        "max_steps": 400,
        "learning_rate": 1e-5,
        "temperature": 1.0,
    },
    "hq-batch-bs512-lr2e5-temp1": {
        "batch_size": 512,
        "max_steps": 200,
        "learning_rate": 2e-5,
        "temperature": 1.0,
    },
    "hq-batch-bs1024-lr4e5-temp1": {
        "batch_size": 1024,
        "max_steps": 100,
        "learning_rate": 4e-5,
        "temperature": 1.0,
    },
    "hq-batch-bs2048-lr8e5-temp1": {
        "batch_size": 2048,
        "max_steps": 50,
        "learning_rate": 8e-5,
        "temperature": 1.0,
    },
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the HQ sweep runner."""

    parser = argparse.ArgumentParser(
        description=(
            "Run a single ReasonIR HQ train+eval job under W&B Sweeps. The script "
            "trains ColBERT-Zero, runs BRIGHT subset eval on the resulting final "
            "checkpoint, and logs the BRIGHT summary back into the same W&B run."
        )
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="generic",
        choices=("batch", "lr", "temp", "generic"),
        help="Naming mode for the run. Default: generic.",
    )
    parser.add_argument(
        "--batch-preset",
        type=str,
        default=None,
        help="Optional named batch preset that resolves batch size, max steps, LR, and temp.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Per-device train/eval batch size when not using --batch-preset.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Number of optimizer steps for training. Default: 100.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate when not using --batch-preset. Default: 1e-5.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Contrastive temperature when not using --batch-preset. Default: 1.0.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Base model path. Default: {DEFAULT_MODEL_PATH}.",
    )
    parser.add_argument(
        "--train-script",
        type=Path,
        default=DEFAULT_TRAIN_SCRIPT,
        help=f"Training script entrypoint. Default: {DEFAULT_TRAIN_SCRIPT}.",
    )
    parser.add_argument(
        "--eval-script",
        type=Path,
        default=DEFAULT_EVAL_SCRIPT,
        help=f"Evaluation script entrypoint. Default: {DEFAULT_EVAL_SCRIPT}.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root for trained models and eval JSONs. Default: {DEFAULT_OUTPUT_ROOT}.",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        type=Path,
        default=DEFAULT_DATASET_CACHE_DIR,
        help=f"HF dataset cache for training. Default: {DEFAULT_DATASET_CACHE_DIR}.",
    )
    parser.add_argument(
        "--bright-cache-dir",
        type=Path,
        default=DEFAULT_BRIGHT_CACHE_DIR,
        help=f"HF dataset cache for BRIGHT eval. Default: {DEFAULT_BRIGHT_CACHE_DIR}.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.01,
        help="Held-out validation fraction for internal trainer eval. Default: 0.01.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Learning-rate warmup ratio. Default: 0.1.",
    )
    parser.add_argument(
        "--mini-batch-size",
        type=int,
        default=32,
        help="CachedContrastive mini-batch size. Default: 32.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Dataloader worker count. Default: 4.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=1,
        help="Trainer logging cadence. Default: 1.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=25,
        help="Checkpoint save cadence. Default: 25.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="Internal validation cadence. Defaults to --save-steps when omitted.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=20,
        help="Checkpoint retention count. Default: 20.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=DEFAULT_TASKS,
        help=f"Comma-separated BRIGHT subset tasks. Default: {DEFAULT_TASKS}.",
    )
    parser.add_argument(
        "--reasoning",
        type=str,
        default=DEFAULT_REASONING,
        choices=("none", "gpt4"),
        help=f"BRIGHT reasoning mode. Default: {DEFAULT_REASONING}.",
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=16,
        help="MaxSim query batch size for BRIGHT eval. Default: 16.",
    )
    parser.add_argument(
        "--query-encode-batch-size",
        type=int,
        default=32,
        help="Query encoding batch size for BRIGHT eval. Default: 32.",
    )
    parser.add_argument(
        "--document-batch-size",
        type=int,
        default=128,
        help="Document encoding batch size for BRIGHT eval. Default: 128.",
    )
    parser.add_argument(
        "--corpus-chunk-size",
        type=int,
        default=1024,
        help="Corpus chunk size for BRIGHT eval. Default: 1024.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help="Top-k retained in BRIGHT eval. Default: 1000.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional explicit run name override.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory for the training job.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=DEFAULT_WANDB_PROJECT,
        help=f"W&B project name. Default: {DEFAULT_WANDB_PROJECT}.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=DEFAULT_WANDB_ENTITY,
        help=f"W&B entity name. Default: {DEFAULT_WANDB_ENTITY}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved run configuration as JSON and exit.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Expand and resolve a repository-relative path."""

    return path.expanduser().resolve()


def serialize_runtime_value(value: Any) -> Any:
    """Convert runtime values into W&B-config-safe primitives."""

    if isinstance(value, Path):
        return str(value)
    return value


def normalize_metric_name(value: str) -> str:
    """Normalize a metric label into a stable W&B-safe key."""

    value = value.lower().replace("@", "_at_").replace("%", "_percent")
    return re.sub(pattern=r"[^a-z0-9]+", repl="_", string=value).strip("_")


def format_lr_token(value: float) -> str:
    """Format a learning rate token to match the repo's existing run names."""

    scientific = f"{value:.0e}"
    scientific = scientific.replace(".0e", "e")
    scientific = re.sub(pattern=r"e-0*(\d+)$", repl=r"e\1", string=scientific)
    scientific = re.sub(pattern=r"e\+0*(\d+)$", repl=r"ep\1", string=scientific)
    return scientific


def format_temp_token(value: float) -> str:
    """Format a temperature token to match the repo's existing run names."""

    if math.isclose(value, round(value)):
        return str(int(round(value)))
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "")


def build_default_run_name(
    *,
    stage: str,
    batch_preset: str | None,
    batch_size: int,
    learning_rate: float,
    temperature: float,
) -> str:
    """Build the default run name for this configuration."""

    if batch_preset is not None:
        return batch_preset
    lr_token = format_lr_token(learning_rate)
    temp_token = format_temp_token(temperature)
    if stage == "lr":
        return f"hq-lr-bs{batch_size}-lr{lr_token}-temp{temp_token}"
    if stage == "temp":
        return f"hq-temp-bs{batch_size}-lr{lr_token}-temp{temp_token}"
    if stage == "batch":
        return f"hq-batch-bs{batch_size}-lr{lr_token}-temp{temp_token}"
    return f"hq-sweep-bs{batch_size}-lr{lr_token}-temp{temp_token}"


def resolve_config_value(
    config: wandb.sdk.wandb_config.Config,
    args: argparse.Namespace,
    key: str,
    default: Any = None,
) -> Any:
    """Resolve a value from W&B config first, then CLI args, then fallback."""

    configured_value = config.get(key)
    if configured_value is not None:
        return configured_value
    if hasattr(args, key):
        return getattr(args, key)
    return default


def resolve_runtime_configuration(
    config: wandb.sdk.wandb_config.Config,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Resolve the final train+eval configuration for a single sweep run."""

    stage = str(resolve_config_value(config, args, "stage"))
    batch_preset = resolve_config_value(config, args, "batch_preset")
    if batch_preset is not None:
        if batch_preset not in BATCH_PRESETS:
            raise ValueError(
                f"Unknown batch preset {batch_preset!r}. Available presets: {sorted(BATCH_PRESETS)}"
            )
        preset_values = BATCH_PRESETS[batch_preset]
        batch_size = int(preset_values["batch_size"])
        max_steps = int(preset_values["max_steps"])
        learning_rate = float(preset_values["learning_rate"])
        temperature = float(preset_values["temperature"])
    else:
        batch_size = int(resolve_config_value(config, args, "batch_size"))
        max_steps = int(resolve_config_value(config, args, "max_steps"))
        learning_rate = float(resolve_config_value(config, args, "learning_rate"))
        temperature = float(resolve_config_value(config, args, "temperature"))

    save_steps = int(resolve_config_value(config, args, "save_steps"))
    eval_steps = resolve_config_value(config, args, "eval_steps")
    if eval_steps is None:
        eval_steps = save_steps

    run_name = resolve_config_value(config, args, "run_name")
    if run_name is None:
        run_name = build_default_run_name(
            stage=stage,
            batch_preset=batch_preset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            temperature=temperature,
        )

    output_root = resolve_path(Path(resolve_config_value(config, args, "output_root")))
    output_dir_value = resolve_config_value(config, args, "output_dir")
    output_dir = (
        resolve_path(Path(output_dir_value))
        if output_dir_value is not None
        else output_root / run_name
    )
    eval_output_json = output_root / f"{run_name}-bright-subset.json"

    return {
        "stage": stage,
        "batch_preset": batch_preset,
        "batch_size": batch_size,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "temperature": temperature,
        "model": resolve_path(Path(resolve_config_value(config, args, "model"))),
        "train_script": resolve_path(Path(resolve_config_value(config, args, "train_script"))),
        "eval_script": resolve_path(Path(resolve_config_value(config, args, "eval_script"))),
        "dataset_cache_dir": resolve_path(
            Path(resolve_config_value(config, args, "dataset_cache_dir"))
        ),
        "bright_cache_dir": resolve_path(
            Path(resolve_config_value(config, args, "bright_cache_dir"))
        ),
        "output_root": output_root,
        "output_dir": output_dir,
        "eval_output_json": eval_output_json,
        "validation_size": float(resolve_config_value(config, args, "validation_size")),
        "warmup_ratio": float(resolve_config_value(config, args, "warmup_ratio")),
        "mini_batch_size": int(resolve_config_value(config, args, "mini_batch_size")),
        "num_workers": int(resolve_config_value(config, args, "num_workers")),
        "logging_steps": int(resolve_config_value(config, args, "logging_steps")),
        "save_steps": save_steps,
        "eval_steps": int(eval_steps),
        "save_total_limit": int(resolve_config_value(config, args, "save_total_limit")),
        "tasks": str(resolve_config_value(config, args, "tasks")),
        "reasoning": str(resolve_config_value(config, args, "reasoning")),
        "query_batch_size": int(resolve_config_value(config, args, "query_batch_size")),
        "query_encode_batch_size": int(
            resolve_config_value(config, args, "query_encode_batch_size")
        ),
        "document_batch_size": int(
            resolve_config_value(config, args, "document_batch_size")
        ),
        "corpus_chunk_size": int(resolve_config_value(config, args, "corpus_chunk_size")),
        "top_k": int(resolve_config_value(config, args, "top_k")),
        "wandb_project": str(resolve_config_value(config, args, "wandb_project")),
        "wandb_entity": str(resolve_config_value(config, args, "wandb_entity")),
        "run_name": run_name,
    }


def base_wandb_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build the default W&B config for a sweep run."""

    return {
        "stage": args.stage,
        "batch_preset": args.batch_preset,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "model": str(args.model),
        "train_script": str(args.train_script),
        "eval_script": str(args.eval_script),
        "output_root": str(args.output_root),
        "dataset_cache_dir": str(args.dataset_cache_dir),
        "bright_cache_dir": str(args.bright_cache_dir),
        "validation_size": args.validation_size,
        "warmup_ratio": args.warmup_ratio,
        "mini_batch_size": args.mini_batch_size,
        "num_workers": args.num_workers,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit,
        "tasks": args.tasks,
        "reasoning": args.reasoning,
        "query_batch_size": args.query_batch_size,
        "query_encode_batch_size": args.query_encode_batch_size,
        "document_batch_size": args.document_batch_size,
        "corpus_chunk_size": args.corpus_chunk_size,
        "top_k": args.top_k,
    }


def load_module(module_path: Path, module_name: str) -> Any:
    """Load a Python module from an arbitrary file path."""

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def temporary_argv(argv: list[str]) -> Iterator[None]:
    """Temporarily replace sys.argv while calling a script's main()."""

    original_argv = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = original_argv


def run_script_main(
    *,
    module_path: Path,
    module_name: str,
    argv: list[str],
) -> None:
    """Load a script module from disk and execute its main() function."""

    module = load_module(module_path=module_path, module_name=module_name)
    with temporary_argv([str(module_path), *argv]):
        module.main()


def build_training_argv(runtime: dict[str, Any]) -> list[str]:
    """Build CLI arguments for the HQ training script."""

    return [
        "--model",
        str(runtime["model"]),
        "--max-steps",
        str(runtime["max_steps"]),
        "--validation-size",
        str(runtime["validation_size"]),
        "--warmup-ratio",
        str(runtime["warmup_ratio"]),
        "--bs",
        str(runtime["batch_size"]),
        "--eval-bs",
        str(runtime["batch_size"]),
        "--mini-batch-size",
        str(runtime["mini_batch_size"]),
        "--lr",
        str(runtime["learning_rate"]),
        "--temp",
        str(runtime["temperature"]),
        "--num-workers",
        str(runtime["num_workers"]),
        "--logging-steps",
        str(runtime["logging_steps"]),
        "--eval-steps",
        str(runtime["eval_steps"]),
        "--save-steps",
        str(runtime["save_steps"]),
        "--save-total-limit",
        str(runtime["save_total_limit"]),
        "--report-to",
        "wandb",
        "--wandb-project",
        str(runtime["wandb_project"]),
        "--wandb-entity",
        str(runtime["wandb_entity"]),
        "--run-name",
        str(runtime["run_name"]),
        "--dataset-cache-dir",
        str(runtime["dataset_cache_dir"]),
        "--output-dir",
        str(runtime["output_dir"]),
    ]


def build_eval_argv(runtime: dict[str, Any]) -> list[str]:
    """Build CLI arguments for the BRIGHT subset evaluator."""

    return [
        "--model_name_or_path",
        str(runtime["output_dir"] / "final"),
        "--tasks",
        str(runtime["tasks"]),
        "--reasoning",
        str(runtime["reasoning"]),
        "--query_batch_size",
        str(runtime["query_batch_size"]),
        "--query_encode_batch_size",
        str(runtime["query_encode_batch_size"]),
        "--document_batch_size",
        str(runtime["document_batch_size"]),
        "--corpus_chunk_size",
        str(runtime["corpus_chunk_size"]),
        "--top_k",
        str(runtime["top_k"]),
        "--cache_dir",
        str(runtime["bright_cache_dir"]),
        "--output_json",
        str(runtime["eval_output_json"]),
        "--report-to",
        "none",
    ]


def cleanup_after_training() -> None:
    """Free Python and CUDA memory before evaluation starts."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def log_bright_results(
    run: wandb.sdk.wandb_run.Run,
    *,
    runtime: dict[str, Any],
    output: dict[str, Any],
) -> None:
    """Log BRIGHT summary and per-task metrics to the active W&B run."""

    summary_metrics = output["summary_ndcg10_percent"]
    task_metrics = output["task_metrics"]
    payload: dict[str, Any] = {
        "bright/progress/completed_tasks": len(task_metrics),
        "bright/summary/full_mean": summary_metrics["Full mean"],
    }
    for summary_name, summary_value in summary_metrics.items():
        payload[f"bright/summary/{normalize_metric_name(summary_name)}"] = summary_value
    for task, metrics in task_metrics.items():
        for metric_name, metric_value in metrics.items():
            payload[
                f"bright/task/{task}/{normalize_metric_name(metric_name)}"
            ] = metric_value
        payload[f"bright/task/{task}/ndcg_at_10_percent"] = round(
            metrics["NDCG@10"] * 100,
            2,
        )

    wandb.log(payload)
    run.summary["paths/output_dir"] = str(runtime["output_dir"])
    run.summary["paths/final_model"] = str(runtime["output_dir"] / "final")
    run.summary["paths/bright_output_json"] = str(runtime["eval_output_json"])
    run.summary["bright/reasoning"] = output["reasoning"]
    run.summary["bright/document_length"] = output["document_length"]
    run.summary["bright/task_count"] = len(task_metrics)


def main() -> None:
    """Execute one sweep trial: train, evaluate, and log the BRIGHT metric."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        job_type="reasonir-hq-sweep",
        config=base_wandb_config(args),
    )
    if run is None:
        raise RuntimeError("wandb.init returned None.")

    try:
        runtime = resolve_runtime_configuration(config=wandb.config, args=args)
        wandb.config.update(
            {key: serialize_runtime_value(value) for key, value in runtime.items()},
            allow_val_change=True,
        )
        run.name = str(runtime["run_name"])
        run.define_metric("bright/summary/full_mean", summary="max")

        if args.dry_run:
            print(json.dumps(runtime, indent=2, default=str))
            return

        LOGGER.info("Starting sweep run %s", runtime["run_name"])
        LOGGER.info("Resolved runtime configuration: %s", runtime)
        Path(runtime["output_dir"]).mkdir(parents=True, exist_ok=True)

        run_script_main(
            module_path=Path(runtime["train_script"]),
            module_name="reasonir_hq_train_module",
            argv=build_training_argv(runtime=runtime),
        )

        cleanup_after_training()

        run_script_main(
            module_path=Path(runtime["eval_script"]),
            module_name="reasonir_hq_eval_module",
            argv=build_eval_argv(runtime=runtime),
        )

        output = json.loads(Path(runtime["eval_output_json"]).read_text())
        log_bright_results(run=run, runtime=runtime, output=output)
        print(json.dumps(output["summary_ndcg10_percent"], indent=2, sort_keys=True))
    finally:
        run.finish()


if __name__ == "__main__":
    main()
