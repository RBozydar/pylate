from __future__ import annotations

import argparse
import configparser
import json
import math
import os
import sys
from netrc import netrc
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import wandb

DEFAULT_PROJECT = "ColBERT-Zero"
DEFAULT_ENTITY = "rbw"
DEFAULT_CONFIG_KEYS = (
    "run_name",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "learning_rate",
    "max_steps",
    "warmup_ratio",
    "temperature",
    "temp",
)
DEFAULT_SUMMARY_KEYS = (
    "train_loss",
    "epoch",
    "global_step",
    "train_runtime",
    "train_steps_per_second",
    "train_samples_per_second",
)
DEFAULT_HISTORY_KEYS = (
    "_step",
    "_runtime",
    "_timestamp",
    "loss",
    "train_loss",
    "grad_norm",
    "learning_rate",
    "eval_loss",
    "accuracy",
)
SETTINGS_PATHS = (
    Path.home() / ".config" / "wandb" / "settings",
    Path.home() / ".wandb" / "settings",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for W&B project run summaries."""

    parser = argparse.ArgumentParser(
        description=(
            "Summarize recent W&B runs for a project as compact JSON. Supports "
            "self-hosted W&B instances via --base-url or WANDB_BASE_URL."
        )
    )
    parser.add_argument(
        "--project",
        type=str,
        default=DEFAULT_PROJECT,
        help=f"W&B project to inspect. Default: {DEFAULT_PROJECT}.",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=DEFAULT_ENTITY,
        help=f"W&B entity/user to inspect. Default: {DEFAULT_ENTITY}.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help=(
            "Explicit W&B base URL. If omitted, the script checks "
            "WANDB_BASE_URL and local W&B settings."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of runs to return. Default: 10.",
    )
    parser.add_argument(
        "--state",
        type=str,
        default=None,
        help="Optional run state filter such as running, finished, or killed.",
    )
    parser.add_argument(
        "--name-contains",
        type=str,
        default=None,
        help="Optional substring filter applied to run names.",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Optional exact-match W&B group filter.",
    )
    parser.add_argument(
        "--config-keys",
        type=str,
        default=",".join(DEFAULT_CONFIG_KEYS),
        help="Comma-separated config keys to include in each run summary.",
    )
    parser.add_argument(
        "--summary-keys",
        type=str,
        default=",".join(DEFAULT_SUMMARY_KEYS),
        help="Comma-separated summary keys to include in each run summary.",
    )
    parser.add_argument(
        "--history-keys",
        type=str,
        default=",".join(DEFAULT_HISTORY_KEYS),
        help="Comma-separated history keys to inspect when --history-tail > 0.",
    )
    parser.add_argument(
        "--history-tail",
        type=int,
        default=0,
        help="Include the last N history rows per run. Default: 0.",
    )
    return parser.parse_args()


def parse_csv_values(raw_value: str) -> tuple[str, ...]:
    """Parse a comma-separated CLI argument into a normalized tuple."""

    return tuple(value.strip() for value in raw_value.split(",") if value.strip())


def read_base_url_from_settings(settings_path: Path) -> str | None:
    """Read a W&B base URL from a local settings file if present."""

    if not settings_path.exists():
        return None

    parser = configparser.ConfigParser()
    parser.read(settings_path, encoding="utf-8")
    if parser.has_section("default"):
        return parser.get("default", "base_url", fallback=None)
    if parser.default_section in parser:
        return parser.get(parser.default_section, "base_url", fallback=None)
    return None


def resolve_base_url(explicit_base_url: str | None) -> str | None:
    """Resolve the W&B base URL from CLI, env, or local settings."""

    if explicit_base_url:
        return explicit_base_url

    environment_base_url = os.environ.get("WANDB_BASE_URL")
    if environment_base_url:
        return environment_base_url

    for settings_path in SETTINGS_PATHS:
        parsed_base_url = read_base_url_from_settings(settings_path)
        if parsed_base_url:
            return parsed_base_url

    return None


def netrc_has_credentials(base_url: str | None) -> bool | None:
    """Check whether .netrc contains credentials for the resolved W&B host."""

    netrc_path = Path.home() / ".netrc"
    if base_url is None or not netrc_path.exists():
        return None

    host = urlparse(base_url).netloc
    try:
        auth_data = netrc(str(netrc_path)).authenticators(host)
    except OSError:
        return None

    if auth_data is None:
        return False
    login, _, password = auth_data
    return bool(login and password)


def normalize_value(value: Any) -> Any:
    """Normalize API values into JSON-safe output."""

    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def maybe_matches_filters(
    run: Any,
    *,
    state: str | None,
    name_contains: str | None,
    group: str | None,
) -> bool:
    """Return whether a run matches the requested filters."""

    if state is not None and str(getattr(run, "state", "")).lower() != state.lower():
        return False
    if name_contains is not None and name_contains.lower() not in str(
        getattr(run, "name", "")
    ).lower():
        return False
    if group is not None and str(getattr(run, "group", "")) != group:
        return False
    return True


def extract_selected_mapping(
    mapping: dict[str, Any],
    keys: tuple[str, ...],
) -> dict[str, Any]:
    """Extract selected keys from a mapping while dropping null values."""

    result: dict[str, Any] = {}
    for key in keys:
        value = normalize_value(mapping.get(key))
        if value is not None:
            result[key] = value
    return result


def extract_history_tail(
    run: Any,
    *,
    history_keys: tuple[str, ...],
    history_tail: int,
) -> list[dict[str, Any]]:
    """Extract the last N non-empty history rows for a run."""

    if history_tail <= 0:
        return []

    rows: list[dict[str, Any]] = []
    for row in run.scan_history(page_size=max(100, history_tail * 4)):
        cleaned_row: dict[str, Any] = {}
        for key in history_keys:
            value = normalize_value(row.get(key))
            if value is not None:
                cleaned_row[key] = value
        if cleaned_row:
            rows.append(cleaned_row)
    return rows[-history_tail:]


def summarize_run(
    run: Any,
    *,
    config_keys: tuple[str, ...],
    summary_keys: tuple[str, ...],
    history_keys: tuple[str, ...],
    history_tail: int,
) -> dict[str, Any]:
    """Build a compact JSON-serializable summary for a W&B run."""

    config = extract_selected_mapping(dict(run.config), config_keys)
    summary = extract_selected_mapping(dict(run.summary), summary_keys)
    output: dict[str, Any] = {
        "name": run.name,
        "id": run.id,
        "state": run.state,
        "group": run.group,
        "job_type": run.job_type,
        "created_at": str(run.created_at) if getattr(run, "created_at", None) else None,
        "url": run.url,
        "config": config,
        "summary": summary,
    }
    history = extract_history_tail(
        run,
        history_keys=history_keys,
        history_tail=history_tail,
    )
    if history:
        output["history_tail"] = history
    return output


def main() -> None:
    """Query W&B and write a compact project summary to stdout as JSON."""

    args = parse_args()
    base_url = resolve_base_url(args.base_url)
    if base_url:
        os.environ["WANDB_BASE_URL"] = base_url

    api = wandb.Api()
    config_keys = parse_csv_values(args.config_keys)
    summary_keys = parse_csv_values(args.summary_keys)
    history_keys = parse_csv_values(args.history_keys)

    runs = api.runs(
        f"{args.entity}/{args.project}",
        per_page=max(args.limit * 2, args.limit),
    )
    selected_runs: list[dict[str, Any]] = []
    for run in runs:
        if not maybe_matches_filters(
            run,
            state=args.state,
            name_contains=args.name_contains,
            group=args.group,
        ):
            continue
        selected_runs.append(
            summarize_run(
                run,
                config_keys=config_keys,
                summary_keys=summary_keys,
                history_keys=history_keys,
                history_tail=args.history_tail,
            )
        )
        if len(selected_runs) >= args.limit:
            break

    payload = {
        "base_url": base_url,
        "credentials_present": netrc_has_credentials(base_url),
        "entity": args.entity,
        "project": args.project,
        "returned_runs": len(selected_runs),
        "filters": {
            "state": args.state,
            "name_contains": args.name_contains,
            "group": args.group,
            "history_tail": args.history_tail,
        },
        "runs": selected_runs,
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
