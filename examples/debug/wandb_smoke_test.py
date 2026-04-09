from __future__ import annotations

import argparse
import configparser
import logging
import os
import sys
from netrc import netrc
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import wandb

LOGGER = logging.getLogger(__name__)

DEFAULT_PROJECT = "ColBERT-Zero"
DEFAULT_ENTITY = "rbw"
DEFAULT_RUN_NAME = "wandb-smoke-test"
SETTINGS_PATHS = (
    Path.home() / ".config" / "wandb" / "settings",
    Path.home() / ".wandb" / "settings",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the W&B smoke test."""

    parser = argparse.ArgumentParser(
        description=(
            "Run a minimal Weights & Biases smoke test that logs a single metric. "
            "This bypasses the training loop and helps isolate authentication or "
            "host configuration issues."
        )
    )
    parser.add_argument(
        "--project",
        type=str,
        default=DEFAULT_PROJECT,
        help=f"W&B project to write to. Default: {DEFAULT_PROJECT}.",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=DEFAULT_ENTITY,
        help=f"W&B entity/user to write to. Default: {DEFAULT_ENTITY}.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help=(
            "Explicit W&B base URL, for example http://localhost:8080. "
            "If omitted, the script checks WANDB_BASE_URL and local W&B settings."
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=DEFAULT_RUN_NAME,
        help=f"Run name to use for the smoke test. Default: {DEFAULT_RUN_NAME}.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("online", "offline", "disabled"),
        default="online",
        help="W&B mode. Default: online.",
    )
    parser.add_argument(
        "--login-timeout",
        type=int,
        default=15,
        help="Seconds to wait for login verification. Default: 15.",
    )
    parser.add_argument(
        "--relogin",
        action="store_true",
        help="Force W&B to refresh credentials before init.",
    )
    return parser.parse_args()


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


def read_base_url_from_settings(settings_path: Path) -> str | None:
    """Read `base_url` from a local W&B settings file if present."""

    if not settings_path.exists():
        return None

    parser = configparser.ConfigParser()
    parser.read(settings_path, encoding="utf-8")
    if parser.has_section("default"):
        return parser.get("default", "base_url", fallback=None)
    if parser.default_section in parser:
        return parser.get(parser.default_section, "base_url", fallback=None)
    return None


def netrc_has_credentials(base_url: str | None) -> bool | None:
    """Check whether `.netrc` contains credentials for the resolved W&B host."""

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


def build_settings_summary(base_url: str | None) -> dict[str, Any]:
    """Collect non-secret local W&B configuration details."""

    return {
        "resolved_base_url": base_url,
        "env_WANDB_BASE_URL": os.environ.get("WANDB_BASE_URL"),
        "env_WANDB_ENTITY": os.environ.get("WANDB_ENTITY"),
        "env_WANDB_PROJECT": os.environ.get("WANDB_PROJECT"),
        "netrc_credentials_present": netrc_has_credentials(base_url),
    }


def main() -> None:
    """Run the W&B smoke test."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    base_url = resolve_base_url(args.base_url)

    if base_url:
        os.environ["WANDB_BASE_URL"] = base_url

    LOGGER.info("W&B configuration summary: %s", build_settings_summary(base_url))

    try:
        login_result = wandb.login(
            host=base_url,
            relogin=args.relogin,
            timeout=args.login_timeout,
            verify=True,
        )
        LOGGER.info("wandb.login completed: %s", login_result)

        run = wandb.init(
            entity=args.entity,
            project=args.project,
            name=args.run_name,
            mode=args.mode,
            config={
                "purpose": "wandb connectivity smoke test",
                "base_url": base_url,
            },
            settings=wandb.Settings(init_timeout=args.login_timeout),
        )

        if run is None:
            raise RuntimeError("wandb.init returned None.")

        LOGGER.info("Initialized W&B run: id=%s url=%s", run.id, run.url)
        wandb.log({"smoke_test/step": 1, "smoke_test/value": 0.123})
        run.finish()
        LOGGER.info("Smoke test finished successfully.")
    except Exception:
        LOGGER.exception("W&B smoke test failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
