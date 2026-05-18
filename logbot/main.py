"""
logbot/main.py
───────────────
LogBot application entrypoint.

Run with:
    python -m logbot.main          # starts FastAPI server
    python -m logbot.main --check  # config + dependency check only
"""

from __future__ import annotations

import argparse
import sys

from logbot.core.config import get_settings
from logbot.core.logging import configure_logging, get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description="LogBot — LLM-powered log analyzer")
    parser.add_argument("--check", action="store_true",
                        help="Run startup checks only, don't start server")
    parser.add_argument("--host", type=str, help="Override API host")
    parser.add_argument("--port", type=int, help="Override API port")
    args = parser.parse_args()

    configure_logging()
    log = get_logger(__name__, component="main")
    cfg = get_settings()

    print(cfg.service_banner)
    log.info("logbot_starting", version=cfg.app_version, env=cfg.environment.value)

    if args.check:
        log.info("startup_check_passed")
        print("\n✅  Startup check passed — all config valid.")
        sys.exit(0)

    # Start server
    from logbot.api.server import start
    start()


if __name__ == "__main__":
    main()
