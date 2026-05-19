#!/usr/bin/env python3
"""
scripts/run_pipeline.py
────────────────────────
CLI to trigger the LogBot ingestion pipeline.

Usage:
    python scripts/run_pipeline.py --source data/raw/
    python scripts/run_pipeline.py --file data/raw/payments.log
    python scripts/run_pipeline.py --source data/raw/ --stub
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logbot.core.logging import configure_logging, get_logger
from logbot.retrieval.spark_pipeline import LogPipeline

log = get_logger(__name__, component="run_pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="LogBot ingestion pipeline")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", type=str, help="Directory of log files to ingest")
    group.add_argument("--file",   type=str, help="Single log file to ingest")
    parser.add_argument("--stub",   action="store_true",
                        help="Run in stub mode (no Spark/ChromaDB needed)")
    parser.add_argument("--window-size", type=int, default=100,
                        help="Anomaly detection window size (default: 100)")
    args = parser.parse_args()

    configure_logging()
    log.info("pipeline_cli_start", stub=args.stub)

    pipeline = LogPipeline(stub=args.stub, window_size=args.window_size)

    if args.source:
        stats = pipeline.run_directory(args.source)
    else:
        stats = pipeline.run_files([args.file])

    print(f"\n{'='*55}")
    print(f"  Pipeline complete")
    print(f"  {stats.summary()}")
    print(f"  Stage times: {stats.stage_times}")
    print(f"{'='*55}")

    if stats.upsert_errors > 0:
        print(f"\n⚠️  {stats.upsert_errors} upsert errors — check logs")
        sys.exit(1)


if __name__ == "__main__":
    main()
