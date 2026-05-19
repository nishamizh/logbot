#!/usr/bin/env python3
"""
scripts/load_vector_store.py
──────────────────────────────
CLI to seed ChromaDB from the synthetic anomalies.csv or any CSV of logs.

Usage:
    python scripts/load_vector_store.py                        # loads data/anomalies.csv
    python scripts/load_vector_store.py --csv data/my_logs.csv
    python scripts/load_vector_store.py --reset                # clears collection first
    python scripts/load_vector_store.py --stub                 # in-memory, no ChromaDB
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logbot.core.logging import configure_logging, get_logger
from logbot.retrieval.vector_store import VectorStore

log = get_logger(__name__, component="load_vector_store")

DEFAULT_CSV = Path(__file__).resolve().parents[1] / "data" / "anomalies.csv"


def load_csv(path: str) -> list:
    """Load log entries from a CSV file."""
    entries = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                "timestamp": row.get("timestamp", ""),
                "level":     row.get("level",     "INFO"),
                "service":   row.get("service",   "unknown"),
                "source":    row.get("source",    "unknown"),
                "message":   row.get("message",   ""),
            })
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed ChromaDB with log data")
    parser.add_argument("--csv",   type=str, default=str(DEFAULT_CSV),
                        help=f"CSV file to load (default: {DEFAULT_CSV})")
    parser.add_argument("--reset", action="store_true",
                        help="Clear the collection before loading")
    parser.add_argument("--stub",  action="store_true",
                        help="Use in-memory store (no ChromaDB needed)")
    args = parser.parse_args()

    configure_logging()
    log.info("load_vector_store_start", csv=args.csv, reset=args.reset)

    store = VectorStore(stub=args.stub)

    if args.reset:
        store.reset()
        print("🗑️  Collection cleared")

    print(f"📂  Loading: {args.csv}")
    entries = load_csv(args.csv)
    print(f"   Found {len(entries)} entries")

    stats = store.upsert_entries(entries)

    print(f"\n{'='*55}")
    print(f"  Load complete")
    print(f"  Upserted:  {stats.upserted}")
    print(f"  Errors:    {stats.errors}")
    print(f"  Elapsed:   {stats.elapsed_ms}ms")
    print(f"  Total docs: {store.count()}")
    print(f"{'='*55}")

    if stats.errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
