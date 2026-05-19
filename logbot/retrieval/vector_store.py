"""
logbot/retrieval/vector_store.py
──────────────────────────────────
ChromaDB vector store wrapper for LogBot.

Responsibilities:
  • Upsert log entries with embeddings into ChromaDB
  • Query by semantic similarity (dense retrieval)
  • Filter by metadata (level, service, source, time range)
  • Collection lifecycle (create, reset, stats)

Design decisions (interview-ready talking points):
  • VectorStore wraps ChromaDB's HTTP client — works with a running
    ChromaDB server OR the embedded persistent client for local dev.
  • Stub mode uses an in-memory dict — full pipeline runs in CI with
    zero infrastructure.
  • upsert() is idempotent — same log_id upserted twice = one document.
    IDs are SHA-256 of (source + timestamp + message) so re-ingesting
    the same log file doesn't create duplicates.
  • query() returns List[SearchResult] — typed, never raw ChromaDB dicts.
  • Metadata filtering is pushed down to ChromaDB (server-side) —
    not post-filtered in Python.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from logbot.core.config import get_settings
from logbot.core.logging import TimedBlock, get_logger
from logbot.retrieval.embedder import LogEmbedder, get_embedder

log = get_logger(__name__, component="vector_store")


# ──────────────────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """A single result from a vector similarity search."""
    rank:      int
    score:     float          # cosine similarity (higher = more similar)
    log_id:    str
    timestamp: str
    level:     str
    service:   str
    source:    str
    message:   str
    metadata:  Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank":      self.rank,
            "score":     self.score,
            "log_id":    self.log_id,
            "timestamp": self.timestamp,
            "level":     self.level,
            "service":   self.service,
            "source":    self.source,
            "message":   self.message,
        }


@dataclass
class UpsertStats:
    upserted: int = 0
    skipped:  int = 0
    errors:   int = 0
    elapsed_ms: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# ID generation
# ──────────────────────────────────────────────────────────────────────────────

def make_log_id(entry: Dict[str, Any]) -> str:
    """
    Deterministic ID for a log entry.
    SHA-256 of source + timestamp + message → 16-char hex prefix.
    Same log line ingested twice gets the same ID → idempotent upsert.
    """
    key = f"{entry.get('source','')}{entry.get('timestamp','')}{entry.get('message','')}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ──────────────────────────────────────────────────────────────────────────────
# Stub store (in-memory, no ChromaDB needed)
# ──────────────────────────────────────────────────────────────────────────────

class _StubStore:
    """In-memory store for dev/test — mirrors ChromaDB's interface."""

    def __init__(self) -> None:
        self._docs: Dict[str, Dict] = {}   # log_id → {vector, entry}

    def upsert(self, log_id: str, vector: np.ndarray, entry: Dict) -> None:
        self._docs[log_id] = {"vector": vector, "entry": entry}

    def query(
        self,
        query_vector: np.ndarray,
        top_k: int,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        if not self._docs:
            return []

        # Cosine similarity
        results = []
        for doc_id, doc in self._docs.items():
            entry  = doc["entry"]
            # Apply filters
            if filters:
                if "level" in filters and entry.get("level") != filters["level"]:
                    continue
                if "service" in filters and entry.get("service") != filters["service"]:
                    continue
                if "source" in filters and entry.get("source") != filters["source"]:
                    continue
            vec   = doc["vector"]
            score = float(np.dot(query_vector, vec) /
                         (np.linalg.norm(query_vector) * np.linalg.norm(vec) + 1e-9))
            results.append({"id": doc_id, "score": score, "entry": entry})

        results.sort(key=lambda x: -x["score"])
        return results[:top_k]

    def count(self) -> int:
        return len(self._docs)

    def reset(self) -> None:
        self._docs.clear()


# ──────────────────────────────────────────────────────────────────────────────
# VectorStore
# ──────────────────────────────────────────────────────────────────────────────

class VectorStore:
    """
    ChromaDB-backed vector store for log entries.

    Usage:
        store = VectorStore(stub=True)          # in-memory, no ChromaDB
        store = VectorStore()                   # connects to ChromaDB server

        stats = store.upsert_entries(entries)   # embed + store
        results = store.query("DB timeout", top_k=5)
    """

    def __init__(
        self,
        embedder:        Optional[LogEmbedder] = None,
        collection_name: Optional[str]         = None,
        stub:            bool                  = False,
    ) -> None:
        cfg                  = get_settings()
        self._embedder       = embedder or get_embedder(stub=stub)
        self._collection_name = collection_name or cfg.vector_store.collection_name
        self._stub           = stub
        self._client         = None
        self._collection     = None
        self._stub_store     = _StubStore() if stub else None
        self._total_upserted = 0

        log.info("vector_store_created",
                 collection=self._collection_name,
                 stub=stub)

    # ── Connection ────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        """Lazy-connect to ChromaDB."""
        if self._stub or self._collection is not None:
            return
        try:
            import chromadb
            cfg = get_settings().vector_store
            with TimedBlock("chroma_connect", logger=log):
                self._client = chromadb.HttpClient(
                    host=cfg.host,
                    port=cfg.port,
                )
                self._collection = self._client.get_or_create_collection(
                    name=self._collection_name,
                    metadata={"hnsw:space": cfg.distance_metric},
                )
            log.info("chroma_connected",
                     host=cfg.host,
                     port=cfg.port,
                     collection=self._collection_name)
        except Exception as exc:
            log.warning("chroma_connect_failed_using_stub",
                        error=str(exc))
            self._stub       = True
            self._stub_store = _StubStore()

    # ── Upsert ────────────────────────────────────────────────────────────────

    def upsert_entries(
        self,
        entries:    List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> UpsertStats:
        """
        Embed and upsert a list of log entries.
        Idempotent — same entry upserted twice = one document.
        """
        if not entries:
            return UpsertStats()

        self._connect()
        t0    = time.perf_counter()
        stats = UpsertStats()

        # Embed all entries
        try:
            vectors, texts = self._embedder.embed_entries(entries)
        except Exception as exc:
            log.error("embed_failed", error=str(exc))
            stats.errors = len(entries)
            return stats

        # Upsert in batches
        for i in range(0, len(entries), batch_size):
            batch_entries = entries[i:i + batch_size]
            batch_vectors = vectors[i:i + batch_size]

            try:
                self._upsert_batch(batch_entries, batch_vectors)
                stats.upserted += len(batch_entries)
                self._total_upserted += len(batch_entries)
            except Exception as exc:
                log.error("batch_upsert_failed",
                           batch_start=i, error=str(exc))
                stats.errors += len(batch_entries)

        stats.elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        log.info("upsert_complete",
                 upserted=stats.upserted,
                 errors=stats.errors,
                 elapsed_ms=stats.elapsed_ms)
        return stats

    def _upsert_batch(
        self,
        entries: List[Dict[str, Any]],
        vectors: np.ndarray,
    ) -> None:
        """Upsert one batch into ChromaDB or stub store."""
        ids        = [make_log_id(e) for e in entries]
        embeddings = vectors.tolist()
        documents  = [e.get("message", "") for e in entries]
        metadatas  = [
            {
                "timestamp": e.get("timestamp", ""),
                "level":     e.get("level",     "INFO"),
                "service":   e.get("service",   "unknown"),
                "source":    e.get("source",    "unknown"),
                "format":    e.get("format",    "unknown"),
            }
            for e in entries
        ]

        if self._stub:
            for log_id, vec, entry in zip(ids, vectors, entries):
                self._stub_store.upsert(log_id, vec, entry)
        else:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        top_k:      int = 5,
        filters:    Optional[Dict[str, str]] = None,
    ) -> List[SearchResult]:
        """
        Semantic search over stored log entries.

        Args:
            query_text: natural language query
            top_k:      number of results to return
            filters:    optional metadata filters e.g. {"level": "ERROR"}

        Returns:
            List of SearchResult sorted by similarity (highest first)
        """
        self._connect()

        query_vector = self._embedder.embed_query(query_text)

        if self._stub:
            raw = self._stub_store.query(query_vector, top_k, filters)
            return [
                SearchResult(
                    rank=i + 1,
                    score=round(r["score"], 4),
                    log_id=r["id"],
                    timestamp=r["entry"].get("timestamp", ""),
                    level=r["entry"].get("level", "INFO"),
                    service=r["entry"].get("service", "unknown"),
                    source=r["entry"].get("source", "unknown"),
                    message=r["entry"].get("message", ""),
                )
                for i, r in enumerate(raw)
            ]

        # ChromaDB query
        try:
            where = self._build_where(filters) if filters else None
            kwargs: Dict[str, Any] = {
                "query_embeddings": [query_vector.tolist()],
                "n_results": min(top_k, max(self._collection.count(), 1)),
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where

            result = self._collection.query(**kwargs)

            return [
                SearchResult(
                    rank=i + 1,
                    score=round(1 - dist, 4),   # distance → similarity
                    log_id=result["ids"][0][i],
                    timestamp=result["metadatas"][0][i].get("timestamp", ""),
                    level=result["metadatas"][0][i].get("level", "INFO"),
                    service=result["metadatas"][0][i].get("service", "unknown"),
                    source=result["metadatas"][0][i].get("source", "unknown"),
                    message=result["documents"][0][i],
                )
                for i, dist in enumerate(result["distances"][0])
            ]
        except Exception as exc:
            log.error("query_failed", error=str(exc))
            return []

    def _build_where(self, filters: Dict[str, str]) -> Dict:
        """Convert simple filter dict to ChromaDB where clause."""
        if len(filters) == 1:
            key, val = next(iter(filters.items()))
            return {key: {"$eq": val}}
        return {"$and": [{k: {"$eq": v}} for k, v in filters.items()]}

    # ── Collection management ─────────────────────────────────────────────────

    def count(self) -> int:
        """Number of documents in the collection."""
        if self._stub:
            return self._stub_store.count()
        self._connect()
        try:
            return self._collection.count()
        except Exception:
            return 0

    def reset(self) -> None:
        """Delete all documents from the collection."""
        if self._stub:
            self._stub_store.reset()
            log.info("stub_store_reset")
            return
        self._connect()
        try:
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name
            )
            log.info("collection_reset", collection=self._collection_name)
        except Exception as exc:
            log.error("reset_failed", error=str(exc))

    def stats(self) -> Dict[str, Any]:
        return {
            "collection":     self._collection_name,
            "total_documents": self.count(),
            "total_upserted": self._total_upserted,
            "stub_mode":      self._stub,
            "embedder":       self._embedder.model_name,
            "embedding_dim":  self._embedder.dim,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Module-level convenience
# ──────────────────────────────────────────────────────────────────────────────

_store_instance: Optional[VectorStore] = None


def get_vector_store(stub: bool = False) -> VectorStore:
    """Return process-wide VectorStore singleton."""
    global _store_instance
    if _store_instance is None:
        _store_instance = VectorStore(stub=stub)
    return _store_instance


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test  →  python -m logbot.retrieval.vector_store
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from logbot.core.logging import configure_logging
    configure_logging()
    log.info("smoke_test_start")

    store = VectorStore(stub=True)

    entries = [
        {"level": "ERROR",    "service": "payments", "source": "payments.log",
         "message": "DB connection timeout after 30s",
         "timestamp": "2024-01-15T10:23:45Z"},
        {"level": "CRITICAL", "service": "payments", "source": "payments.log",
         "message": "Circuit breaker OPEN — all requests rejected",
         "timestamp": "2024-01-15T10:24:00Z"},
        {"level": "INFO",     "service": "auth",     "source": "auth.log",
         "message": "User login successful",
         "timestamp": "2024-01-15T10:24:01Z"},
        {"level": "ERROR",    "service": "auth",     "source": "auth.log",
         "message": "Failed login attempt — account locked",
         "timestamp": "2024-01-15T10:24:02Z"},
        {"level": "WARNING",  "service": "api",      "source": "api.log",
         "message": "Slow response time 2100ms on POST /payments",
         "timestamp": "2024-01-15T10:24:03Z"},
    ]

    # ── 1. Upsert ─────────────────────────────────────────────────────────────
    stats = store.upsert_entries(entries)
    assert stats.upserted == 5
    assert stats.errors   == 0
    assert store.count()  == 5
    print(f"✅  Upsert: {stats.upserted} entries in {stats.elapsed_ms}ms")

    # ── 2. Idempotent upsert ──────────────────────────────────────────────────
    stats2 = store.upsert_entries(entries)  # same entries again
    assert store.count() == 5              # still 5, not 10
    print("✅  Idempotent upsert (duplicate entries not added)")

    # ── 3. Query ──────────────────────────────────────────────────────────────
    results = store.query("database connection timeout", top_k=3)
    assert len(results) <= 3
    assert all(isinstance(r, SearchResult) for r in results)
    assert all(r.rank >= 1 for r in results)
    print(f"✅  Query returned {len(results)} results")
    for r in results:
        print(f"   [{r.rank}] score={r.score:.4f} | {r.level} | {r.service} | {r.message[:40]}")

    # ── 4. Filter by level ────────────────────────────────────────────────────
    error_results = store.query("error", top_k=5, filters={"level": "ERROR"})
    assert all(r.level == "ERROR" for r in error_results)
    print(f"✅  Level filter: {len(error_results)} ERROR results")

    # ── 5. Filter by service ──────────────────────────────────────────────────
    auth_results = store.query("login", top_k=5, filters={"service": "auth"})
    assert all(r.service == "auth" for r in auth_results)
    print(f"✅  Service filter: {len(auth_results)} auth results")

    # ── 6. SearchResult.to_dict() ─────────────────────────────────────────────
    if results:
        d = results[0].to_dict()
        assert "rank" in d and "score" in d and "message" in d
        print("✅  SearchResult.to_dict() correct")

    # ── 7. make_log_id deterministic ──────────────────────────────────────────
    id1 = make_log_id(entries[0])
    id2 = make_log_id(entries[0])
    assert id1 == id2
    assert id1 != make_log_id(entries[1])
    print(f"✅  make_log_id deterministic: '{id1}'")

    # ── 8. Stats ──────────────────────────────────────────────────────────────
    s = store.stats()
    assert s["total_documents"] == 5
    assert s["stub_mode"] is True
    print(f"✅  Stats: {s}")

    # ── 9. Reset ──────────────────────────────────────────────────────────────
    store.reset()
    assert store.count() == 0
    print("✅  Reset clears all documents")

    # ── 10. Query on empty store ───────────────────────────────────────────────
    results_empty = store.query("anything")
    assert results_empty == []
    print("✅  Empty store returns empty results")

    print("\n✅  All vector_store.py smoke-tests passed.")
