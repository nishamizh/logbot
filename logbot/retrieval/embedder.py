"""
logbot/retrieval/embedder.py
──────────────────────────────
Converts structured log entries into dense vector embeddings using
HuggingFace sentence-transformers.

Design decisions (interview-ready talking points):
  • LogEmbedder.embed_entries() is the single public API — callers pass
    structured dicts, get back numpy arrays. No raw text handling outside.
  • Text template is explicit and documented — the LLM/retrieval quality
    depends entirely on what text we embed. template() is overridable.
  • Batch encoding with show_progress_bar=False — silent in production,
    no tqdm noise in logs.
  • Model is lazy-loaded on first call (not at import time) — startup is
    fast even if the model is large.
  • EmbedderStats tracks throughput and timing for monitoring.
  • Stub mode (model=None) returns deterministic random vectors — lets
    the full pipeline run in CI without downloading a model.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from logbot.core.config import EmbeddingModel, get_settings
from logbot.core.logging import TimedBlock, get_logger

log = get_logger(__name__, component="embedder")


# ──────────────────────────────────────────────────────────────────────────────
# Text template
# ──────────────────────────────────────────────────────────────────────────────

def default_log_template(entry: Dict[str, Any]) -> str:
    """
    Convert a structured log entry to the text string that gets embedded.

    Template design matters for retrieval quality:
      - Include level so 'find all ERROR logs' works
      - Include service so 'find auth service errors' works
      - Keep it short — sentence-transformers work best under ~256 tokens
    """
    level   = entry.get("level",   "INFO")
    service = entry.get("service", "unknown")
    message = entry.get("message", "")
    return f"[{level}] {service}: {message}"


# ──────────────────────────────────────────────────────────────────────────────
# Stats
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EmbedderStats:
    total_embedded: int   = 0
    total_batches:  int   = 0
    total_ms:       float = 0.0
    model_name:     str   = ""

    @property
    def avg_ms_per_entry(self) -> float:
        if self.total_embedded == 0:
            return 0.0
        return round(self.total_ms / self.total_embedded, 3)

    @property
    def throughput_per_sec(self) -> float:
        if self.total_ms == 0:
            return 0.0
        return round(self.total_embedded / (self.total_ms / 1000), 1)


# ──────────────────────────────────────────────────────────────────────────────
# LogEmbedder
# ──────────────────────────────────────────────────────────────────────────────

class LogEmbedder:
    """
    Embeds log entries using a sentence-transformer model.

    Usage:
        embedder = LogEmbedder()                        # uses config defaults
        vectors, texts = embedder.embed_entries(entries)
        # vectors: np.ndarray shape (n, embedding_dim)
        # texts:   list of strings that were embedded

        # Stub mode (no model download):
        embedder = LogEmbedder(stub=True)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device:     Optional[str] = None,
        stub:       bool          = False,
        template_fn = None,
    ) -> None:
        cfg              = get_settings().llm
        self._model_name = model_name or cfg.embedding_model.value
        self._device     = device     or cfg.device
        self._dim        = cfg.embedding_dim
        self._stub       = stub
        self._template   = template_fn or default_log_template
        self._model      = None        # lazy-loaded
        self._stats      = EmbedderStats(model_name=self._model_name)

        log.info("embedder_created",
                 model=self._model_name,
                 device=self._device,
                 stub=stub)

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Lazy-load the sentence-transformer model."""
        if self._model is not None:
            return
        if self._stub:
            log.info("embedder_stub_mode")
            return

        try:
            from sentence_transformers import SentenceTransformer
            with TimedBlock("model_load", logger=log,
                            extra={"model": self._model_name}):
                self._model = SentenceTransformer(
                    self._model_name,
                    device=self._device,
                )
            self._dim = self._model.get_sentence_embedding_dimension()
            log.info("model_loaded",
                     model=self._model_name,
                     dim=self._dim,
                     device=self._device)
        except ImportError:
            log.warning("sentence_transformers_not_installed_using_stub")
            self._stub = True
        except Exception as exc:
            log.error("model_load_failed", error=str(exc))
            log.warning("falling_back_to_stub_mode")
            self._stub = True

    # ── Embedding ─────────────────────────────────────────────────────────────

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of text strings.
        Returns np.ndarray of shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)

        self._load_model()
        t0 = time.perf_counter()

        if self._stub:
            vectors = self._stub_embed(texts)
        else:
            vectors = np.array(
                self._model.encode(
                    texts,
                    batch_size=64,
                    show_progress_bar=False,
                    normalize_embeddings=True,   # cosine similarity ready
                ),
                dtype=np.float32,
            )

        elapsed = (time.perf_counter() - t0) * 1000
        self._stats.total_embedded += len(texts)
        self._stats.total_batches  += 1
        self._stats.total_ms       += elapsed

        log.debug("texts_embedded",
                  count=len(texts),
                  dim=vectors.shape[1] if vectors.ndim > 1 else self._dim,
                  elapsed_ms=round(elapsed, 2))
        return vectors

    def embed_entries(
        self,
        entries: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Embed a list of structured log entries.

        Returns:
            vectors — shape (n, embedding_dim)
            texts   — the text strings that were embedded (for debugging)
        """
        texts   = [self._template(e) for e in entries]
        vectors = self.embed_texts(texts)
        return vectors, texts

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string for similarity search.
        Returns shape (embedding_dim,) — 1D vector.
        """
        vectors = self.embed_texts([query])
        return vectors[0]

    # ── Stub embedding ────────────────────────────────────────────────────────

    def _stub_embed(self, texts: List[str]) -> np.ndarray:
        """
        Deterministic pseudo-embeddings for tests/CI.
        Same text → same vector (via SHA-256 seed).
        Normalised to unit length so cosine similarity works.
        """
        vectors = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            rng  = np.random.default_rng(seed)
            vec  = rng.standard_normal(self._dim).astype(np.float32)
            vec  = vec / (np.linalg.norm(vec) + 1e-9)
            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_stub(self) -> bool:
        return self._stub

    @property
    def stats(self) -> EmbedderStats:
        return self._stats


# ──────────────────────────────────────────────────────────────────────────────
# Module-level convenience
# ──────────────────────────────────────────────────────────────────────────────

_embedder_instance: Optional[LogEmbedder] = None


def get_embedder(stub: bool = False) -> LogEmbedder:
    """Return process-wide LogEmbedder singleton."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = LogEmbedder(stub=stub)
    return _embedder_instance


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test  →  python -m logbot.retrieval.embedder
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from logbot.core.logging import configure_logging
    configure_logging()
    log.info("smoke_test_start")

    # Use stub mode — no model download needed
    embedder = LogEmbedder(stub=True)

    # ── 1. embed_texts ────────────────────────────────────────────────────────
    texts = [
        "[ERROR] payments: DB connection timeout",
        "[INFO] auth: Login successful",
        "[CRITICAL] payments: Circuit breaker OPEN",
    ]
    vectors = embedder.embed_texts(texts)
    assert vectors.shape == (3, embedder.dim)
    assert vectors.dtype == np.float32
    print(f"✅  embed_texts: shape={vectors.shape} dtype={vectors.dtype}")

    # ── 2. Deterministic — same text → same vector ────────────────────────────
    v1 = embedder.embed_texts(["[ERROR] payments: timeout"])
    v2 = embedder.embed_texts(["[ERROR] payments: timeout"])
    assert np.allclose(v1, v2), "Same text must produce same vector"
    print("✅  Deterministic embeddings (same text → same vector)")

    # ── 3. Different texts → different vectors ────────────────────────────────
    va = embedder.embed_texts(["[ERROR] service-a: timeout"])
    vb = embedder.embed_texts(["[INFO] service-b: started"])
    assert not np.allclose(va, vb), "Different texts must differ"
    print("✅  Different texts produce different vectors")

    # ── 4. Unit normalisation ─────────────────────────────────────────────────
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"Vectors not unit-normalised: {norms}"
    print(f"✅  Unit-normalised vectors (norms ≈ 1.0)")

    # ── 5. embed_entries ──────────────────────────────────────────────────────
    entries = [
        {"level": "ERROR",    "service": "payments", "message": "DB timeout"},
        {"level": "INFO",     "service": "auth",     "message": "Login OK"},
        {"level": "CRITICAL", "service": "payments", "message": "Circuit breaker OPEN"},
    ]
    vecs, texts_out = embedder.embed_entries(entries)
    assert vecs.shape == (3, embedder.dim)
    assert len(texts_out) == 3
    assert "[ERROR]" in texts_out[0]
    assert "payments" in texts_out[0]
    print(f"✅  embed_entries: shape={vecs.shape}")
    print(f"     sample text: '{texts_out[0]}'")

    # ── 6. embed_query ────────────────────────────────────────────────────────
    q_vec = embedder.embed_query("payment service errors")
    assert q_vec.shape == (embedder.dim,)
    assert abs(np.linalg.norm(q_vec) - 1.0) < 1e-5
    print(f"✅  embed_query: shape={q_vec.shape}")

    # ── 7. Empty input ────────────────────────────────────────────────────────
    empty = embedder.embed_texts([])
    assert empty.shape == (0, embedder.dim)
    print("✅  Empty input returns (0, dim) array")

    # ── 8. Stats ──────────────────────────────────────────────────────────────
    stats = embedder.stats
    assert stats.total_embedded > 0
    assert stats.total_batches  > 0
    print(f"✅  Stats: {stats.total_embedded} embedded, "
          f"{stats.avg_ms_per_entry}ms/entry, "
          f"{stats.throughput_per_sec} entries/sec")

    # ── 9. Cosine similarity — similar texts closer than dissimilar ───────────
    v_err1 = embedder.embed_query("[ERROR] payments: DB connection timeout after 30s")
    v_err2 = embedder.embed_query("[ERROR] payments: DB connection pool exhausted")
    v_info = embedder.embed_query("[INFO] auth: User login successful")

    sim_errors = float(np.dot(v_err1, v_err2))
    sim_diff   = float(np.dot(v_err1, v_info))
    # Stub uses hash-based vectors so similarity is random — just check it runs
    print(f"✅  Cosine sim (similar errors): {sim_errors:.4f}")
    print(f"   Cosine sim (error vs info):   {sim_diff:.4f}")

    print(f"\n✅  All embedder.py smoke-tests passed.")
