"""
logbot/detection/model_loader.py
─────────────────────────────────
Owns the full lifecycle of LogBot's anomaly detection models:
  • IsolationForest  — unsupervised, catches structural outliers in feature space
  • ZScoreDetector   — univariate, catches statistical spikes per metric
  • ModelRegistry    — fit / persist / load / hot-swap with zero downtime

Design decisions (interview-ready talking points):
  • joblib persistence → sklearn models serialise cleanly; versioned filenames
    prevent silent overwrites when hyperparams change.
  • ModelRegistry.get() → lazy singleton per model type; thread-safe via a lock.
  • feature_names stored alongside model → detect schema drift on load.
  • ZScoreDetector is stateful (rolling mean/std) → explicit reset() for retraining.
  • All public methods return typed dataclasses → callers never parse raw dicts.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from logbot.core.config import get_settings
from logbot.core.logging import TimedBlock, get_logger

log = get_logger(__name__, component="model_loader")


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AnomalyScore:
    """Score for a single sample."""
    index:        int
    is_anomaly:   bool
    score:        float          # raw decision function value (IsolationForest) or z-score
    severity:     str            # "normal" | "warning" | "critical"
    detector:     str            # "isolation_forest" | "zscore"
    feature_contributions: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelMetadata:
    """Persisted alongside every saved model."""
    model_type:    str
    version:       str
    trained_at:    float          # Unix timestamp
    n_samples:     int
    feature_names: List[str]
    hyperparams:   Dict
    checksum:      str            # SHA-256 of the serialised model file


# ──────────────────────────────────────────────────────────────────────────────
# IsolationForest wrapper
# ──────────────────────────────────────────────────────────────────────────────

class IsolationForestDetector:
    """
    Thin wrapper around sklearn IsolationForest with:
      - StandardScaler preprocessing (required for meaningful decision scores)
      - feature_names tracking for drift detection
      - joblib save / load with metadata sidecar
    """

    MODEL_TYPE = "isolation_forest"

    def __init__(self) -> None:
        cfg = get_settings().anomaly
        self._clf = IsolationForest(
            n_estimators=200,  # override config for better boundary learning
            contamination=cfg.isolation_forest_contamination,
            random_state=42,
            n_jobs=-1,
        )
        self._scaler:       StandardScaler   = StandardScaler()
        self._feature_names: List[str]       = []
        self._is_fitted:     bool            = False
        self._metadata:      Optional[ModelMetadata] = None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, feature_names: List[str]) -> "IsolationForestDetector":
        """
        Fit scaler + IsolationForest on training data.
        X shape: (n_samples, n_features)
        """
        if X.shape[0] < get_settings().anomaly.min_log_samples:
            raise ValueError(
                f"Need at least {get_settings().anomaly.min_log_samples} samples to fit, "
                f"got {X.shape[0]}"
            )
        if X.shape[1] != len(feature_names):
            raise ValueError(
                f"X has {X.shape[1]} columns but feature_names has {len(feature_names)}"
            )

        with TimedBlock("isolation_forest_fit", logger=log,
                        extra={"n_samples": X.shape[0], "n_features": X.shape[1]}):
            X_scaled = self._scaler.fit_transform(X)
            self._clf.fit(X_scaled)

        self._feature_names = feature_names
        self._is_fitted     = True
        log.info("isolation_forest_fitted",
                 n_samples=X.shape[0], n_features=X.shape[1])
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> List[AnomalyScore]:
        """
        Score rows in X. Returns one AnomalyScore per row.
        decision_function: negative = more anomalous.
        sklearn convention: predict() returns -1 (anomaly) or 1 (normal).
        """
        self._assert_fitted()
        if X.shape[1] != len(self._feature_names):
            raise ValueError(
                f"Schema drift: model expects {len(self._feature_names)} features, "
                f"got {X.shape[1]}"
            )

        X_scaled    = self._scaler.transform(X)
        raw_scores  = self._clf.decision_function(X_scaled)   # lower = more anomalous
        predictions = self._clf.predict(X_scaled)             # -1 or 1

        results = []
        for i, (score, pred) in enumerate(zip(raw_scores, predictions)):
            is_anomaly = pred == -1
            severity   = self._severity(score, is_anomaly)

            # Rough per-feature contribution: scaled deviation from mean
            contributions = {}
            if is_anomaly and len(self._feature_names):
                deviations = np.abs(X_scaled[i])
                total      = deviations.sum() or 1.0
                contributions = {
                    name: round(float(deviations[j] / total), 4)
                    for j, name in enumerate(self._feature_names)
                }

            results.append(AnomalyScore(
                index=i,
                is_anomaly=is_anomaly,
                score=round(float(score), 6),
                severity=severity,
                detector=self.MODEL_TYPE,
                feature_contributions=contributions,
            ))
        return results

    def _severity(self, score: float, is_anomaly: bool) -> str:
        if not is_anomaly:
            return "normal"
        if score < -0.2:
            return "critical"
        return "warning"

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: Path, version: str = "latest") -> Path:
        """Save model + scaler + metadata sidecar to directory."""
        directory.mkdir(parents=True, exist_ok=True)
        self._assert_fitted()

        model_path = directory / f"isolation_forest_{version}.joblib"
        meta_path  = directory / f"isolation_forest_{version}.meta.json"

        payload = {"clf": self._clf, "scaler": self._scaler,
                   "feature_names": self._feature_names}
        joblib.dump(payload, model_path)

        checksum = self._sha256(model_path)
        meta = ModelMetadata(
            model_type=self.MODEL_TYPE,
            version=version,
            trained_at=time.time(),
            n_samples=int(self._clf.max_samples_),   # set by sklearn after fit
            feature_names=self._feature_names,
            hyperparams={
                "n_estimators":  self._clf.n_estimators,
                "contamination": self._clf.contamination,
            },
            checksum=checksum,
        )
        meta_path.write_text(json.dumps(meta.__dict__, indent=2))

        log.info("model_saved", path=str(model_path), checksum=checksum)
        return model_path

    @classmethod
    def load(cls, directory: Path, version: str = "latest") -> "IsolationForestDetector":
        """Load model from directory; verify checksum."""
        model_path = directory / f"isolation_forest_{version}.joblib"
        meta_path  = directory / f"isolation_forest_{version}.meta.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Checksum verification
        if meta_path.exists():
            meta_raw  = json.loads(meta_path.read_text())
            expected  = meta_raw.get("checksum", "")
            actual    = cls._sha256(model_path)
            if expected and actual != expected:
                raise RuntimeError(
                    f"Model checksum mismatch! Expected {expected}, got {actual}. "
                    "File may be corrupted."
                )

        with TimedBlock("isolation_forest_load", logger=log):
            payload = joblib.load(model_path)

        instance = cls.__new__(cls)
        instance._clf           = payload["clf"]
        instance._scaler        = payload["scaler"]
        instance._feature_names = payload["feature_names"]
        instance._is_fitted     = True
        instance._metadata      = ModelMetadata(**meta_raw) if meta_path.exists() else None

        log.info("model_loaded", path=str(model_path),
                 features=instance._feature_names)
        return instance

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _assert_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()

    @property
    def feature_names(self) -> List[str]:
        return list(self._feature_names)

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


# ──────────────────────────────────────────────────────────────────────────────
# Z-Score detector (univariate, per-metric)
# ──────────────────────────────────────────────────────────────────────────────

class ZScoreDetector:
    """
    Stateful rolling Z-score detector for scalar time-series metrics.

    Maintains a running mean and std per metric name using Welford's
    online algorithm — O(1) memory, no history buffer needed.

    Usage:
        detector = ZScoreDetector()
        for value in stream:
            score = detector.update_and_score("error_rate", value)
            if score.is_anomaly:
                alert(score)
    """

    def __init__(self) -> None:
        cfg              = get_settings().anomaly
        self._threshold  = cfg.z_score_threshold
        # Per-metric Welford state: {metric: (count, mean, M2)}
        self._state: Dict[str, Tuple[int, float, float]] = {}
        self._lock        = threading.Lock()

    def update_and_score(self, metric: str, value: float) -> AnomalyScore:
        """
        Score `value` against CURRENT stats FIRST, then update Welford state.
        Updating before scoring causes spikes to dilute their own z-score.
        Thread-safe.
        """
        with self._lock:
            count, mean, M2 = self._state.get(metric, (0, 0.0, 0.0))

            if count < 2:
                # Not enough history yet — update and return normal
                count += 1
                delta  = value - mean
                mean  += delta / count
                M2    += delta * (value - mean)
                self._state[metric] = (count, mean, M2)
                return AnomalyScore(
                    index=count - 1,
                    is_anomaly=False,
                    score=0.0,
                    severity="normal",
                    detector="zscore",
                )

            # Score FIRST against existing mean/std
            variance = M2 / (count - 1)
            std      = variance ** 0.5 or 1e-9
            z        = abs(value - mean) / std

            # THEN update Welford state
            count += 1
            delta  = value - mean
            mean  += delta / count
            M2    += delta * (value - mean)
            self._state[metric] = (count, mean, M2)

        is_anomaly = z > self._threshold
        severity   = "critical" if z > self._threshold * 1.5 else \
                     "warning"  if is_anomaly else "normal"

        return AnomalyScore(
            index=count - 1,
            is_anomaly=is_anomaly,
            score=round(z, 6),
            severity=severity,
            detector="zscore",
            feature_contributions={metric: round(z, 4)},
        )

    def reset(self, metric: Optional[str] = None) -> None:
        """Reset state for one metric, or all metrics if metric=None."""
        with self._lock:
            if metric:
                self._state.pop(metric, None)
            else:
                self._state.clear()
        log.info("zscore_reset", metric=metric or "ALL")

    def stats(self) -> Dict[str, Dict]:
        """Return current running stats (for monitoring endpoints)."""
        with self._lock:
            out = {}
            for metric, (count, mean, M2) in self._state.items():
                variance = (M2 / (count - 1)) if count > 1 else 0.0
                out[metric] = {
                    "count": count,
                    "mean":  round(mean, 6),
                    "std":   round(variance ** 0.5, 6),
                }
            return out


# ──────────────────────────────────────────────────────────────────────────────
# ModelRegistry — lazy singleton, thread-safe hot-swap
# ──────────────────────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Central registry for all detection models in LogBot.

    - Lazy initialisation: models are loaded/fitted on first access.
    - Thread-safe: a per-model RLock guards get() and swap().
    - hot_swap(): atomically replaces a live model (zero-downtime retraining).

    Usage:
        registry = ModelRegistry()
        ifd = registry.get_isolation_forest()
        scores = ifd.predict(X)
    """

    _instance: Optional["ModelRegistry"] = None
    _init_lock = threading.Lock()

    def __new__(cls) -> "ModelRegistry":
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialised = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialised:
            return
        self._ifd:    Optional[IsolationForestDetector] = None
        self._zsd:    ZScoreDetector                    = ZScoreDetector()
        self._ifd_lock = threading.RLock()
        self._model_dir = Path(get_settings().data_dir) / "models"
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._initialised = True
        log.info("model_registry_created", model_dir=str(self._model_dir))

    # ── IsolationForest ───────────────────────────────────────────────────────

    def get_isolation_forest(self) -> IsolationForestDetector:
        """Return the fitted IsolationForest, loading from disk if available."""
        with self._ifd_lock:
            if self._ifd is None:
                self._ifd = self._load_or_create_ifd()
            return self._ifd

    def _load_or_create_ifd(self) -> IsolationForestDetector:
        try:
            ifd = IsolationForestDetector.load(self._model_dir)
            log.info("isolation_forest_loaded_from_disk")
            return ifd
        except FileNotFoundError:
            log.info("isolation_forest_not_on_disk_creating_blank")
            return IsolationForestDetector()

    def fit_isolation_forest(
        self, X: np.ndarray, feature_names: List[str], save: bool = True
    ) -> IsolationForestDetector:
        """Fit a new IsolationForest and optionally persist it."""
        ifd = IsolationForestDetector()
        ifd.fit(X, feature_names)
        if save:
            ifd.save(self._model_dir)
        with self._ifd_lock:
            self._ifd = ifd
        log.info("isolation_forest_fitted_and_registered")
        return ifd

    def hot_swap_isolation_forest(self, new_model: IsolationForestDetector) -> None:
        """
        Atomically swap in a newly trained model.
        In-flight predict() calls finish with the old model; new calls use the new one.
        """
        with self._ifd_lock:
            old = self._ifd
            self._ifd = new_model
        log.info("isolation_forest_hot_swapped",
                 old_fitted=old.is_fitted if old else False,
                 new_features=new_model.feature_names)

    # ── ZScoreDetector ────────────────────────────────────────────────────────

    def get_zscore_detector(self) -> ZScoreDetector:
        return self._zsd

    # ── Registry state ────────────────────────────────────────────────────────

    def status(self) -> Dict:
        return {
            "isolation_forest_fitted": self._ifd.is_fitted if self._ifd else False,
            "isolation_forest_features": self._ifd.feature_names if self._ifd else [],
            "zscore_metrics": list(self._zsd.stats().keys()),
            "model_dir": str(self._model_dir),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Module-level convenience accessor
# ──────────────────────────────────────────────────────────────────────────────

def get_model_registry() -> ModelRegistry:
    """Return the process-wide ModelRegistry singleton."""
    return ModelRegistry()


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test  →  python -m logbot.detection.model_loader
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    from logbot.core.logging import configure_logging

    configure_logging()
    log.info("smoke_test_start")

    rng = np.random.default_rng(42)

    # ── 1. IsolationForest fit + predict ──────────────────────────────────────
    features = ["error_rate", "latency_p99", "log_volume", "unique_services"]
    X_train  = rng.standard_normal((200, len(features)))

    # Inject obvious anomalies
    X_train[10]  = [10.0, 15.0, -8.0, 12.0]
    X_train[150] = [-9.0, 20.0,  9.0, -11.0]

    ifd = IsolationForestDetector()
    ifd.fit(X_train, features)

    X_test = np.vstack([
        rng.standard_normal((5, len(features))),       # normal
        np.array([[12.0, 18.0, -10.0, 15.0]]),         # obvious anomaly
    ])
    scores = ifd.predict(X_test)

    print("\n── IsolationForest predictions ──")
    for s in scores:
        print(f"  [{s.index}] anomaly={s.is_anomaly}  severity={s.severity:8s}  "
              f"score={s.score:+.4f}")

    assert any(s.is_anomaly for s in scores), "Expected at least one anomaly in test set"

    # ── 2. Save + load + checksum ─────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        saved     = ifd.save(model_dir, version="v1")
        loaded    = IsolationForestDetector.load(model_dir, version="v1")
        assert loaded.is_fitted
        assert loaded.feature_names == features
        reload_scores = loaded.predict(X_test)
        # Scores must be identical after round-trip
        for a, b in zip(scores, reload_scores):
            assert a.is_anomaly == b.is_anomaly, f"Mismatch at index {a.index}"
    print("\n✅  Save / load / checksum round-trip passed")

    # ── 3. ZScoreDetector ─────────────────────────────────────────────────────
    zsd    = ZScoreDetector()
    values = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 50.0]   # spike at end

    print("\n── ZScore predictions ──")
    for v in values:
        s = zsd.update_and_score("error_rate", v)
        print(f"  value={v:5.2f}  z={s.score:.4f}  anomaly={s.is_anomaly}  severity={s.severity}")

    # The spike at index 6 (50.0) should have been flagged as anomalous in the stream above
    spike_score = [zsd.update_and_score("error_rate", v) for v in [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 50.0]]
    assert spike_score[-1].is_anomaly, f"Spike should be anomalous, got z={spike_score[-1].score}"

    # ── 4. ModelRegistry singleton ────────────────────────────────────────────
    r1 = ModelRegistry()
    r2 = ModelRegistry()
    assert r1 is r2, "ModelRegistry must be a singleton"

    registry = get_model_registry()
    registry.fit_isolation_forest(X_train, features, save=False)
    status = registry.status()
    assert status["isolation_forest_fitted"]
    print(f"\n── Registry status ──\n  {status}")

    print("\n✅  All model_loader smoke-tests passed.")