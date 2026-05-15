"""
logbot/detection/anomaly_detector.py
──────────────────────────────────────
Orchestration layer for LogBot's anomaly detection pipeline.

Responsibilities:
  1. Feature extraction   — raw log dicts → numeric feature matrix
  2. Dual-detector run    — IsolationForest (structural) + ZScore (spike)
  3. Alert fusion         — merge scores, deduplicate, apply severity rules
  4. Alert emission       — structured Alert objects ready for API / dashboard

Design decisions (interview-ready talking points):
  • Feature engineering is explicit and documented — no silent magic.
  • Dual-detector fusion: a log window flagged by EITHER detector is an alert.
    IsolationForest catches multi-dimensional structural outliers;
    ZScore catches univariate rate/volume spikes. Together they cover both.
  • AlertBuffer is a bounded deque — O(1) append, no unbounded memory growth.
  • AnomalyDetector.analyze() is the single public entry point — callers
    never touch detectors directly.
  • All timestamps are UTC ISO-8601 strings for log-aggregator compatibility.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from logbot.core.config import get_settings
from logbot.core.logging import TimedBlock, get_logger
from logbot.detection.model_loader import (
    AnomalyScore,
    IsolationForestDetector,
    ModelRegistry,
    ZScoreDetector,
    get_model_registry,
)

log = get_logger(__name__, component="anomaly_detector")


# ──────────────────────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LogWindow:
    """
    A time-windowed batch of parsed log entries ready for feature extraction.
    Each entry is a dict with at minimum: timestamp, level, service, message.
    """
    entries:      List[Dict[str, Any]]
    window_start: str   # ISO-8601 UTC
    window_end:   str   # ISO-8601 UTC
    source:       str = "unknown"


@dataclass
class Alert:
    """
    A fused anomaly alert emitted by AnomalyDetector.
    This is what the API layer and dashboard consume.
    """
    alert_id:       str
    detected_at:    str                    # ISO-8601 UTC
    severity:       str                    # normal | warning | critical
    detectors_fired: List[str]             # which detectors flagged this
    window_start:   str
    window_end:     str
    source:         str
    anomaly_scores: List[AnomalyScore]
    features:       Dict[str, float]       # extracted feature values
    top_contributors: Dict[str, float]     # feature → contribution weight
    summary:        str                    # human-readable one-liner
    raw_entry_count: int
    error_count:    int
    critical_count: int

    @property
    def is_critical(self) -> bool:
        return self.severity == "critical"


@dataclass
class AnalysisResult:
    """Return type of AnomalyDetector.analyze()."""
    window:         LogWindow
    alerts:         List[Alert]
    features:       Dict[str, float]
    scores_if:      List[AnomalyScore]   # IsolationForest scores
    scores_zs:      List[AnomalyScore]   # ZScore scores per metric
    elapsed_ms:     float
    model_fitted:   bool

    @property
    def has_anomalies(self) -> bool:
        return len(self.alerts) > 0

    @property
    def highest_severity(self) -> str:
        if any(a.severity == "critical" for a in self.alerts):
            return "critical"
        if any(a.severity == "warning" for a in self.alerts):
            return "warning"
        return "normal"


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────

# These are the exact features the IsolationForest is trained on.
# Order matters — must match training feature_names.
FEATURE_NAMES = [
    "error_rate",       # fraction of ERROR+ entries in window
    "critical_rate",    # fraction of CRITICAL entries
    "warning_rate",     # fraction of WARNING entries
    "error_count",      # absolute ERROR+ count (scale-sensitive signal)
    "critical_count",   # absolute CRITICAL count
    "unique_services",  # number of distinct services in window
    "error_burst",      # consecutive ERROR entries at window tail
]


def extract_features(entries: List[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Convert a list of parsed log dicts into a (1, n_features) numpy array.

    Expected entry keys (all optional — missing → 0):
      level, service, message, timestamp

    Returns:
      X       — shape (1, len(FEATURE_NAMES)) for predict()
      feat_d  — same values as a named dict for logging / alerts
    """
    n = len(entries)
    if n == 0:
        zero = np.zeros((1, len(FEATURE_NAMES)))
        return zero, {k: 0.0 for k in FEATURE_NAMES}

    levels   = [str(e.get("level", "INFO")).upper() for e in entries]
    services = [str(e.get("service", "unknown"))     for e in entries]
    messages = [str(e.get("message", ""))            for e in entries]

    error_count    = sum(1 for l in levels if l in ("ERROR", "CRITICAL"))
    critical_count = sum(1 for l in levels if l == "CRITICAL")
    warning_count  = sum(1 for l in levels if l == "WARNING")

    # Error burst: count consecutive ERROR/CRITICAL from the end of the window
    burst = 0
    for l in reversed(levels):
        if l in ("ERROR", "CRITICAL"):
            burst += 1
        else:
            break

    feat_d = {
        "error_rate":      round(error_count    / n, 6),
        "critical_rate":   round(critical_count / n, 6),
        "warning_rate":    round(warning_count  / n, 6),
        "error_count":     float(error_count),
        "critical_count":  float(critical_count),
        "unique_services": float(len(set(services))),
        "error_burst":     float(burst),
    }

    X = np.array([[feat_d[k] for k in FEATURE_NAMES]], dtype=np.float64)
    return X, feat_d


# ──────────────────────────────────────────────────────────────────────────────
# Alert buffer (bounded, thread-safe via GIL + deque)
# ──────────────────────────────────────────────────────────────────────────────

class AlertBuffer:
    """
    Bounded ring-buffer of recent alerts.
    maxlen=1000 → oldest alerts auto-evicted; O(1) append.
    """

    def __init__(self, maxlen: int = 1000) -> None:
        self._buf: Deque[Alert] = deque(maxlen=maxlen)

    def append(self, alert: Alert) -> None:
        self._buf.append(alert)

    def recent(self, n: int = 50) -> List[Alert]:
        """Return the n most recent alerts, newest last."""
        items = list(self._buf)
        return items[-n:]

    def critical(self) -> List[Alert]:
        return [a for a in self._buf if a.is_critical]

    def since(self, iso_ts: str) -> List[Alert]:
        """Return alerts with detected_at >= iso_ts."""
        return [a for a in self._buf if a.detected_at >= iso_ts]

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)


# ──────────────────────────────────────────────────────────────────────────────
# AnomalyDetector
# ──────────────────────────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Main orchestrator.  One instance lives for the lifetime of the process
    (created in server.py lifespan).

    Workflow for each LogWindow:
      1. extract_features()
      2. IsolationForest.predict()   (if fitted)
      3. ZScoreDetector per metric
      4. fuse_scores() → Alert list
      5. append to AlertBuffer
    """

    def __init__(self, registry: Optional[ModelRegistry] = None) -> None:
        self._registry    = registry or get_model_registry()
        self._alert_buf   = AlertBuffer()
        self._cfg         = get_settings().anomaly
        self._window_idx  = 0
        log.info("anomaly_detector_created")

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, window: LogWindow) -> AnalysisResult:
        """
        Analyze one LogWindow. Returns AnalysisResult with all scores + alerts.
        Safe to call from async context (no blocking I/O).
        """
        t0 = time.perf_counter()
        self._window_idx += 1

        with TimedBlock("anomaly_analyze", logger=log,
                        extra={"entries": len(window.entries),
                               "source": window.source}):

            X, feat_d = extract_features(window.entries)

            # ── IsolationForest ───────────────────────────────────────────────
            ifd          = self._registry.get_isolation_forest()
            scores_if: List[AnomalyScore] = []

            if ifd.is_fitted:
                try:
                    scores_if = ifd.predict(X)
                except Exception as exc:
                    log.warning("isolation_forest_predict_failed", error=str(exc))
            else:
                log.debug("isolation_forest_not_fitted_skipping")

            # ── ZScore per metric ─────────────────────────────────────────────
            zsd       = self._registry.get_zscore_detector()
            scores_zs = self._run_zscore(feat_d, zsd)

            # ── Fuse → Alerts ─────────────────────────────────────────────────
            alerts = self._fuse_scores(
                window=window,
                feat_d=feat_d,
                scores_if=scores_if,
                scores_zs=scores_zs,
            )

            for alert in alerts:
                self._alert_buf.append(alert)
                log.warning(
                    "alert_emitted",
                    alert_id=alert.alert_id,
                    severity=alert.severity,
                    detectors=alert.detectors_fired,
                    source=alert.source,
                ) if alert.severity != "normal" else None

        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        return AnalysisResult(
            window=window,
            alerts=alerts,
            features=feat_d,
            scores_if=scores_if,
            scores_zs=scores_zs,
            elapsed_ms=elapsed,
            model_fitted=ifd.is_fitted,
        )

    def fit(self, windows: List[LogWindow]) -> None:
        """
        Fit the IsolationForest on a list of historical windows.
        Called once at startup (or on-demand retraining via API).
        Minimum 10 windows required (min_log_samples applies to raw entries,
        not windows — a window already aggregates many entries).
        """
        min_windows = max(10, self._cfg.min_log_samples // 10)
        if len(windows) < min_windows:
            raise ValueError(
                f"Need at least {min_windows} windows to fit, "
                f"got {len(windows)}"
            )

        rows = []
        for w in windows:
            X, _ = extract_features(w.entries)
            rows.append(X[0])

        X_all = np.array(rows)
        log.info("fitting_isolation_forest",
                 n_windows=len(windows), n_features=X_all.shape[1])
        self._registry.fit_isolation_forest(X_all, FEATURE_NAMES, save=True)

    # ── Alert buffer accessors ────────────────────────────────────────────────

    def recent_alerts(self, n: int = 50) -> List[Alert]:
        return self._alert_buf.recent(n)

    def critical_alerts(self) -> List[Alert]:
        return self._alert_buf.critical()

    def alerts_since(self, iso_ts: str) -> List[Alert]:
        return self._alert_buf.since(iso_ts)

    def alert_buffer_size(self) -> int:
        return len(self._alert_buf)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_zscore(
        self, feat_d: Dict[str, float], zsd: ZScoreDetector
    ) -> List[AnomalyScore]:
        """Score each metric individually through ZScoreDetector."""
        scores = []
        # Only score rate/volume metrics — not counts that naturally grow
        zscore_metrics = [
            "error_rate", "critical_rate", "warning_rate",
            "error_count", "critical_count", "error_burst",
        ]
        for i, metric in enumerate(zscore_metrics):
            value = feat_d.get(metric, 0.0)
            score = zsd.update_and_score(metric, value)
            score.index = i
            scores.append(score)
        return scores

    def _fuse_scores(
        self,
        window:    LogWindow,
        feat_d:    Dict[str, float],
        scores_if: List[AnomalyScore],
        scores_zs: List[AnomalyScore],
    ) -> List[Alert]:
        """
        Fusion rules:
          - No detectors fired → no alert.
          - Either detector fires → emit one Alert for this window.
          - Severity = max(if_severity, zs_severity).
          - top_contributors = merged feature contributions from all fired scores.
        """
        detectors_fired: List[str] = []
        all_anomalous:   List[AnomalyScore] = []

        # IsolationForest
        if_anomalies = [s for s in scores_if if s.is_anomaly]
        if if_anomalies:
            detectors_fired.append("isolation_forest")
            all_anomalous.extend(if_anomalies)

        # ZScore
        zs_anomalies = [s for s in scores_zs if s.is_anomaly]
        if zs_anomalies:
            detectors_fired.append("zscore")
            all_anomalous.extend(zs_anomalies)

        if not detectors_fired:
            return []

        # Severity: highest across all fired scores
        severity = self._max_severity(all_anomalous)

        # Only alert on configured severity levels.
        # Config uses log-level names ("ERROR","CRITICAL"); detector uses
        # ("warning","critical") — normalize both to lowercase for comparison.
        alert_levels = {s.lower() for s in self._cfg.alert_on_severity}
        # Map config level names to detector severity names
        level_map = {"error": "warning", "critical": "critical"}
        mapped_levels = {level_map.get(l, l) for l in alert_levels}
        if severity not in mapped_levels:
            return []

        # Merge feature contributions
        contributors: Dict[str, float] = {}
        for s in all_anomalous:
            for feat, weight in s.feature_contributions.items():
                contributors[feat] = max(contributors.get(feat, 0.0), weight)
        top = dict(sorted(contributors.items(), key=lambda x: -x[1])[:5])

        n            = len(window.entries)
        error_count  = sum(1 for e in window.entries
                           if str(e.get("level","")).upper() in ("ERROR","CRITICAL"))
        crit_count   = sum(1 for e in window.entries
                           if str(e.get("level","")).upper() == "CRITICAL")

        summary = (
            f"{severity.upper()} anomaly detected in '{window.source}': "
            f"{len(detectors_fired)} detector(s) fired "
            f"[{', '.join(detectors_fired)}]. "
            f"error_rate={feat_d['error_rate']:.1%}, "
            f"errors={int(feat_d['error_count'])}, "
            f"critical={int(feat_d['critical_count'])}"
        )

        alert = Alert(
            alert_id        = f"alert_{self._window_idx}_{int(time.time())}",
            detected_at     = datetime.now(timezone.utc).isoformat(),
            severity        = severity,
            detectors_fired = detectors_fired,
            window_start    = window.window_start,
            window_end      = window.window_end,
            source          = window.source,
            anomaly_scores  = all_anomalous,
            features        = feat_d,
            top_contributors= top,
            summary         = summary,
            raw_entry_count = n,
            error_count     = error_count,
            critical_count  = crit_count,
        )
        return [alert]

    @staticmethod
    def _max_severity(scores: List[AnomalyScore]) -> str:
        order = {"normal": 0, "warning": 1, "critical": 2}
        return max((s.severity for s in scores), key=lambda s: order.get(s, 0))


# ──────────────────────────────────────────────────────────────────────────────
# Module-level convenience accessor
# ──────────────────────────────────────────────────────────────────────────────

_detector_instance: Optional[AnomalyDetector] = None

def get_anomaly_detector() -> AnomalyDetector:
    """Return the process-wide AnomalyDetector singleton."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = AnomalyDetector()
    return _detector_instance


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test  →  python -m logbot.detection.anomaly_detector
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from logbot.core.logging import configure_logging

    configure_logging()
    log.info("smoke_test_start")

    import random
    random.seed(42)

    def _make_window(
        n: int,
        error_frac: float = 0.0,
        critical_frac: float = 0.0,
        source: str = "app.log",
    ) -> LogWindow:
        levels   = ["INFO"] * n
        services = ["auth", "api", "db", "cache"]
        now      = datetime.now(timezone.utc).isoformat()

        crit_end  = int(n * critical_frac)
        error_end = min(n, int(n * (critical_frac + error_frac)))
        for i in range(crit_end):
            levels[i] = "CRITICAL"
        for i in range(crit_end, error_end):
            levels[i] = "ERROR"

        entries = [
            {
                "timestamp": now,
                "level":     levels[i],
                "service":   random.choice(services),
                "message":   f"log message {i}" if levels[i] == "INFO"
                             else f"ERROR: service failure in component {i}",
            }
            for i in range(n)
        ]
        return LogWindow(
            entries=entries,
            window_start=now,
            window_end=now,
            source=source,
        )

    # ── 1. Feature extraction ─────────────────────────────────────────────────
    normal_window = _make_window(100, error_frac=0.02)
    X, feat_d = extract_features(normal_window.entries)
    print(f"\n── Extracted features ──")
    for k, v in feat_d.items():
        print(f"  {k:25s} = {v}")
    assert X.shape == (1, len(FEATURE_NAMES))
    assert 0.0 <= feat_d["error_rate"] <= 1.0
    print("✅  Feature extraction passed")

    # ── 2. Warm up ZScore with normal baseline then spike it ─────────────────
    # NOTE: IsolationForest requires diverse real training data (varied services,
    # levels, volumes). Synthetic data with constant fields (unique_services=4
    # always, warning_rate=0 always) produces zero-variance features that break
    # StandardScaler normalization. In production, fit() is called with real logs.
    # The smoke test validates ZScore detection which works on any data distribution.
    detector = AnomalyDetector()

    print("\n── Warming up ZScore with 50 normal windows ──")
    for _ in range(50):
        w = _make_window(100, error_frac=random.uniform(0.01, 0.05))
        detector.analyze(w)

    result_normal = detector.analyze(_make_window(100, error_frac=0.03))
    print(f"\n── Normal window ──")
    print(f"  has_anomalies={result_normal.has_anomalies}  "
          f"highest_severity={result_normal.highest_severity}  "
          f"elapsed_ms={result_normal.elapsed_ms}")
    assert not result_normal.has_anomalies, "Normal window should not trigger alert"

    # ── 4. ZScore spike detection ─────────────────────────────────────────────
    # After 50 normal windows (error_rate ~0.01-0.05), a window with
    # error_rate=0.9 is z >> 3.0 → guaranteed ZScore alert
    anomaly_window = _make_window(100, error_frac=0.5, critical_frac=0.4,
                                  source="payments.log")
    result_bad = detector.analyze(anomaly_window)
    print(f"\n── Anomalous window ──")
    print(f"  has_anomalies={result_bad.has_anomalies}  "
          f"highest_severity={result_bad.highest_severity}  "
          f"alerts={len(result_bad.alerts)}")
    print(f"  zscore results: {[(s.score, s.severity) for s in result_bad.scores_zs if s.is_anomaly]}")

    assert result_bad.has_anomalies, (
        f"Expected ZScore alert for error_rate=0.9 window. "
        f"ZScores: {[(s.score, s.is_anomaly) for s in result_bad.scores_zs]}"
    )
    if result_bad.alerts:
        a = result_bad.alerts[0]
        print(f"  summary: {a.summary}")
        print(f"  detectors_fired: {a.detectors_fired}")
        print(f"  top_contributors: {a.top_contributors}")

    # ── 5. Alert buffer ───────────────────────────────────────────────────────
    print(f"\n── Alert buffer size: {detector.alert_buffer_size()} ──")
    print(f"   Critical alerts: {len(detector.critical_alerts())}")
    assert detector.alert_buffer_size() > 0, "Alert buffer should have entries"

    # ── 6. Singleton ──────────────────────────────────────────────────────────
    d1 = get_anomaly_detector()
    d2 = get_anomaly_detector()
    assert d1 is d2, "get_anomaly_detector() must return singleton"

    # ── 7. IsolationForest fit API (just verify it runs, not detection accuracy) ──
    print("\n── Testing IsolationForest fit API ──")
    train_windows = [_make_window(100, error_frac=random.uniform(0.01, 0.05))
                     for _ in range(50)]
    detector.fit(train_windows)
    assert detector._registry.get_isolation_forest().is_fitted
    print("   IsolationForest fit API: ✅")

    print("\n✅  All anomaly_detector smoke-tests passed.")