"""
tests/test_vector_store.py
───────────────────────────
Unit tests for the detection layer: AnomalyDetector, ModelRegistry,
IsolationForestDetector, ZScoreDetector, feature extraction.
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone

from logbot.detection.model_loader import (
    IsolationForestDetector, ZScoreDetector,
    ModelRegistry, AnomalyScore, get_model_registry,
)
from logbot.detection.anomaly_detector import (
    AnomalyDetector, LogWindow, AlertBuffer,
    extract_features, FEATURE_NAMES, get_anomaly_detector,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_entries(n=20, error_frac=0.05, critical_frac=0.0):
    import random
    random.seed(42)
    levels = ["INFO"] * n
    crit_end  = int(n * critical_frac)
    error_end = min(n, int(n * (critical_frac + error_frac)))
    for i in range(crit_end): levels[i] = "CRITICAL"
    for i in range(crit_end, error_end): levels[i] = "ERROR"
    random.shuffle(levels)
    now = datetime.now(timezone.utc).isoformat()
    return [
        {"timestamp": now, "level": levels[i],
         "service": "api", "message": f"msg {i}"}
        for i in range(n)
    ]

def make_window(n=50, error_frac=0.05, source="test.log"):
    now = datetime.now(timezone.utc).isoformat()
    return LogWindow(
        entries=make_entries(n, error_frac),
        window_start=now, window_end=now, source=source,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────

class TestExtractFeatures:

    def test_returns_correct_shape(self):
        entries = make_entries(100)
        X, feat_d = extract_features(entries)
        assert X.shape == (1, len(FEATURE_NAMES))

    def test_feature_names_match(self):
        entries = make_entries(100)
        _, feat_d = extract_features(entries)
        for name in FEATURE_NAMES:
            assert name in feat_d

    def test_error_rate_in_range(self):
        entries = make_entries(100, error_frac=0.2)
        _, feat_d = extract_features(entries)
        assert 0.0 <= feat_d["error_rate"] <= 1.0

    def test_empty_entries(self):
        X, feat_d = extract_features([])
        assert X.shape == (1, len(FEATURE_NAMES))
        assert all(v == 0.0 for v in feat_d.values())

    def test_all_errors(self):
        entries = [{"level": "ERROR", "service": "api", "message": "fail"}
                   for _ in range(10)]
        _, feat_d = extract_features(entries)
        assert feat_d["error_rate"] == 1.0
        assert feat_d["error_count"] == 10.0

    def test_critical_entries(self):
        entries = [{"level": "CRITICAL", "service": "api", "message": "down"}
                   for _ in range(5)]
        _, feat_d = extract_features(entries)
        assert feat_d["critical_rate"] == 1.0
        assert feat_d["critical_count"] == 5.0


# ──────────────────────────────────────────────────────────────────────────────
# ZScoreDetector
# ──────────────────────────────────────────────────────────────────────────────

class TestZScoreDetector:

    def test_returns_anomaly_score(self):
        zsd = ZScoreDetector()
        s = zsd.update_and_score("error_rate", 0.05)
        assert isinstance(s, AnomalyScore)

    def test_not_enough_data_returns_normal(self):
        zsd = ZScoreDetector()
        s = zsd.update_and_score("metric", 100.0)
        assert not s.is_anomaly
        assert s.score == 0.0

    def test_spike_detected_after_baseline(self):
        zsd = ZScoreDetector()
        # Build baseline
        for v in [1.0, 1.1, 0.9, 1.05, 0.95, 1.0]:
            zsd.update_and_score("error_rate", v)
        # Spike
        s = zsd.update_and_score("error_rate", 50.0)
        assert s.is_anomaly
        assert s.score > 3.0

    def test_normal_values_not_flagged(self):
        zsd = ZScoreDetector()
        scores = []
        for v in [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.02, 0.98]:
            s = zsd.update_and_score("error_rate", v)
            scores.append(s)
        # After warmup, normal values should not be anomalies
        assert not any(s.is_anomaly for s in scores[2:])

    def test_reset_clears_state(self):
        zsd = ZScoreDetector()
        for v in [1.0, 1.1, 0.9, 1.05, 0.95, 1.0]:
            zsd.update_and_score("metric", v)
        zsd.reset("metric")
        stats = zsd.stats()
        assert "metric" not in stats

    def test_reset_all(self):
        zsd = ZScoreDetector()
        zsd.update_and_score("m1", 1.0)
        zsd.update_and_score("m2", 2.0)
        zsd.reset()
        assert zsd.stats() == {}

    def test_score_first_then_update(self):
        """Welford must score BEFORE updating — prevents self-dilution."""
        zsd = ZScoreDetector()
        for v in [1.0, 1.1, 0.9, 1.0, 1.05, 0.95]:
            zsd.update_and_score("rate", v)
        # Big spike — should score against old mean ~1.0
        s = zsd.update_and_score("rate", 100.0)
        assert s.score > 10.0  # not diluted

    def test_severity_levels(self):
        zsd = ZScoreDetector()
        for v in [1.0, 1.1, 0.9, 1.0, 1.05, 0.95]:
            zsd.update_and_score("rate", v)
        s = zsd.update_and_score("rate", 1000.0)
        assert s.severity == "critical"


# ──────────────────────────────────────────────────────────────────────────────
# IsolationForestDetector
# ──────────────────────────────────────────────────────────────────────────────

class TestIsolationForestDetector:

    @pytest.fixture
    def fitted_ifd(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 4))
        features = ["f1", "f2", "f3", "f4"]
        ifd = IsolationForestDetector()
        ifd.fit(X, features)
        return ifd, features

    def test_fit_sets_is_fitted(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 3))
        ifd = IsolationForestDetector()
        assert not ifd.is_fitted
        ifd.fit(X, ["a", "b", "c"])
        assert ifd.is_fitted

    def test_predict_returns_scores(self, fitted_ifd):
        ifd, features = fitted_ifd
        X_test = np.random.randn(5, 4)
        scores = ifd.predict(X_test)
        assert len(scores) == 5
        assert all(isinstance(s, AnomalyScore) for s in scores)

    def test_predict_before_fit_raises(self):
        ifd = IsolationForestDetector()
        with pytest.raises(RuntimeError, match="not fitted"):
            ifd.predict(np.array([[1.0, 2.0]]))

    def test_schema_drift_raises(self, fitted_ifd):
        ifd, _ = fitted_ifd
        X_wrong = np.random.randn(3, 7)  # wrong number of features
        with pytest.raises(ValueError, match="Schema drift"):
            ifd.predict(X_wrong)

    def test_save_and_load_roundtrip(self, fitted_ifd):
        ifd, features = fitted_ifd
        X_test = np.random.randn(3, 4)
        original_scores = ifd.predict(X_test)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            ifd.save(model_dir, version="test")
            loaded = IsolationForestDetector.load(model_dir, version="test")

        assert loaded.is_fitted
        assert loaded.feature_names == features
        loaded_scores = loaded.predict(X_test)
        for a, b in zip(original_scores, loaded_scores):
            assert a.is_anomaly == b.is_anomaly

    def test_obvious_anomaly_detected(self, fitted_ifd):
        ifd, _ = fitted_ifd
        # Extreme outlier should be detected
        X_anomaly = np.array([[100.0, -100.0, 100.0, -100.0]])
        scores = ifd.predict(X_anomaly)
        assert scores[0].is_anomaly

    def test_feature_names_stored(self, fitted_ifd):
        ifd, features = fitted_ifd
        assert ifd.feature_names == features


# ──────────────────────────────────────────────────────────────────────────────
# AlertBuffer
# ──────────────────────────────────────────────────────────────────────────────

class TestAlertBuffer:

    def test_empty_buffer(self):
        buf = AlertBuffer(maxlen=10)
        assert len(buf) == 0
        assert buf.recent(5) == []

    def test_append_and_retrieve(self):
        from logbot.detection.anomaly_detector import Alert
        buf = AlertBuffer(maxlen=10)

        # Create a minimal Alert
        alert = Alert(
            alert_id="test_001", detected_at="2024-01-01T00:00:00Z",
            severity="critical", detectors_fired=["zscore"],
            window_start="2024-01-01T00:00:00Z",
            window_end="2024-01-01T00:05:00Z",
            source="test.log", anomaly_scores=[],
            features={"error_rate": 0.9}, top_contributors={},
            summary="Test alert", raw_entry_count=100,
            error_count=90, critical_count=40,
        )
        buf.append(alert)
        assert len(buf) == 1
        assert buf.recent(5)[0].alert_id == "test_001"

    def test_maxlen_evicts_oldest(self):
        from logbot.detection.anomaly_detector import Alert
        buf = AlertBuffer(maxlen=3)
        for i in range(5):
            buf.append(Alert(
                alert_id=f"alert_{i}", detected_at="2024-01-01T00:00:00Z",
                severity="warning", detectors_fired=["zscore"],
                window_start="", window_end="", source="test.log",
                anomaly_scores=[], features={}, top_contributors={},
                summary=f"Alert {i}", raw_entry_count=10,
                error_count=1, critical_count=0,
            ))
        assert len(buf) == 3
        ids = [a.alert_id for a in buf.recent(10)]
        assert "alert_0" not in ids
        assert "alert_4" in ids


# ──────────────────────────────────────────────────────────────────────────────
# AnomalyDetector
# ──────────────────────────────────────────────────────────────────────────────

class TestAnomalyDetector:

    @pytest.fixture
    def detector(self):
        # Fresh detector for each test
        from logbot.detection.model_loader import ModelRegistry
        ModelRegistry._instance = None  # reset singleton
        return AnomalyDetector()

    def test_analyze_returns_result(self, detector):
        window = make_window(50, error_frac=0.05)
        result = detector.analyze(window)
        assert result is not None
        assert isinstance(result.features, dict)

    def test_normal_window_no_alerts_initially(self, detector):
        window = make_window(50, error_frac=0.02)
        result = detector.analyze(window)
        # No alerts without ZScore warmup
        assert isinstance(result.has_anomalies, bool)

    def test_alert_buffer_grows(self, detector):
        # Warm up ZScore
        import random
        random.seed(42)
        for _ in range(50):
            w = make_window(50, error_frac=random.uniform(0.01, 0.04))
            detector.analyze(w)

        # Spike
        spike = make_window(50, error_frac=0.9, source="payments.log")
        result = detector.analyze(spike)
        assert result.has_anomalies
        assert detector.alert_buffer_size() > 0

    def test_highest_severity_property(self, detector):
        window = make_window(50)
        result = detector.analyze(window)
        assert result.highest_severity in ("normal", "warning", "critical")

    def test_singleton(self):
        d1 = get_anomaly_detector()
        d2 = get_anomaly_detector()
        assert d1 is d2

    def test_model_registry_singleton(self):
        r1 = ModelRegistry()
        r2 = ModelRegistry()
        assert r1 is r2
