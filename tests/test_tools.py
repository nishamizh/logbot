"""
tests/test_tools.py
────────────────────
Unit tests for ToolRegistry, tool handlers, and parameter validation.
"""

import pytest
from logbot.agent.tools import (
    ToolRegistry, ToolResult, ToolParam,
    validate_and_coerce,
    make_search_logs_handler,
    make_get_anomalies_handler,
    make_analyze_window_handler,
    make_summarize_logs_handler,
    make_get_service_health_handler,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def registry():
    """Stub registry — no real detector or vector store."""
    return ToolRegistry.build(detector=None, vector_store=None)


# ──────────────────────────────────────────────────────────────────────────────
# ToolRegistry tests
# ──────────────────────────────────────────────────────────────────────────────

class TestToolRegistry:

    def test_all_tools_registered(self, registry):
        tools = registry.list_tools()
        assert "search_logs"       in tools
        assert "get_anomalies"     in tools
        assert "analyze_window"    in tools
        assert "summarize_logs"    in tools
        assert "get_service_health" in tools

    def test_execute_returns_tool_result(self, registry):
        r = registry.execute("search_logs", {"query": "timeout"})
        assert isinstance(r, ToolResult)

    def test_unknown_tool_returns_error(self, registry):
        r = registry.execute("nonexistent_tool", {})
        assert not r.success
        assert "Unknown tool" in r.error

    def test_missing_required_param_returns_error(self, registry):
        r = registry.execute("search_logs", {})  # query required
        assert not r.success
        assert "query" in r.error

    def test_missing_required_param_analyze_window(self, registry):
        r = registry.execute("analyze_window", {})  # source required
        assert not r.success
        assert "source" in r.error

    def test_type_coercion_str_to_int(self, registry):
        r = registry.execute("search_logs", {"query": "error", "top_k": "3"})
        assert r.success  # "3" coerced to int

    def test_execute_never_raises(self, registry):
        """execute() must catch all exceptions and return ToolResult."""
        r = registry.execute("search_logs", {"query": "test", "top_k": -1})
        assert isinstance(r, ToolResult)

    def test_get_spec_returns_spec(self, registry):
        spec = registry.get_spec("search_logs")
        assert spec is not None
        assert spec.name == "search_logs"
        assert len(spec.params) > 0

    def test_get_spec_unknown_returns_none(self, registry):
        assert registry.get_spec("does_not_exist") is None


# ──────────────────────────────────────────────────────────────────────────────
# Individual tool handler tests (stub mode)
# ──────────────────────────────────────────────────────────────────────────────

class TestSearchLogs:

    def test_returns_results(self, registry):
        r = registry.execute("search_logs", {"query": "database timeout"})
        assert r.success
        assert "results" in r.data
        assert isinstance(r.data["results"], list)

    def test_top_k_respected(self, registry):
        r = registry.execute("search_logs", {"query": "error", "top_k": 2})
        assert r.success
        assert len(r.data["results"]) <= 2

    def test_observation_format(self, registry):
        r = registry.execute("search_logs", {"query": "error"})
        obs = r.to_observation()
        assert obs.startswith("[search_logs]")

    def test_stub_mode_marker(self, registry):
        r = registry.execute("search_logs", {"query": "test"})
        assert r.data.get("mode") == "stub"


class TestGetAnomalies:

    def test_returns_alerts(self, registry):
        r = registry.execute("get_anomalies", {})
        assert r.success
        assert "alerts" in r.data
        assert "total" in r.data

    def test_severity_filter_param(self, registry):
        r = registry.execute("get_anomalies", {"severity": "critical"})
        assert r.success

    def test_limit_param(self, registry):
        r = registry.execute("get_anomalies", {"limit": 3})
        assert r.success


class TestAnalyzeWindow:

    def test_returns_analysis(self, registry):
        r = registry.execute("analyze_window", {"source": "payments.log"})
        assert r.success
        assert "has_anomalies" in r.data
        assert "highest_severity" in r.data

    def test_default_time_range(self, registry):
        r = registry.execute("analyze_window", {"source": "auth.log"})
        assert r.success
        assert r.data.get("time_range_minutes") == 5


class TestSummarizeLogs:

    def test_returns_summary(self, registry):
        r = registry.execute("summarize_logs", {})
        assert r.success
        assert "total_entries" in r.data
        assert "breakdown" in r.data

    def test_group_by_level(self, registry):
        r = registry.execute("summarize_logs", {"group_by": "level"})
        assert r.success
        breakdown = r.data["breakdown"]
        assert "ERROR" in breakdown or "INFO" in breakdown


class TestGetServiceHealth:

    def test_returns_health(self, registry):
        r = registry.execute("get_service_health", {"service": "payments"})
        assert r.success
        assert "status" in r.data
        assert r.data["status"] in ("healthy", "degraded", "critical")

    def test_has_recommendation(self, registry):
        r = registry.execute("get_service_health", {"service": "auth"})
        assert r.success
        assert "recommendation" in r.data


# ──────────────────────────────────────────────────────────────────────────────
# validate_and_coerce tests
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateAndCoerce:

    def test_required_param_missing_raises(self):
        specs = [ToolParam("query", "str", required=True)]
        with pytest.raises(ValueError, match="query"):
            validate_and_coerce({}, specs)

    def test_optional_param_uses_default(self):
        specs = [ToolParam("top_k", "int", required=False, default=5)]
        result = validate_and_coerce({}, specs)
        assert result["top_k"] == 5

    def test_string_to_int_coercion(self):
        specs = [ToolParam("top_k", "int", required=False, default=5)]
        result = validate_and_coerce({"top_k": "10"}, specs)
        assert result["top_k"] == 10
        assert isinstance(result["top_k"], int)

    def test_string_to_float_coercion(self):
        specs = [ToolParam("threshold", "float", required=False, default=0.5)]
        result = validate_and_coerce({"threshold": "0.8"}, specs)
        assert result["threshold"] == pytest.approx(0.8)

    def test_invalid_type_raises(self):
        specs = [ToolParam("count", "int", required=True)]
        with pytest.raises(ValueError):
            validate_and_coerce({"count": "not_a_number"}, specs)


# ──────────────────────────────────────────────────────────────────────────────
# ToolResult tests
# ──────────────────────────────────────────────────────────────────────────────

class TestToolResult:

    def test_success_to_observation(self):
        r = ToolResult(tool="search_logs", success=True, data={"results": []})
        obs = r.to_observation()
        assert "[search_logs]" in obs

    def test_error_to_observation(self):
        r = ToolResult(tool="search_logs", success=False, error="timeout")
        obs = r.to_observation()
        assert "ERROR" in obs
        assert "timeout" in obs

    def test_dict_data_json_serialized(self):
        r = ToolResult(tool="get_anomalies", success=True,
                       data={"alerts": [], "total": 0})
        obs = r.to_observation()
        assert "alerts" in obs
        assert "total" in obs
