"""
logbot/agent/tools.py
──────────────────────
Tool definitions and execution layer for LogBot's ReAct agent.

Each tool is:
  1. Defined as a ToolSpec dataclass (name, description, parameters, handler)
  2. Registered in ToolRegistry
  3. Called by the planner via ToolRegistry.execute(name, params)

Design decisions (interview-ready talking points):
  • ToolResult is typed — planner never parses raw dicts.
  • Every tool handler is a pure function (no class state) — easy to test.
  • ToolRegistry.execute() catches ALL exceptions and returns ToolResult(error=...)
    so the agent loop never crashes on a bad tool call.
  • Handlers are injected with dependencies (detector, vector_store) at registry
    build time — no hidden globals, easy to mock in tests.
  • Tool execution is synchronous here; async wrapper added in server.py via
    asyncio.to_thread() to avoid blocking the event loop.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from logbot.core.logging import TimedBlock, get_logger

log = get_logger(__name__, component="tools")


# ──────────────────────────────────────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolParam:
    """Schema for a single tool parameter."""
    name:        str
    type:        str           # "str" | "int" | "float" | "bool"
    required:    bool = False
    default:     Any  = None
    description: str  = ""


@dataclass
class ToolSpec:
    """Complete specification for one tool."""
    name:        str
    description: str
    params:      List[ToolParam]
    handler:     Callable       # (params: Dict) -> Any


@dataclass
class ToolResult:
    """
    Typed result returned by every tool call.
    On success: data is populated, error is None.
    On failure: error is populated, data is None.
    elapsed_ms is always set.
    """
    tool:       str
    success:    bool
    data:       Optional[Any]  = None
    error:      Optional[str]  = None
    elapsed_ms: float          = 0.0

    def to_observation(self) -> str:
        """
        Render as the OBSERVATION string the LLM sees.
        Keep it concise — every token counts.
        """
        if not self.success:
            return f"[{self.tool}] ERROR: {self.error}"
        if isinstance(self.data, (dict, list)):
            try:
                return f"[{self.tool}] {json.dumps(self.data, indent=2, default=str)}"
            except Exception:
                return f"[{self.tool}] {str(self.data)}"
        return f"[{self.tool}] {self.data}"


# ──────────────────────────────────────────────────────────────────────────────
# Tool parameter validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_and_coerce(
    raw: Dict[str, Any], specs: List[ToolParam]
) -> Dict[str, Any]:
    """
    Validate raw ACTION_INPUT dict against ToolParam specs.
    Returns coerced params dict or raises ValueError with a clear message.
    """
    coerced: Dict[str, Any] = {}
    spec_map = {p.name: p for p in specs}

    for spec in specs:
        if spec.name in raw:
            val = raw[spec.name]
            # Type coercion
            try:
                if spec.type == "int":
                    val = int(val)
                elif spec.type == "float":
                    val = float(val)
                elif spec.type == "bool":
                    val = bool(val)
                elif spec.type == "str":
                    val = str(val)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Parameter '{spec.name}' expected {spec.type}, "
                    f"got {type(val).__name__}: {e}"
                )
            coerced[spec.name] = val
        elif spec.required:
            raise ValueError(f"Required parameter '{spec.name}' is missing")
        else:
            coerced[spec.name] = spec.default

    return coerced


# ──────────────────────────────────────────────────────────────────────────────
# Tool handler factories
# (each returns a handler function closed over its dependencies)
# ──────────────────────────────────────────────────────────────────────────────

def make_search_logs_handler(vector_store: Optional[Any] = None) -> Callable:
    """
    Search the vector store for semantically similar log entries.
    Falls back to a mock response if vector_store is None (dev mode).
    """
    def handler(params: Dict[str, Any]) -> Any:
        query              = params["query"]
        top_k              = params.get("top_k", 5)
        source             = params.get("source")
        level              = params.get("level")
        time_range_minutes = params.get("time_range_minutes")

        if vector_store is None:
            # Dev-mode stub — returns plausible-looking results
            log.debug("search_logs_stub_mode", query=query)
            return {
                "query": query,
                "results": [
                    {
                        "rank": i + 1,
                        "score": round(0.95 - i * 0.08, 3),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "level": level or "ERROR",
                        "service": source or "api-gateway",
                        "message": f"[STUB] Log entry matching '{query}' — result {i+1}",
                    }
                    for i in range(min(top_k, 3))
                ],
                "total_searched": 1000,
                "mode": "stub",
            }

        # Real path
        filters = {}
        if source:
            filters["source"] = source
        if level:
            filters["level"] = level.upper()
        if time_range_minutes:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=time_range_minutes)
            filters["after"] = cutoff.isoformat()

        results = vector_store.query(query, top_k=top_k, filters=filters)
        return {"query": query, "results": results, "total": len(results)}

    return handler


def make_get_anomalies_handler(detector: Optional[Any] = None) -> Callable:
    """
    Retrieve recent anomaly alerts from the AnomalyDetector alert buffer.
    """
    def handler(params: Dict[str, Any]) -> Any:
        severity       = params.get("severity")
        source         = params.get("source")
        limit          = params.get("limit", 10)
        since_minutes  = params.get("since_minutes")

        if detector is None:
            log.debug("get_anomalies_stub_mode")
            return {
                "alerts": [
                    {
                        "alert_id":      "alert_stub_001",
                        "severity":      severity or "critical",
                        "source":        source or "payments.log",
                        "detected_at":   datetime.now(timezone.utc).isoformat(),
                        "detectors_fired": ["zscore"],
                        "summary":       "[STUB] High error rate detected",
                        "error_rate":    0.45,
                        "error_count":   45,
                    }
                ],
                "total": 1,
                "mode": "stub",
            }

        # Real path
        alerts = detector.recent_alerts(n=limit)

        if severity:
            alerts = [a for a in alerts if a.severity == severity.lower()]
        if source:
            alerts = [a for a in alerts if a.source == source]
        if since_minutes:
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=since_minutes)
            cutoff_iso = cutoff.isoformat()
            alerts = [a for a in alerts if a.detected_at >= cutoff_iso]

        return {
            "alerts": [
                {
                    "alert_id":        a.alert_id,
                    "severity":        a.severity,
                    "source":          a.source,
                    "detected_at":     a.detected_at,
                    "detectors_fired": a.detectors_fired,
                    "summary":         a.summary,
                    "error_rate":      a.features.get("error_rate", 0),
                    "error_count":     int(a.features.get("error_count", 0)),
                    "top_contributors": a.top_contributors,
                }
                for a in alerts[:limit]
            ],
            "total": len(alerts),
        }

    return handler


def make_analyze_window_handler(detector: Optional[Any] = None) -> Callable:
    """
    Run anomaly detection on a specific log source + time window.
    """
    def handler(params: Dict[str, Any]) -> Any:
        source             = params["source"]
        time_range_minutes = params.get("time_range_minutes", 5)

        if detector is None:
            log.debug("analyze_window_stub_mode", source=source)
            return {
                "source":           source,
                "time_range_minutes": time_range_minutes,
                "has_anomalies":    True,
                "highest_severity": "warning",
                "alert_count":      1,
                "features": {
                    "error_rate":    0.08,
                    "error_count":   4,
                    "critical_count": 0,
                    "error_burst":   2,
                },
                "mode": "stub",
            }

        # Real path — would pull from vector store / log buffer by source+time
        # For now returns current detector status for the source
        recent = [
            a for a in detector.recent_alerts(n=50)
            if a.source == source
        ]
        return {
            "source":           source,
            "time_range_minutes": time_range_minutes,
            "has_anomalies":    len(recent) > 0,
            "highest_severity": max(
                (a.severity for a in recent),
                key=lambda s: {"normal": 0, "warning": 1, "critical": 2}.get(s, 0),
                default="normal",
            ),
            "alert_count":      len(recent),
            "recent_alerts":    [a.alert_id for a in recent[:3]],
        }

    return handler


def make_summarize_logs_handler(vector_store: Optional[Any] = None) -> Callable:
    """
    Summarize log patterns and statistics for a given source/time window.
    """
    def handler(params: Dict[str, Any]) -> Any:
        source             = params.get("source", "all")
        time_range_minutes = params.get("time_range_minutes", 60)
        group_by           = params.get("group_by", "level")

        if vector_store is None:
            log.debug("summarize_logs_stub_mode", source=source)
            return {
                "source":           source,
                "time_range_minutes": time_range_minutes,
                "total_entries":    1247,
                "group_by":         group_by,
                "breakdown": {
                    "INFO":     983,
                    "WARNING":   198,
                    "ERROR":      61,
                    "CRITICAL":    5,
                },
                "error_rate":    round(66 / 1247, 4),
                "top_errors": [
                    "DB connection timeout (n=34)",
                    "Auth service unavailable (n=18)",
                    "Rate limit exceeded (n=9)",
                ],
                "mode": "stub",
            }

        # Real path would aggregate from vector store
        return {"source": source, "status": "aggregation_not_implemented"}

    return handler


def make_get_service_health_handler(detector: Optional[Any] = None) -> Callable:
    """
    Get the current health status of a specific service.
    """
    def handler(params: Dict[str, Any]) -> Any:
        service            = params["service"]
        time_range_minutes = params.get("time_range_minutes", 15)

        if detector is None:
            log.debug("get_service_health_stub_mode", service=service)
            return {
                "service":          service,
                "status":           "degraded",
                "time_range_minutes": time_range_minutes,
                "error_rate":       0.12,
                "critical_alerts":  0,
                "warning_alerts":   2,
                "recommendation":   "Monitor closely. Error rate above 10% threshold.",
                "mode": "stub",
            }

        # Real path
        recent = [
            a for a in detector.recent_alerts(n=100)
            if service.lower() in a.source.lower()
        ]
        critical = [a for a in recent if a.severity == "critical"]
        warning  = [a for a in recent if a.severity == "warning"]

        if critical:
            status = "critical"
            recommendation = f"IMMEDIATE ACTION: {len(critical)} critical alert(s) detected."
        elif warning:
            status = "degraded"
            recommendation = f"Monitor closely: {len(warning)} warning(s) in last {time_range_minutes}m."
        else:
            status = "healthy"
            recommendation = "No anomalies detected."

        return {
            "service":          service,
            "status":           status,
            "time_range_minutes": time_range_minutes,
            "critical_alerts":  len(critical),
            "warning_alerts":   len(warning),
            "recommendation":   recommendation,
        }

    return handler


# ──────────────────────────────────────────────────────────────────────────────
# ToolRegistry
# ──────────────────────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    Central registry of all agent tools.

    Usage:
        registry = ToolRegistry.build(detector=my_detector)
        result   = registry.execute("get_anomalies", {"severity": "critical"})
        print(result.to_observation())
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec
        log.debug("tool_registered", name=spec.name)

    def execute(self, name: str, raw_params: Dict[str, Any]) -> ToolResult:
        """
        Validate params, run handler, return ToolResult.
        Never raises — all exceptions are caught and returned as error results.
        """
        t0 = time.perf_counter()

        if name not in self._tools:
            return ToolResult(
                tool=name, success=False,
                error=f"Unknown tool '{name}'. Available: {list(self._tools)}",
                elapsed_ms=0.0,
            )

        spec = self._tools[name]
        try:
            params = validate_and_coerce(raw_params, spec.params)
        except ValueError as e:
            return ToolResult(
                tool=name, success=False,
                error=f"Parameter error: {e}",
                elapsed_ms=round((time.perf_counter() - t0) * 1000, 2),
            )

        try:
            with TimedBlock(f"tool_{name}", logger=log,
                            extra={"params": str(raw_params)[:120]}):
                data = spec.handler(params)
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            return ToolResult(tool=name, success=True, data=data, elapsed_ms=elapsed)

        except Exception as exc:
            elapsed = round((time.perf_counter() - t0) * 1000, 2)
            log.error("tool_execution_failed", tool=name, error=str(exc))
            return ToolResult(
                tool=name, success=False,
                error=f"Execution error: {exc}",
                elapsed_ms=elapsed,
            )

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_spec(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        detector:     Optional[Any] = None,
        vector_store: Optional[Any] = None,
    ) -> "ToolRegistry":
        """
        Build and return a fully wired ToolRegistry.
        Pass None for detector/vector_store to use stub handlers (dev/test mode).
        """
        registry = cls()

        registry.register(ToolSpec(
            name="search_logs",
            description="Search logs semantically using vector similarity",
            params=[
                ToolParam("query",              "str",  required=True,  description="Search query"),
                ToolParam("top_k",              "int",  default=5,      description="Number of results"),
                ToolParam("source",             "str",  default=None,   description="Filter by source"),
                ToolParam("level",              "str",  default=None,   description="Filter by log level"),
                ToolParam("time_range_minutes", "int",  default=None,   description="Lookback window"),
            ],
            handler=make_search_logs_handler(vector_store),
        ))

        registry.register(ToolSpec(
            name="get_anomalies",
            description="Get recent anomaly alerts from the detection engine",
            params=[
                ToolParam("severity",       "str", default=None, description="Filter by severity"),
                ToolParam("source",         "str", default=None, description="Filter by source"),
                ToolParam("limit",          "int", default=10,   description="Max results"),
                ToolParam("since_minutes",  "int", default=None, description="Lookback window"),
            ],
            handler=make_get_anomalies_handler(detector),
        ))

        registry.register(ToolSpec(
            name="analyze_window",
            description="Run anomaly detection on a specific log source and time window",
            params=[
                ToolParam("source",             "str", required=True, description="Log source"),
                ToolParam("time_range_minutes", "int", default=5,     description="Window size"),
            ],
            handler=make_analyze_window_handler(detector),
        ))

        registry.register(ToolSpec(
            name="summarize_logs",
            description="Get log statistics and patterns for a source/time window",
            params=[
                ToolParam("source",             "str", default="all",   description="Log source"),
                ToolParam("time_range_minutes", "int", default=60,      description="Window"),
                ToolParam("group_by",           "str", default="level", description="Grouping key"),
            ],
            handler=make_summarize_logs_handler(vector_store),
        ))

        registry.register(ToolSpec(
            name="get_service_health",
            description="Get health status of a specific service",
            params=[
                ToolParam("service",            "str", required=True, description="Service name"),
                ToolParam("time_range_minutes", "int", default=15,    description="Lookback window"),
            ],
            handler=make_get_service_health_handler(detector),
        ))

        log.info("tool_registry_built", tools=registry.list_tools())
        return registry


# ──────────────────────────────────────────────────────────────────────────────
# Module-level convenience accessor
# ──────────────────────────────────────────────────────────────────────────────

def get_tool_registry(
    detector:     Optional[Any] = None,
    vector_store: Optional[Any] = None,
) -> ToolRegistry:
    return ToolRegistry.build(detector=detector, vector_store=vector_store)


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test  →  python -m logbot.agent.tools
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from logbot.core.logging import configure_logging
    configure_logging()

    log.info("smoke_test_start")

    # Build registry in stub mode (no real detector/vector_store)
    registry = ToolRegistry.build()
    assert set(registry.list_tools()) == {
        "search_logs", "get_anomalies", "analyze_window",
        "summarize_logs", "get_service_health"
    }
    print(f"\n✅  Registry built — tools: {registry.list_tools()}")

    # ── 1. search_logs ────────────────────────────────────────────────────────
    r1 = registry.execute("search_logs", {"query": "database connection timeout"})
    assert r1.success
    assert r1.data["results"]
    obs = r1.to_observation()
    assert "search_logs" in obs
    print(f"\n✅  search_logs      — {len(r1.data['results'])} results, "
          f"{r1.elapsed_ms}ms")

    # ── 2. get_anomalies ──────────────────────────────────────────────────────
    r2 = registry.execute("get_anomalies", {"severity": "critical", "limit": 5})
    assert r2.success
    assert r2.data["total"] >= 0
    print(f"✅  get_anomalies    — {r2.data['total']} alerts, {r2.elapsed_ms}ms")

    # ── 3. analyze_window ─────────────────────────────────────────────────────
    r3 = registry.execute("analyze_window", {"source": "payments.log"})
    assert r3.success
    assert "has_anomalies" in r3.data
    print(f"✅  analyze_window   — has_anomalies={r3.data['has_anomalies']}, "
          f"{r3.elapsed_ms}ms")

    # ── 4. summarize_logs ─────────────────────────────────────────────────────
    r4 = registry.execute("summarize_logs", {"source": "api.log", "group_by": "level"})
    assert r4.success
    assert "breakdown" in r4.data
    print(f"✅  summarize_logs   — {r4.data['total_entries']} entries, "
          f"{r4.elapsed_ms}ms")

    # ── 5. get_service_health ─────────────────────────────────────────────────
    r5 = registry.execute("get_service_health", {"service": "payments"})
    assert r5.success
    assert r5.data["status"] in ("healthy", "degraded", "critical")
    print(f"✅  get_service_health — status={r5.data['status']}, {r5.elapsed_ms}ms")

    # ── 6. Unknown tool ───────────────────────────────────────────────────────
    r6 = registry.execute("nonexistent_tool", {})
    assert not r6.success
    assert "Unknown tool" in r6.error
    print(f"✅  unknown tool     — error handled: '{r6.error[:50]}...'")

    # ── 7. Missing required param ─────────────────────────────────────────────
    r7 = registry.execute("search_logs", {})   # query is required
    assert not r7.success
    assert "query" in r7.error
    print(f"✅  missing param    — error handled: '{r7.error}'")

    # ── 8. Type coercion ──────────────────────────────────────────────────────
    r8 = registry.execute("search_logs", {"query": "timeout", "top_k": "3"})
    assert r8.success   # "3" coerced to int 3
    print(f"✅  type coercion    — top_k='3' coerced to int correctly")

    # ── 9. to_observation format ──────────────────────────────────────────────
    obs = r2.to_observation()
    assert obs.startswith("[get_anomalies]")
    print(f"✅  to_observation   — '{obs[:60]}...'")

    print("\n✅  All tools.py smoke-tests passed.")