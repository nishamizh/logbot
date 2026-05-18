"""
logbot/api/server.py
─────────────────────
FastAPI application for LogBot.

Endpoints:
  GET  /health              → liveness + readiness probe
  GET  /metrics             → Prometheus-style counters
  POST /analyze             → run ReAct agent on a question
  POST /anomalies/detect    → run anomaly detection on a log window
  GET  /anomalies/recent    → recent alerts from the alert buffer
  POST /logs/ingest         → ingest raw log entries
  GET  /agent/tools         → list available agent tools

Design decisions (interview-ready talking points):
  • Lifespan context manager (not deprecated on_event) wires all singletons
    at startup and tears them down cleanly on shutdown.
  • Dependency injection via FastAPI Depends() — Planner, AnomalyDetector,
    and ToolRegistry are created once and shared across requests.
  • Request-ID middleware stamps every log line with a trace ID.
  • All endpoints return typed Pydantic response models — no raw dicts.
  • /health returns 503 if critical dependencies are not ready.
  • Prometheus counters on every endpoint — ready for Grafana dashboards.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from logbot.agent.planner import AgentResponse, Planner, get_planner
from logbot.agent.tools import ToolRegistry, get_tool_registry
from logbot.core.config import get_settings
from logbot.core.logging import configure_logging, get_logger, log_request_middleware
from logbot.detection.anomaly_detector import (
    AnomalyDetector,
    LogWindow,
    get_anomaly_detector,
)

log = get_logger(__name__, component="server")

# ──────────────────────────────────────────────────────────────────────────────
# Prometheus-style in-process counters (no external dependency)
# ──────────────────────────────────────────────────────────────────────────────

_metrics: Dict[str, int] = {
    "requests_total":        0,
    "analyze_total":         0,
    "analyze_errors":        0,
    "anomalies_detected":    0,
    "logs_ingested":         0,
    "guardrail_violations":  0,
}


def _inc(key: str, n: int = 1) -> None:
    _metrics[key] = _metrics.get(key, 0) + n


# ──────────────────────────────────────────────────────────────────────────────
# Singletons (initialised in lifespan)
# ──────────────────────────────────────────────────────────────────────────────

_planner:  Optional[Planner]         = None
_detector: Optional[AnomalyDetector] = None
_registry: Optional[ToolRegistry]    = None
_started_at: float                   = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup → yield → shutdown."""
    global _planner, _detector, _registry, _started_at

    configure_logging()
    cfg = get_settings()
    log.info("server_starting", app=cfg.app_name, version=cfg.app_version,
             env=cfg.environment.value)

    # Wire singletons
    _registry = get_tool_registry()
    _detector = get_anomaly_detector()
    _planner  = get_planner(registry=_registry)
    _started_at = time.time()

    log.info("server_ready", host=cfg.api.host, port=cfg.api.port)
    print(cfg.service_banner)

    yield  # ← server runs here

    log.info("server_shutdown")


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    cfg = get_settings()

    app = FastAPI(
        title="LogBot",
        description="LLM-powered log analysis and anomaly detection API",
        version=cfg.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.api.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request-ID + latency logging middleware
    app.middleware("http")(log_request_middleware)

    # ── Request counter middleware ─────────────────────────────────────────────
    @app.middleware("http")
    async def count_requests(request: Request, call_next):
        _inc("requests_total")
        return await call_next(request)

    return app


app = create_app()


# ──────────────────────────────────────────────────────────────────────────────
# Dependency injectors
# ──────────────────────────────────────────────────────────────────────────────

def get_planner_dep() -> Planner:
    if _planner is None:
        raise HTTPException(status_code=503, detail="Planner not initialised")
    return _planner


def get_detector_dep() -> AnomalyDetector:
    if _detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialised")
    return _detector


def get_registry_dep() -> ToolRegistry:
    if _registry is None:
        raise HTTPException(status_code=503, detail="Tool registry not initialised")
    return _registry


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ──────────────────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000,
                          description="Natural language question about logs")
    context:  Optional[Dict[str, Any]] = Field(None,
                          description="Optional context dict (recent alerts, etc.)")

class AnalyzeResponse(BaseModel):
    session_id:          str
    question:            str
    answer:              Optional[str]
    success:             bool
    iterations:          int
    tool_calls:          List[Dict[str, Any]]
    elapsed_ms:          float
    error:               Optional[str]    = None
    guardrail_violation: Optional[str]    = None
    state:               str


class LogEntry(BaseModel):
    timestamp: Optional[str] = Field(None, description="ISO-8601 timestamp")
    level:     str            = Field("INFO", description="Log level")
    service:   str            = Field("unknown")
    message:   str            = Field(..., min_length=1)


class IngestRequest(BaseModel):
    entries: List[LogEntry] = Field(..., min_items=1, max_items=500)
    source:  str            = Field("api", description="Log source identifier")


class IngestResponse(BaseModel):
    accepted:      int
    source:        str
    has_anomalies: bool
    alerts:        List[Dict[str, Any]]
    elapsed_ms:    float


class DetectRequest(BaseModel):
    entries:      List[LogEntry] = Field(..., min_items=1, max_items=500)
    source:       str            = Field("api")
    window_start: Optional[str]  = None
    window_end:   Optional[str]  = None


class AlertResponse(BaseModel):
    alert_id:        str
    severity:        str
    source:          str
    detected_at:     str
    detectors_fired: List[str]
    summary:         str
    error_rate:      float
    error_count:     int


class HealthResponse(BaseModel):
    status:      str
    app:         str
    version:     str
    environment: str
    uptime_s:    float
    components:  Dict[str, str]


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    """
    Liveness + readiness probe.
    Returns 200 if all components are ready, 503 otherwise.
    """
    cfg        = get_settings()
    uptime     = round(time.time() - _started_at, 1) if _started_at else 0.0
    components = {
        "planner":  "ready" if _planner  else "not_initialised",
        "detector": "ready" if _detector else "not_initialised",
        "registry": "ready" if _registry else "not_initialised",
    }
    all_ready  = all(v == "ready" for v in components.values())
    status_str = "healthy" if all_ready else "degraded"

    if not all_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=HealthResponse(
                status=status_str,
                app=cfg.app_name,
                version=cfg.app_version,
                environment=cfg.environment.value,
                uptime_s=uptime,
                components=components,
            ).model_dump(),
        )

    return HealthResponse(
        status=status_str,
        app=cfg.app_name,
        version=cfg.app_version,
        environment=cfg.environment.value,
        uptime_s=uptime,
        components=components,
    )


@app.get("/metrics", tags=["ops"])
async def metrics():
    """Prometheus-style metrics snapshot."""
    return {
        "metrics":    _metrics,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "uptime_s":   round(time.time() - _started_at, 1) if _started_at else 0,
    }


@app.post("/analyze", response_model=AnalyzeResponse, tags=["agent"])
async def analyze(
    request:  AnalyzeRequest,
    planner:  Planner = Depends(get_planner_dep),
):
    """
    Run the ReAct agent on a natural language question about logs.

    The agent will:
    1. Validate and sanitize the question (guardrails)
    2. Reason about which tools to call
    3. Execute tools (search_logs, get_anomalies, etc.)
    4. Return a structured answer with evidence
    """
    _inc("analyze_total")
    log.info("analyze_request", question=request.question[:80])

    import asyncio
    response: AgentResponse = await asyncio.to_thread(
        planner.run, request.question
    )

    if not response.success:
        _inc("analyze_errors")
        if response.guardrail_violation:
            _inc("guardrail_violations")

    return AnalyzeResponse(
        session_id=response.session_id,
        question=response.question,
        answer=response.answer,
        success=response.success,
        iterations=response.iterations,
        tool_calls=response.tool_calls,
        elapsed_ms=response.elapsed_ms,
        error=response.error,
        guardrail_violation=response.guardrail_violation,
        state=response.state,
    )


@app.post("/anomalies/detect", tags=["detection"])
async def detect_anomalies(
    request:  DetectRequest,
    detector: AnomalyDetector = Depends(get_detector_dep),
):
    """
    Run anomaly detection on a submitted log window.
    Returns analysis result including any alerts triggered.
    """
    now = datetime.now(timezone.utc).isoformat()
    entries = [e.model_dump() for e in request.entries]

    window = LogWindow(
        entries=entries,
        window_start=request.window_start or now,
        window_end=request.window_end or now,
        source=request.source,
    )

    import asyncio
    result = await asyncio.to_thread(detector.analyze, window)

    if result.has_anomalies:
        _inc("anomalies_detected", len(result.alerts))

    return {
        "source":           request.source,
        "has_anomalies":    result.has_anomalies,
        "highest_severity": result.highest_severity,
        "alert_count":      len(result.alerts),
        "alerts": [
            {
                "alert_id":        a.alert_id,
                "severity":        a.severity,
                "detectors_fired": a.detectors_fired,
                "summary":         a.summary,
                "top_contributors": a.top_contributors,
            }
            for a in result.alerts
        ],
        "features":    result.features,
        "elapsed_ms":  result.elapsed_ms,
        "model_fitted": result.model_fitted,
    }


@app.get("/anomalies/recent", response_model=List[AlertResponse], tags=["detection"])
async def recent_anomalies(
    limit:    int = 20,
    severity: Optional[str] = None,
    detector: AnomalyDetector = Depends(get_detector_dep),
):
    """Get recent anomaly alerts from the in-memory alert buffer."""
    alerts = detector.recent_alerts(n=limit)
    if severity:
        alerts = [a for a in alerts if a.severity == severity.lower()]

    return [
        AlertResponse(
            alert_id=a.alert_id,
            severity=a.severity,
            source=a.source,
            detected_at=a.detected_at,
            detectors_fired=a.detectors_fired,
            summary=a.summary,
            error_rate=a.features.get("error_rate", 0.0),
            error_count=int(a.features.get("error_count", 0)),
        )
        for a in alerts
    ]


@app.post("/logs/ingest", response_model=IngestResponse, tags=["ingestion"])
async def ingest_logs(
    request:  IngestRequest,
    detector: AnomalyDetector = Depends(get_detector_dep),
):
    """
    Ingest a batch of log entries, run anomaly detection, return alerts.
    This is the main ingestion endpoint for log shippers (Fluentd, Logstash).
    """
    t0      = time.perf_counter()
    entries = [e.model_dump() for e in request.entries]
    _inc("logs_ingested", len(entries))

    now    = datetime.now(timezone.utc).isoformat()
    window = LogWindow(
        entries=entries,
        window_start=now,
        window_end=now,
        source=request.source,
    )

    import asyncio
    result = await asyncio.to_thread(detector.analyze, window)

    if result.has_anomalies:
        _inc("anomalies_detected", len(result.alerts))

    elapsed = round((time.perf_counter() - t0) * 1000, 2)
    log.info("logs_ingested",
             source=request.source,
             count=len(entries),
             has_anomalies=result.has_anomalies,
             elapsed_ms=elapsed)

    return IngestResponse(
        accepted=len(entries),
        source=request.source,
        has_anomalies=result.has_anomalies,
        alerts=[
            {
                "alert_id": a.alert_id,
                "severity": a.severity,
                "summary":  a.summary,
            }
            for a in result.alerts
        ],
        elapsed_ms=elapsed,
    )


@app.get("/agent/tools", tags=["agent"])
async def list_tools(
    registry: ToolRegistry = Depends(get_registry_dep),
):
    """List all available agent tools and their parameter schemas."""
    tools = []
    for name in registry.list_tools():
        spec = registry.get_spec(name)
        if spec:
            tools.append({
                "name":        spec.name,
                "description": spec.description,
                "params": [
                    {
                        "name":     p.name,
                        "type":     p.type,
                        "required": p.required,
                        "default":  p.default,
                    }
                    for p in spec.params
                ],
            })
    return {"tools": tools, "count": len(tools)}


# ──────────────────────────────────────────────────────────────────────────────
# Exception handlers
# ──────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("unhandled_exception",
              path=str(request.url.path),
              error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def start():
    cfg = get_settings()
    uvicorn.run(
        "logbot.api.server:app",
        host=cfg.api.host,
        port=cfg.api.port,
        workers=cfg.api.workers,
        reload=cfg.api.reload,
        log_level=cfg.log_level.value.lower(),
    )


if __name__ == "__main__":
    start()
