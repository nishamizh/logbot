"""
logbot/core/logging.py
──────────────────────
Production-grade structured logging for LogBot.

Design decisions (interview-ready talking points):
  • structlog  → JSON in production, coloured console in dev — zero code change.
  • contextvars → request_id / trace_id propagate automatically across async
                  call stacks without threading.local hacks.
  • TimedBlock  → context-manager that logs duration + outcome for any block;
                  used in agent, retrieval, and detection layers.
  • get_logger() → thin wrapper so callers never import structlog directly;
                   easy to swap backend later.
  • configure_logging() → called ONCE at startup (main.py / server.py);
                          idempotent guard prevents double-init.
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Generator, Optional

import structlog
from structlog.types import EventDict, WrappedLogger

from logbot.core.config import Environment, get_settings

# ──────────────────────────────────────────────────────────────────────────────
# Context variables  (async-safe, no threading.local)
# ──────────────────────────────────────────────────────────────────────────────

_request_id_var: ContextVar[str] = ContextVar("request_id", default="")
_trace_id_var:   ContextVar[str] = ContextVar("trace_id",   default="")
_user_id_var:    ContextVar[str] = ContextVar("user_id",    default="")

# Guard: only configure once per process
_logging_configured = False


# ──────────────────────────────────────────────────────────────────────────────
# Public context helpers
# ──────────────────────────────────────────────────────────────────────────────

def set_request_id(request_id: Optional[str] = None) -> str:
    """Set (or auto-generate) a request ID for the current async context."""
    rid = request_id or str(uuid.uuid4())
    _request_id_var.set(rid)
    return rid


def get_request_id() -> str:
    return _request_id_var.get()


def set_trace_id(trace_id: Optional[str] = None) -> str:
    tid = trace_id or str(uuid.uuid4())
    _trace_id_var.set(tid)
    return tid


def get_trace_id() -> str:
    return _trace_id_var.get()


def set_user_id(user_id: str) -> None:
    _user_id_var.set(user_id)


def get_user_id() -> str:
    return _user_id_var.get()


# ──────────────────────────────────────────────────────────────────────────────
# structlog processors
# ──────────────────────────────────────────────────────────────────────────────

def _inject_context(
    logger: WrappedLogger, method: str, event_dict: EventDict
) -> EventDict:
    """Inject request_id / trace_id / user_id from contextvars into every log record."""
    if rid := _request_id_var.get():
        event_dict["request_id"] = rid
    if tid := _trace_id_var.get():
        event_dict["trace_id"] = tid
    if uid := _user_id_var.get():
        event_dict["user_id"] = uid
    return event_dict


def _add_app_metadata(
    logger: WrappedLogger, method: str, event_dict: EventDict
) -> EventDict:
    """Stamp every log line with app name + version (read once, cached)."""
    cfg = get_settings()
    event_dict["app"]     = cfg.app_name
    event_dict["version"] = cfg.app_version
    event_dict["env"]     = cfg.environment.value
    return event_dict


def _drop_color_message(
    logger: WrappedLogger, method: str, event_dict: EventDict
) -> EventDict:
    """Remove the 'color_message' key added by uvicorn's access logger."""
    event_dict.pop("color_message", None)
    return event_dict


# ──────────────────────────────────────────────────────────────────────────────
# Configure
# ──────────────────────────────────────────────────────────────────────────────

def configure_logging() -> None:
    """
    Call ONCE at application startup.
    - Development  → human-friendly coloured console output.
    - Production   → newline-delimited JSON (log aggregator ready).
    Idempotent: safe to call multiple times (no-op after first call).
    """
    global _logging_configured
    if _logging_configured:
        return

    cfg      = get_settings()
    is_prod  = cfg.is_production
    level    = getattr(logging, cfg.log_level.value, logging.INFO)

    # ── stdlib root logger ──────────────────────────────────────────────────
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    # Quieten noisy third-party loggers
    for noisy in ("urllib3", "httpx", "httpcore", "transformers", "pyspark",
                  "py4j", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ── structlog shared processors ─────────────────────────────────────────
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        _inject_context,
        _add_app_metadata,
        _drop_color_message,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]

    if is_prod:
        # JSON — one object per line, ingested by Datadog / CloudWatch / ELK
        renderer = structlog.processors.JSONRenderer()
    else:
        # Pretty coloured console for local dev
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _logging_configured = True

    # Emit the very first log line
    log = get_logger(__name__)
    log.info(
        "logging_configured",
        level=cfg.log_level.value,
        renderer="json" if is_prod else "console",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Public logger factory
# ──────────────────────────────────────────────────────────────────────────────

def get_logger(name: str = "logbot", **initial_values: Any) -> structlog.BoundLogger:
    """
    Return a structlog BoundLogger pre-bound with `name` and any extra fields.

    Usage:
        log = get_logger(__name__)
        log.info("event", key="value")

        # With extra permanent context:
        log = get_logger(__name__, component="retrieval", index="logbot_logs")
        log.debug("query_executed", hits=5)
    """
    return structlog.get_logger(name).bind(**initial_values)


# ──────────────────────────────────────────────────────────────────────────────
# TimedBlock — context manager for instrumented code sections
# ──────────────────────────────────────────────────────────────────────────────

@contextmanager
def TimedBlock(
    name: str,
    logger: Optional[structlog.BoundLogger] = None,
    log_level: str = "info",
    extra: Optional[Dict[str, Any]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager that logs start, end, and elapsed time for any code block.
    On exception, logs the error and re-raises.

    Usage:
        with TimedBlock("embed_documents", logger=log, extra={"doc_count": 42}):
            embedder.encode(docs)

        # Access elapsed time inside the block:
        with TimedBlock("vector_search") as ctx:
            results = store.query(q)
            ctx["hits"] = len(results)
    """
    _log    = logger or get_logger("logbot.timed")
    _extra  = extra or {}
    ctx: Dict[str, Any] = {"name": name}  # yielded to caller

    _emit = getattr(_log, log_level, _log.info)
    _emit("block_start", block=name, **_extra)

    t0 = time.perf_counter()
    try:
        yield ctx
        elapsed = time.perf_counter() - t0
        ctx["elapsed_ms"] = round(elapsed * 1000, 2)
        _emit(
            "block_end",
            block=name,
            elapsed_ms=ctx["elapsed_ms"],
            status="ok",
            **{k: v for k, v in ctx.items() if k not in ("name", "elapsed_ms")},
            **_extra,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        ctx["elapsed_ms"] = round(elapsed * 1000, 2)
        ctx["error"]      = str(exc)
        _log.error(
            "block_error",
            block=name,
            elapsed_ms=ctx["elapsed_ms"],
            error=str(exc),
            traceback=traceback.format_exc(),
            **_extra,
        )
        raise


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI middleware helper
# ──────────────────────────────────────────────────────────────────────────────

async def log_request_middleware(request: Any, call_next: Any) -> Any:
    """
    Starlette middleware that:
      1. Reads / generates X-Request-ID header.
      2. Stamps every log line in this request with request_id.
      3. Logs method, path, status, and latency on completion.

    Register in server.py:
        app.middleware("http")(log_request_middleware)
    """
    from starlette.requests import Request  # lazy import — no hard dep at module level

    req: Request = request
    rid = req.headers.get("x-request-id") or set_request_id()
    set_request_id(rid)

    log = get_logger("logbot.http")
    log.info("request_start", method=req.method, path=req.url.path)

    t0       = time.perf_counter()
    response = await call_next(request)
    elapsed  = round((time.perf_counter() - t0) * 1000, 2)

    log.info(
        "request_end",
        method=req.method,
        path=req.url.path,
        status=response.status_code,
        elapsed_ms=elapsed,
    )

    response.headers["X-Request-ID"] = rid
    return response


# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test  →  python -m logbot.core.logging
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    configure_logging()

    log = get_logger(__name__, component="smoke_test")

    # 1 — basic levels
    log.debug("debug_message",   note="only visible when LOG_LEVEL=DEBUG")
    log.info ("info_message",    note="normal operation")
    log.warning("warn_message",  note="something looks off")
    log.error("error_message",   note="something went wrong")

    # 2 — request-id propagation
    rid = set_request_id()
    log.info("with_request_id",  request_id=rid)

    # 3 — TimedBlock happy path
    with TimedBlock("example_block", logger=log, extra={"items": 10}) as ctx:
        time.sleep(0.05)
        ctx["processed"] = 10

    # 4 — TimedBlock exception path
    try:
        with TimedBlock("failing_block", logger=log):
            raise ValueError("intentional test error")
    except ValueError:
        pass   # expected

    # 5 — idempotency check
    configure_logging()   # second call should be silent no-op
    configure_logging()

    print("\n✅  core/logging.py smoke-test passed.")