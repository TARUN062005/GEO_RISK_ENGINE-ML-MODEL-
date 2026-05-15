"""
app/main.py
-----------
FastAPI Application Entrypoint (Log3, upgraded Log10)

Production-grade API with:
  - POST /api/v1/analyze — full multi-mode risk analysis
  - GET  /health          — health check
  - GET  /metrics          — system metrics
  - startup model pre-loading
"""

from __future__ import annotations

import logging
import uuid
import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

try:
    from core.logging_config import configure_logging
    configure_logging()
except Exception:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    origin: str = Field(..., description="Origin location (free text)", min_length=2, max_length=160)
    destination: str = Field(..., description="Destination location (free text)", min_length=2, max_length=160)
    radius_km: float = Field(50.0, ge=1.0, le=500.0, description="Buffer radius in km")
    min_confidence: float = Field(0.50, ge=0.0, le=1.0, description="Min classifier confidence")

    @field_validator("origin", "destination")
    @classmethod
    def clean_location(cls, value: str) -> str:
        value = " ".join(value.strip().split())
        if any(ch in value for ch in "\x00\r\n\t"):
            raise ValueError("location must be a single-line string")
        return value


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    version: str
    request_id: str | None = None


class ReadyResponse(BaseModel):
    status: str
    mongo: str
    indexes: str
    request_id: str | None = None


class MetricsResponse(BaseModel):
    uptime_seconds: float
    total_analyses: int
    avg_latency_ms: float
    last_analysis_at: str | None


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

_app_state = {
    "start_time": None,
    "total_analyses": 0,
    "total_latency_ms": 0.0,
    "last_analysis_at": None,
    "mongo_client": None,
    "ready": False,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    _app_state["start_time"] = time.time()
    logger.info("Geo-Intelligence Engine API starting...")

    try:
        from config.settings import validate_environment
        validate_environment()
        logger.info("Environment validation passed.")
    except Exception as exc:
        logger.warning("Environment validation warning: %s", exc)

    # Log12: API remains read-only and does not load heavy ML models.
    # Ingestion uses singleton model managers and batch inference.

    # Connect MongoDB
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
        from config.settings import MONGO_URI, MONGO_DB, MONGO_COLLECTION
        from ingestion.realtime_worker import _ensure_indexes
        _app_state["mongo_client"] = AsyncIOMotorClient(MONGO_URI)
        collection = _app_state["mongo_client"][MONGO_DB][MONGO_COLLECTION]
        await _app_state["mongo_client"].admin.command("ping")
        await _ensure_indexes(collection)
        _app_state["ready"] = True
        logger.info("MongoDB connected.")
    except Exception as exc:
        _app_state["ready"] = False
        logger.warning("MongoDB connection failed: %s", exc)

    yield

    # Shutdown
    if _app_state["mongo_client"]:
        _app_state["mongo_client"].close()
    logger.info("API shutdown complete.")


app = FastAPI(
    title="Geo Risk Engine",
    description="Production-grade geopolitical risk assessment for logistics routes.",
    version="1.0.0",
    lifespan=lifespan,
)

try:
    from config.settings import CORS_ALLOW_ORIGINS
    _cors_origins = [o.strip() for o in CORS_ALLOW_ORIGINS.split(",") if o.strip()]
except Exception:
    _cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def production_guardrails(request: Request, call_next):
    """Request ID, size limit, timing, and safe error envelope."""
    from config.settings import API_TIMEOUT_SECONDS, MAX_REQUEST_BYTES

    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id

    content_length = request.headers.get("content-length")
    try:
        too_large = bool(content_length and int(content_length) > MAX_REQUEST_BYTES)
    except ValueError:
        too_large = True
    if too_large:
        return JSONResponse(
            status_code=413,
            content={"error": "request_too_large", "request_id": request_id},
            headers={"x-request-id": request_id},
        )

    start = time.time()
    try:
        response = await asyncio.wait_for(call_next(request), timeout=API_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.warning("request timeout path=%s request_id=%s", request.url.path, request_id)
        return JSONResponse(
            status_code=504,
            content={"error": "request_timeout", "request_id": request_id},
            headers={"x-request-id": request_id},
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("request failed path=%s request_id=%s", request.url.path, request_id)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "request_id": request_id},
            headers={"x-request-id": request_id},
        )

    latency_ms = round((time.time() - start) * 1000, 1)
    response.headers["x-request-id"] = request_id
    response.headers["x-response-time-ms"] = str(latency_ms)
    logger.info(
        "request method=%s path=%s status=%s latency_ms=%.1f request_id=%s",
        request.method, request.url.path, response.status_code, latency_ms, request_id,
    )
    return response

# Keep legacy router under an explicit legacy prefix so /api/v1/analyze
# consistently returns the production multi-mode response.
try:
    from app.routes.analyze import router as analyze_router
    app.include_router(analyze_router, prefix="/api/legacy")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    uptime = time.time() - (_app_state["start_time"] or time.time())
    return HealthResponse(
        status="ok",
        uptime_seconds=round(uptime, 1),
        version="1.0.0",
        request_id=getattr(request.state, "request_id", None),
    )


@app.get("/ready", response_model=ReadyResponse)
async def ready(request: Request):
    client = _app_state.get("mongo_client")
    if client is None:
        raise HTTPException(status_code=503, detail="not_ready")
    try:
        await client.admin.command("ping")
        if not _app_state["ready"]:
            raise HTTPException(status_code=503, detail="not_ready")
        return ReadyResponse(status="ready", mongo="connected", indexes="ensured",
                             request_id=getattr(request.state, "request_id", None))
    except Exception:
        raise HTTPException(status_code=503, detail="not_ready")


@app.get("/metrics")
async def metrics_endpoint():
    """Log12: Full pipeline metrics — ingestion, ML, clustering, timings."""
    from core.metrics import get_metrics
    data = get_metrics()
    # Add API-level stats
    data["api"] = {
        "total_analyses": _app_state["total_analyses"],
        "avg_latency_ms": round(
            (_app_state["total_latency_ms"] / _app_state["total_analyses"])
            if _app_state["total_analyses"] > 0 else 0.0, 1
        ),
        "last_analysis_at": _app_state["last_analysis_at"],
    }
    return data


async def _run_analysis(req: AnalyzeRequest, request: Request):
    """
    Full multi-mode risk analysis.

    Returns AIR, SEA, ROAD risk scores with evidence, zones, and source URLs.
    """
    start = time.time()

    client = _app_state.get("mongo_client")
    if client is None:
        raise HTTPException(503, "MongoDB not connected")

    from config.settings import MONGO_DB, MONGO_COLLECTION
    collection = client[MONGO_DB][MONGO_COLLECTION]

    try:
        from core.orchestrator import analyze_multi_mode_v5
        result = await analyze_multi_mode_v5(
            origin=req.origin,
            destination=req.destination,
            mongo_collection=collection,
            radius_km=req.radius_km,
            min_confidence=req.min_confidence,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception:
        logger.exception("Analysis failed request_id=%s", getattr(request.state, "request_id", None))
        raise HTTPException(500, "Analysis failed")

    latency_ms = (time.time() - start) * 1000
    _app_state["total_analyses"] += 1
    _app_state["total_latency_ms"] += latency_ms
    _app_state["last_analysis_at"] = datetime.now(timezone.utc).isoformat()

    result["latency_ms"] = round(latency_ms, 1)
    result["request_id"] = getattr(request.state, "request_id", None)
    return result


@app.post("/api/v1/analyze")
async def analyze_v1(req: AnalyzeRequest, request: Request):
    return await _run_analysis(req, request)


@app.post("/analyze")
async def analyze(req: AnalyzeRequest, request: Request):
    return await _run_analysis(req, request)
