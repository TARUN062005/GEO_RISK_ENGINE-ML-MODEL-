"""
app/routes/analyze.py
---------------------
FastAPI Route Handler — /analyze endpoint (Log3)

Thin HTTP layer only:
  - Validates request via pydantic
  - Calls core.orchestrator.analyze_route_real()
  - Returns structured JSON

NO ML logic. NO model imports. Read-only API (Log2 principle preserved).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    origin: str = Field(..., min_length=2, examples=["Mumbai, India"])
    destination: str = Field(..., min_length=2, examples=["Cairo, Egypt"])
    radius_km: float = Field(default=50.0, ge=10.0, le=500.0)
    min_confidence: float = Field(default=0.50, ge=0.0, le=1.0)
    mode: str | None = Field(default=None, description="Transport mode: air, sea, road, or null for all")


class EventAlert(BaseModel):
    headline: str
    location: list[float]       # [lat, lon]
    distance_km: float
    intensity: float
    label: str


class AnalyzeResponse(BaseModel):
    origin: str
    destination: str
    alerts_count: int
    safety_score: float
    status: str
    total_distance_km: float
    events: list[EventAlert]
    risk_detail: dict


# ---------------------------------------------------------------------------
# Dependency: MongoDB collection
# ---------------------------------------------------------------------------

async def get_mongo_collection():
    """
    Dependency injector for the MongoDB collection.
    In production wire this to app.deps.get_db().
    """
    from motor.motor_asyncio import AsyncIOMotorClient  # type: ignore
    import os
    uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
    client = AsyncIOMotorClient(uri)
    return client["geo_risk"]["geo_events"]


# ---------------------------------------------------------------------------
# Route handler
# ---------------------------------------------------------------------------

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_route(
    body: AnalyzeRequest,
    collection=Depends(get_mongo_collection),
):
    """
    Analyze geopolitical risk for a route between two locations.

    - Geocodes origin/destination dynamically (no hardcoded values)
    - Queries MongoDB for real stored events near route
    - Returns risk score, band, and event alerts
    """
    try:
        from core.orchestrator import analyze_route_real
        result = await analyze_route_real(
            origin=body.origin,
            destination=body.destination,
            mongo_collection=collection,
            radius_km=body.radius_km,
            min_confidence=body.min_confidence,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="Pipeline error")


# ---------------------------------------------------------------------------
# Log5 — Evidence-Enriched Multi-Mode Endpoint
# ---------------------------------------------------------------------------

class EvidenceEvent(BaseModel):
    """Single event with full evidence provenance (Log5)."""
    headline: str
    source_url: str = ""
    image_url: str | None = None
    publisher: str = ""
    location: list[float]          # [lat, lon]
    distance_km: float
    zone: str | None = None
    zones: list[str] = []
    confidence: float
    intensity: float
    label: str
    credibility: float | None = None
    published_at: str = ""


class ZoneIntersection(BaseModel):
    """Route-zone intersection result (Log5)."""
    zone: str
    category: str
    description: str = ""
    min_distance_km: float
    intersects: bool = True
    closest_waypoint_idx: int = 0


class ModeResult(BaseModel):
    """Risk result for a single transport mode (Log5)."""
    status: str
    alerts: int = 0
    risk_score: float | None = None
    safety_score: float | None = None
    distance_km: float | None = None
    message: str = ""
    zone_intersections: list[ZoneIntersection] = []
    events: list[EvidenceEvent] = []


class MultiModeV5Response(BaseModel):
    """Full multi-mode evidence-enriched response (Log5)."""
    origin: str
    destination: str
    recommended_mode: str
    modes: dict[str, ModeResult]
    analyzed_at: str


@router.post("/analyze/v5", response_model=MultiModeV5Response)
async def analyze_multi_mode_v5_endpoint(
    body: AnalyzeRequest,
    collection=Depends(get_mongo_collection),
):
    """
    Log5: Evidence-enriched multi-mode risk analysis.

    Returns air/sea/road risk with:
      - Zone intersection detection
      - Verified source links + images
      - Credibility scores per event
      - Publisher attribution
    """
    try:
        from core.orchestrator import analyze_multi_mode_v5
        result = await analyze_multi_mode_v5(
            origin=body.origin,
            destination=body.destination,
            mongo_collection=collection,
            radius_km=body.radius_km,
            min_confidence=body.min_confidence,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="Pipeline error")


@router.post("/analyze/v5/single")
async def analyze_single_mode_v5_endpoint(
    body: AnalyzeRequest,
    collection=Depends(get_mongo_collection),
):
    """
    Single-mode risk analysis — Phase 1 Optimization.

    Analyzes ONLY the requested transport mode (air/sea/road).
    ~3× faster than /analyze/v5 which computes all 3 modes.

    Requires `mode` field in request body.
    """
    if not body.mode or body.mode not in ("air", "sea", "road"):
        raise HTTPException(
            status_code=422,
            detail=f"mode must be 'air', 'sea', or 'road' (got: {body.mode!r})",
        )
    try:
        from core.orchestrator import analyze_single_mode_v5
        result = await analyze_single_mode_v5(
            origin=body.origin,
            destination=body.destination,
            mode=body.mode,
            mongo_collection=collection,
            radius_km=body.radius_km,
            min_confidence=body.min_confidence,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="Pipeline error")

