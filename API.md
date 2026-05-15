# Geo Risk Engine API

Base URL locally:

```text
http://localhost:8000
```

## Endpoints

### `GET /health`

Liveness check. Returns `ok` when the API process is running.

### `GET /ready`

Readiness check. Verifies MongoDB connectivity and startup index setup.

### `GET /metrics`

Returns lightweight JSON metrics:

- ingestion counters
- source counts
- ML batch counts
- cache hit/miss counts
- clustering reduction
- API latency

### `POST /analyze`

Production route-analysis endpoint.

Request:

```json
{
  "origin": "Mumbai, India",
  "destination": "Dubai, UAE",
  "radius_km": 50,
  "min_confidence": 0.5
}
```

Response includes:

- AIR / SEA / ROAD risk
- zones crossed
- evidence
- source URLs
- timestamps
- corroboration count
- credibility
- canonical incident grouping

Legacy routes are available under `/api/legacy/*`.

## Security Behavior

- Request bodies are size-limited by `MAX_REQUEST_BYTES`.
- Analysis calls are timeout-protected by `API_TIMEOUT_SECONDS`.
- Internal exceptions return safe error messages without stack traces.
- Every response includes `x-request-id`.
