# Geo Risk Engine Architecture

## Production Runtime

```text
Client
  -> FastAPI API service
  -> MongoDB

Background worker service
  -> realtime ingestion + ML enrichment + canonical clustering
  -> MongoDB
```

The API and worker share MongoDB only. The API does not run live ingestion or heavy ML inference.

## Services

| Service | Responsibility |
|---|---|
| `api` | HTTP validation, route analysis, health/readiness/metrics |
| `worker` | GDELT/RSS/API ingestion, ML enrichment, geocoding, clustering, Mongo writes |
| `mongo` | Geo-indexed canonical incidents and enriched events |
| `redis` | Optional future lightweight cache profile; not required for current runtime |

## Data Flow

```text
Worker:
fetch -> validate -> verify -> relevance prefilter -> batch ML -> geocode -> cluster -> MongoDB

API:
request -> validate -> generate route -> geo query MongoDB -> risk aggregation -> response
```

## Runtime Guarantees

- One-command local CLI remains: `python run_live.py --origin "Mumbai, India" --destination "Dubai, UAE"`
- API remains lightweight and read-only.
- Worker failure does not stop API reads from existing data.
- Source failures are isolated and non-fatal.
- MongoDB TTL keeps event storage bounded.
