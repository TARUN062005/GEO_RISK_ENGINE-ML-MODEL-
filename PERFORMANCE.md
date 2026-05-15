# Performance Notes

## Current Optimizations

| Area | Optimization |
|---|---|
| ML | Singleton model loading in ingestion |
| ML | Batch classification after cheap prefiltering |
| API | No heavy ML model load in API process |
| Geo | Geocode LRU cache |
| Routing | Bounded in-process route cache |
| Storage | 2dsphere and timestamp indexes |
| Retention | MongoDB TTL index on `ingested_at` |
| Dedup | Source URL/hash dedup plus canonical clustering |

## Expected Runtime

| Scenario | Target |
|---|---:|
| Cached API analysis | seconds-level |
| Subsequent local `run_live.py` | <= 10-15s when data/cache are warm |
| Fresh ingestion | source/geocoding dependent |

## Bottlenecks

- Cold Nominatim geocoding is the main first-run cost.
- Free news APIs and GDELT can rate-limit.
- CPU-only transformer inference is acceptable only when batched and prefiltered.

## Metrics To Watch

- `ml_batch_seconds`
- `analysis_seconds`
- `ingestion_cycle_seconds`
- `geocode_cache_hits/misses`
- `route_cache_hits/misses`
- `clusters_duplicates_merged`
- `events_errors`
