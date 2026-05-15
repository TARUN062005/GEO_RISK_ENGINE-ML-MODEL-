# Changelog

## Log14

- Fixed the runtime route-generation integration bug in `analyze_multi_mode_v5`.
- Root cause: the v5 orchestrator path referenced `get_or_generate_route` inside async route-generation lambdas but did not import it, causing `NameError` for air, sea, and road routes.
- Affected file: `core/orchestrator.py`.
- Fix: restored the missing import from `core.routing.cache` in the active v5 analysis function and logged successful per-mode route generation before geo queries.
- Runtime before: `[air] route generation failed: name 'get_or_generate_route' is not defined`, repeated for sea and road; analysis returned UNKNOWN/N/A/0 alerts.
- Runtime after: route generation proceeds through the cache wrapper for all three modes, enabling waypoint generation, zone checks, Mongo geo queries, event evidence, and risk scoring.
- Verification: run `python run_live.py --origin "Mumbai, India" --destination "Dubai, UAE" --freshness 0` and confirm route logs appear before `Geo query returned X unique events.`

## Log13

- Added production API/worker separation in Docker Compose.
- Hardened FastAPI with `/health`, `/ready`, `/metrics`, and `/analyze`.
- Added request IDs, request-size limits, timeout handling, configurable CORS, and safe error responses.
- Added JSON logging helper.
- Added MongoDB readiness/index validation during API startup.
- Added worker startup env validation and secret-redacted logs.
- Added production Dockerfile using a non-root user.
- Added GitHub Actions workflow for compile validation, tests, Docker Compose validation, and Docker build validation.
- Added deployment documentation split across architecture, API, deployment, performance, and changelog docs.

## Previous Logs

See `MODEL_ARCHITECTURE.md` for Log1 through Log12.
