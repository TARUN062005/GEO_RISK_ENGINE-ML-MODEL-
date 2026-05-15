# Changelog

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
