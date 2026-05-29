# Deployment Guide

## Step 1 - Push Project To GitHub

Initialize Git if needed:

```bash
git init
git add .
git status
git commit -m "production hardening log13"
```

Create a new GitHub repository, then push:

```bash
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

`.gitignore` excludes `.env`, virtualenvs, caches, and logs. Never commit `.env`.

## Step 2 - Configure `.env` Safely

Create local config:

```bash
cp .env.example .env
```

Required:

```env
MONGO_URI=mongodb://localhost:27017
MONGO_DB=geo_risk
MONGO_COLLECTION=geo_events
```

Optional:

```env
NEWSAPI_KEY=
GNEWS_KEY=
CORS_ALLOW_ORIGINS=*
API_TIMEOUT_SECONDS=45
MAX_REQUEST_BYTES=65536
TTL_EXPIRY_SECONDS=259200
```

Production secrets go into the hosting provider's environment variable UI, not GitHub.

## Step 3 - Deploy Free

Recommended free/low-cost platforms:

| Platform | Best For | Notes |
|---|---|---|
| Railway | Easiest API + worker + Mongo-style service deployment | Free tier limits change; watch sleep/usage caps |
| Render | Simple web service + background worker | Free services may sleep; background workers may require paid tier depending on current policy |
| MongoDB Atlas | Best free MongoDB option | Use M0 free cluster and connection URI |

Best free setup:

1. API on Railway or Render.
2. Worker as a separate Railway/Render background service.
3. MongoDB Atlas M0 free cluster.

## Step 4 - Deploy API Service

Use Docker deployment.

Startup command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT --proxy-headers
```

Environment variables:

```env
APP_ENV=production
MONGO_URI=<atlas-or-provider-uri>
MONGO_DB=geo_risk
MONGO_COLLECTION=geo_events
CORS_ALLOW_ORIGINS=https://your-frontend.example
```

Verify:

```bash
curl https://<api-host>/health
curl https://<api-host>/ready
```

## Step 5 - Deploy Worker Service (Background Worker)

The worker behaves like a continuous background consumer and must be deployed as a **Render Background Worker** (not a Web Service). 

### Key Worker Deployment Characteristics:
* **No Open Ports Required:** The background worker does NOT listen on any HTTP port. Render's Background Worker service does not provision public URLs or expect HTTP port binds.
* **Continuous Process:** Runs indefinitely executing fetching, ML classification, geocoding, and DB writes.
* **Automatic Restart:** Restarts automatically if the process exits or encounters a fatal crash.

### Startup Command:
Deploy using the exact same Docker image as the API, but override the start command:
```bash
python -m ingestion.realtime_worker --interval 180
```

### Environment Variables:
Copy all database and API-key environment variables from the API Service to the Background Worker:
* `APP_ENV=production`
* `MONGO_URI=<your-mongodb-atlas-srv-uri>`
* `NEWSAPI_KEY=<your-key>`
* `GNEWS_KEY=<your-key>`

## Step 6 - MongoDB Setup

Free options:

| Option | Recommendation |
|---|---|
| MongoDB Atlas M0 | Recommended |
| Railway Mongo-compatible service | Convenient if available |
| Render external Mongo | Use only if supported by your plan |

Atlas setup:

1. Create an M0 cluster.
2. Create a database user.
3. Add network access for the host provider.
4. Copy the connection URI.
5. Set `MONGO_URI` in API and worker service env vars.

Indexes are ensured automatically on API/worker startup:

- `location` 2dsphere
- `published_at`
- `ingested_at` TTL
- `canonical_event_id`
- `corroboration_count`
- `source_url`
- ML label/intensity

Retention defaults to 72 hours:

```env
TTL_EXPIRY_SECONDS=259200
```

For 24 hours:

```env
TTL_EXPIRY_SECONDS=86400
```

## Step 7 - Verify Deployment

Health:

```bash
curl https://<api-host>/health
```

Readiness:

```bash
curl https://<api-host>/ready
```

Metrics:

```bash
curl https://<api-host>/metrics
```

Analyze:

```bash
curl -X POST https://<api-host>/analyze \
  -H "Content-Type: application/json" \
  -d '{"origin":"Mumbai, India","destination":"Dubai, UAE"}'
```

Worker:

- Check logs for `Real-time ingestion worker started`.
- Check logs for `Ingestion cycle complete`.
- Check MongoDB document count in `geo_risk.geo_events`.

## Step 8 - Production Recommendations

- Run one API instance and one worker on free tier.
- Keep `UVICORN_WORKERS=1` on low-memory instances.
- Keep `INGEST_INTERVAL_SECONDS=180` or higher to reduce API quota usage.
- Use Atlas TTL retention to avoid unbounded storage growth.
- Set `CORS_ALLOW_ORIGINS` to your frontend domain in production.
- Monitor `/metrics` and provider logs.
- Scale API separately from worker only after traffic requires it.

## Local Docker Verification

```bash
docker compose build
docker compose up -d mongo
docker compose up -d api worker
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"origin":"Mumbai, India","destination":"Dubai, UAE"}'
```

One-command local CLI remains:

```bash
python run_live.py --origin "Mumbai, India" --destination "Dubai, UAE"
```
