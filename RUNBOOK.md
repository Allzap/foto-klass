# foto-klass — RUNBOOK

Quick reference for running the orchestrator and submitting jobs.

The orchestrator lives on **135.181.148.66:8000** (Hetzner VPS). RunPod GPU
pods are started on-demand to do the actual image processing — they read job
manifests from S3 and write results back.

---

## 1. Server access

```bash
ssh root@135.181.148.66
```

Project lives at `/opt/foto-klass`. All operational state is in Docker volumes
managed by `docker compose`.

---

## 2. Daily commands

```bash
cd /opt/foto-klass

# status
docker compose ps
docker compose logs -f api          # live API logs
docker compose logs -f postgres

# restart after pulling new code
git pull
docker compose build api
docker compose up -d

# database shell
docker compose exec postgres psql -U fotoklass

# run a migration manually (normally done at API container startup)
docker compose exec api alembic -c migrations/alembic.ini upgrade head
```

---

## 3. Submitting a job via curl

Get the API key from `.env` on the server (`grep API_KEY /opt/foto-klass/.env`).

```bash
export API="http://135.181.148.66:8000"
export TOKEN="<API_KEY from .env>"

# health check (no auth needed)
curl -s $API/health | jq

# create a resize-only job (mode="resize") for an S3 prefix
curl -s -X POST $API/jobs \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prefix": "inputs/parts/",
    "mode": "resize",
    "notes": "13M bulk run, batch #1"
  }' | jq

# list jobs
curl -s $API/jobs -H "Authorization: Bearer $TOKEN" | jq

# job details
curl -s $API/jobs/<job-id> -H "Authorization: Bearer $TOKEN" | jq

# progress
curl -s $API/jobs/<job-id>/progress -H "Authorization: Bearer $TOKEN" | jq

# stop / resume
curl -s -X POST $API/jobs/<job-id>/stop  -H "Authorization: Bearer $TOKEN" | jq
curl -s -X POST $API/jobs/<job-id>/resume -H "Authorization: Bearer $TOKEN" | jq

# cost
curl -s $API/stats/cost -H "Authorization: Bearer $TOKEN" | jq
```

Interactive docs: <http://135.181.148.66:8000/docs>

---

## 4. Launching a RunPod GPU pod (manual, until auto-launch lands)

The current MVP does **not yet auto-launch pods**. After creating a job, you
launch a pod by hand:

1. Open <https://www.runpod.io/console/deploy>
2. **GPU**: RTX 4090 (community cloud, cheapest)
3. **Template**: select the foto-klass image (TBD — Stage 4 will publish to
   ghcr.io)
4. **Environment variables** (set inside the pod):
   ```
   API_BASE=http://135.181.148.66:8000
   API_KEY=<same as on the server>
   JOB_ID=<the job id you just created>
   HETZNER_S3_ENDPOINT=...
   HETZNER_S3_ACCESS_KEY=...
   HETZNER_S3_SECRET_KEY=...
   HETZNER_S3_BUCKET=...
   ```
5. **Start** the pod. The entrypoint will:
   - fetch the photo list for this job from the API
   - process each photo (Nomos x4 if needed → resize → WebP q90)
   - upload results to `up/<original_key>` in the bucket
   - report progress back to the API every N photos

When the job finishes, the pod terminates itself. If you stop a job via the
API, the next progress report from the pod will see `status=stopped` and the
pod will exit gracefully.

---

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `health` returns `db: down` | Postgres crashed | `docker compose logs postgres` |
| `health` returns `runpod: down` | Bad API key | `grep RUNPOD_API_KEY /opt/foto-klass/.env`, test with curl to RunPod |
| `health` returns `s3: not_configured` | Hetzner S3 creds empty in `.env` | Fill them in, `docker compose up -d api` |
| 401 on every request | Wrong/missing `Authorization` header | `Authorization: Bearer <API_KEY>` exactly |
| Job stuck in `queued` forever | No pod has been started for it | Launch a pod manually (see §4) |

---

## 6. Backups

Postgres data lives in the `pgdata` Docker volume:

```bash
# dump
docker compose exec postgres pg_dump -U fotoklass fotoklass > /root/fk-$(date +%F).sql
# restore
cat dump.sql | docker compose exec -T postgres psql -U fotoklass fotoklass
```
