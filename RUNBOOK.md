# foto-klass — RUNBOOK

Practical recipes for running the pipeline. Most operators (including AI
coding agents) should be able to do their day-to-day from this file plus
`CLAUDE.md`.

---

## Quick start (local machine — Mac or Linux)

```bash
git clone git@github.com:Allzap/foto-klass.git
cd foto-klass

# venv with Python 3.11 (3.12 also fine — adjust accordingly)
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Weights (~190 MB)
mkdir -p weights
huggingface-cli download Phips/4xNomosWebPhoto_RealPLKSR \
    4xNomosWebPhoto_RealPLKSR.safetensors --local-dir weights

# Smoke test on the 7 reference photos
python prototype.py --input sample/ --output output/ --mode resize
```

First run also pulls Florence-2 and BiRefNet from Hugging Face (~2 GB,
goes to `~/.cache/huggingface/`).

---

## Day-to-day commands

```bash
# Bulk path — for 13M run, no AI cleanup, watermarks remain
python prototype.py --input <folder> --output output_resize/ --mode resize

# AI cleanup — slow, for small high-quality batches
python prototype.py --input <folder> --output output/ --mode full

# Force re-process (ignore SHA256 cache)
python prototype.py --input <folder> --output output/ --mode full --force

# Process only first N images
python prototype.py --input <folder> --output output/ --mode resize --limit 50

# Use a specific device
python prototype.py --input <folder> --output output/ --mode resize --device cpu
```

Outputs land at `<output>/<name>.webp`. Per-image debug + metrics are at
`<output>/debug/<name>/`.

---

## What to check after a run

1. Open `<output>/<name>.webp` for several images. Looks publishable?
2. `<output>/summary.json` — review `processed`, `cached`, `errors`,
   `review_count`, `avg_total_ms`.
3. For anything with `review_reason != null`, open
   `<output>/debug/<name>/02_watermark_bboxes.png` (full mode) and the
   `metrics.json` next to it.

---

## Working with RunPod (bulk 13M scale)

RunPod RTX 4090 is the production environment. Local Mac/CPU is for dev only.

### Manual flow (current MVP — until Stage 4 lands)

1. Open <https://www.runpod.io/console/deploy>
2. Pick **RTX 4090** in Community Cloud (cheapest, ~$0.34/hour)
3. Container image: use the official PyTorch 2.5 + CUDA 12.4 image for now
   (`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`). Custom image
   with pre-baked weights is Stage 4 work.
4. Start the pod, open **Web Terminal** (or SSH).
5. Inside:

```bash
cd /workspace
git clone git@github.com:Allzap/foto-klass.git
cd foto-klass
pip install -r requirements.txt
mkdir -p weights
huggingface-cli download Phips/4xNomosWebPhoto_RealPLKSR \
    4xNomosWebPhoto_RealPLKSR.safetensors --local-dir weights

# Mount or sync your inputs first (rsync from your Mac, or boto3 from Hetzner S3)
python prototype.py --input inputs/ --output outputs/ --mode resize --device cuda
```

6. When done, copy outputs back (rsync, or upload to S3), then **stop the pod**
   in RunPod UI to stop billing.

Expected speed on RTX 4090:
- `--mode resize`: ~150-300 ms / photo (depending on whether Nomos runs)
- `--mode full`: ~3-5 s / photo

---

## Hetzner Object Storage (S3) workflow — when set up

When `HETZNER_S3_*` is in `.env`, you can list + sync inputs from S3 using
`boto3` directly. Example one-shot script (write this when you need it):

```python
import boto3, os
s3 = boto3.client(
    "s3",
    endpoint_url=os.environ["HETZNER_S3_ENDPOINT"],
    aws_access_key_id=os.environ["HETZNER_S3_ACCESS_KEY"],
    aws_secret_access_key=os.environ["HETZNER_S3_SECRET_KEY"],
)
# list under prefix
for page in s3.get_paginator("list_objects_v2").paginate(
    Bucket=os.environ["HETZNER_S3_BUCKET"], Prefix="inputs/parts/"):
    for obj in page.get("Contents", []):
        print(obj["Key"])
```

---

## Working with the GitHub repo

`main` is the integration branch. Standard PR flow:

```bash
git checkout -b virtus/<short-task-name>
# work, commit
git push -u origin virtus/<short-task-name>
gh pr create
```

Both vda and virtus1k are admins on `Allzap/foto-klass`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `MPS backend out of memory` | BiRefNet input too big | Drop `bg_removal_input_size: 768` in `config.yaml`, or run on CPU |
| `kornia not found` | requirement missing in older clone | `pip install -r requirements.txt` |
| `deform_conv2d not implemented for MPS` | BiRefNet on Mac MPS | Pipeline auto-routes BiRefNet to CPU; should not appear |
| Output WebP is a blank white square | watermark detector marked whole image as watermark | Already fixed via "whole-image bbox guard"; if it recurs, open `debug/<name>/metrics.json` and check `watermark_regions[].decision` |
| First run hangs at "Loading Florence-2" | HF download still in progress | Wait — ~2 GB on first run, cached after |

---

## Future: orchestrator on the VPS (deferred)

`api/`, `migrations/`, `docker-compose.yml`, `Dockerfile.api` scaffold an
orchestrator that exposes a REST API for batch job management. It's **not
deployed** today — operators run `prototype.py` directly. We'll wire it up
when we move past 1k-scale batches.
