# CLAUDE.md — agent guide for foto-klass

Project-level instructions for any AI coding agent (Claude Code, Cursor, etc.)
working with this repo. Keep this file fresh — it's loaded automatically on
every session.

---

## What this project is

Automated pipeline to clean up ~13M auto-parts photos for **Google Merchant**.
Each input photo becomes a **1000×1000 WebP q90** on white background.

Owner: vda@allzap.pro (chats in Russian/Ukrainian). Cofounder using this repo:
@virtus1k (admin).

## Project state (last update: 2026-05-18)

- **Stage 0** — research and model selection: ✅ done
- **Stage 1** — single-file prototype `prototype.py`: ✅ done, locked
- **Stage 2** — orchestrator API (`api/`, `migrations/`, `docker-compose.yml`):
  scaffolded but **deferred — not deployed**. Operators work directly with
  `prototype.py` for now.
- **Stage 3-8** — not started

## How to work in this repo

**Two pipelines, both in `prototype.py`** (one file, dispatched by `--mode`):

```bash
# Local venv (Mac with MPS or any CUDA machine):
python prototype.py --input sample/ --output output/ --mode full     # AI cleanup
python prototype.py --input sample/ --output output/ --mode resize   # bulk path
```

| Mode | Pipeline | Use for |
|---|---|---|
| `full` | Florence-2 watermark detect + LaMa inpaint + Nomos SR + BiRefNet bg removal + framing + WebP | small batches that go to Google Merchant |
| `resize` | content trim + Nomos SR + framing + WebP | bulk 13M run — watermarks remain (decision B) |

Outputs always: WebP q90, 1000×1000, white RGB(255,255,255), product ~85% of canvas.

**Idempotency:** SHA256-cached in `output/.cache/{hash}_{mode}.json`. Re-running
skips already-processed inputs unless `--force` is passed.

## Locked-in tech choices (do NOT swap without owner approval)

- **Upscaler:** `Phips/4xNomosWebPhoto_RealPLKSR` (CC-BY-4.0). Chosen after
  side-by-side against DAT2, Real-ESRGAN, HAT-L, and iloveimg.com.
- **Watermark detect:** `microsoft/Florence-2-base` (full mode only)
- **Inpaint:** Big-LaMa via `simple-lama-inpainting` (full mode only)
- **Bg removal:** `ZhengPeng7/BiRefNet-matting` (full mode only)
- **Skip rule:** SR is skipped when input long-side ≥ 1000 (Nomos doesn't add
  detail above target; Lanczos downscale handles it)

See `MEMORY.md` and `docs/` if any decision log files appear; otherwise the
canonical record is `PROJECT_BRIEF.md`.

## Where heavy weights live

`weights/` directory (gitignored). Download:

```bash
mkdir -p weights
huggingface-cli download Phips/4xNomosWebPhoto_RealPLKSR \
    4xNomosWebPhoto_RealPLKSR.safetensors --local-dir weights
```

Florence-2 and BiRefNet auto-download to `~/.cache/huggingface/` on first run
(~2 GB total).

## Sample images in `sample/`

7 files covering the variety we'll see in production:
- `1_orig.webp` — photo with heavy "A.F.6" watermark pattern
- `3_orig.webp`, `3_trim.webp` — photo with subtle "AL-KO" watermark + a real
  "AL-KO" sticker on the part (false-positive risk for the watermark detector)
- `6_orig.webp`, `9_orig.webp`, `10_orig.webp` — line drawings with article
  numbers (308014, 313374, etc.) and dimensions
- `3_orig_ilove.jpg` — same source upscaled via iloveimg.com, as a quality
  reference target

If you change the pipeline, **always** test on these 7 first.

## Where things run

| Machine | Role | Status |
|---|---|---|
| owner's Mac M-series (MPS) | dev + small batches | ✅ tested |
| RunPod RTX 4090 (on-demand) | bulk 13M-scale runs | not yet — `Dockerfile.gpu` not built |
| Hetzner VPS 135.181.148.66 | future orchestrator host | scaffold ready, not deployed |
| Hetzner Object Storage (S3) | photo storage (inputs + `/up/` outputs) | credentials not yet shared |

On Mac MPS, BiRefNet has to run on CPU because `torchvision.ops.deform_conv2d`
has no MPS implementation. Pipeline handles this automatically.

## How owners want to be addressed

- Russian/Ukrainian
- Concise. No fluff, no "great question!" intros.
- **Always stop at stage boundaries** and wait for explicit "ок, дальше". See
  the workflow rule in the prototype brief: small samples first, full runs
  only after visual approval.
- Voice messages from the owner can be transcribed via the global
  `transcribe-voice ~/Downloads/<file>.m4a` command (see global CLAUDE.md).

## Things to NEVER do without explicit approval

- Force-push or rewrite history on `main`
- Commit anything matching `*.pth`, `*.safetensors`, `output*/`, `weights/`,
  `.env*` (already in `.gitignore` — verify before committing)
- Add new heavy ML deps to `requirements.txt` (locks in a model choice —
  needs owner sign-off)
- Re-process the entire `sample/` set if only one file changed — use the
  SHA256 cache or pass a specific file
- Deploy anything to 135.181.148.66 without explicit "поехали Stage 2"

## When in doubt

1. Re-read `PROJECT_BRIEF.md` (the source of truth from the owner)
2. Re-read `RUNBOOK.md` for operator workflows
3. Ask. The owner prefers a 2-line clarifying question over a 100-line
   guess.
