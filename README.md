# foto-klass — Stage 1 prototype

Single-file Python pipeline that converts auto-parts source photos into Google
Merchant-ready `1000×1000 WebP q90` outputs on pure white background, with
visual debug artifacts at every stage.

This is a **prototype only**. No infrastructure, no S3, no queues. Pure
local-folder-in → local-folder-out.

---

## What the pipeline does

```
input/<img>
   │
   ▼
1. Load + strip EXIF
   │
   ▼
2. Detect content type  (heuristic: near-white ratio, saturation, edges)
   │
   ├─ photo  ──▶ watermark detect (Florence-2) → article-pattern filter
   │             → LaMa inpaint  → DAT2 upscale (if long side <900px)
   │             → BiRefNet alpha → guided-filter refine → composite on white
   │             → crop to product
   │
   └─ drawing ─▶ trim white margins (BiRefNet skipped — line drawings don't
                  segment cleanly with product-photo trained models)
   │
   ▼
3. Pad to square (product ≈85% of canvas) → resize 1000×1000 → WebP q90
```

False-positive guard on watermark removal: Florence-2 finds candidate regions,
then Florence-2 OCR reads the text inside each. Text matching article patterns
(`^[A-Z0-9\-]{4,}$`, `Ø 47`, `47 mm`, etc.) is **kept**. Text matching phone /
URL / email patterns is **removed**. Ambiguous regions are logged as
`keep_unknown` and shown in the debug overlay.

---

## Models used (all commercially OK)

| Stage | Model | License |
|---|---|---|
| Watermark detection + OCR | `microsoft/Florence-2-base` (or `large`) | MIT |
| Watermark inpainting | LaMa (via `simple-lama-inpainting`) | Apache-2.0 |
| Upscale | `Phips/4xNomosWebPhoto_RealPLKSR` (via `spandrel`) | CC-BY-4.0 |
| Background removal | `ZhengPeng7/BiRefNet-matting` (or HR) | MIT |
| Alpha refinement | Guided filter (OpenCV, no model) | — |

---

## Install (Mac, Apple Silicon — MPS)

```bash
cd /Users/dmytrovoitenko/Projects/foto-klass
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Then download the DAT2 upscaler weights (~190 MB):

```bash
mkdir -p weights
pip install huggingface_hub
huggingface-cli download Phips/4xNomosWebPhoto_RealPLKSR \
    4xNomosWebPhoto_RealPLKSR.safetensors --local-dir weights
```

Florence-2 and BiRefNet weights download automatically on first run (~2 GB total)
to `~/.cache/huggingface/`.

First run will take 1-3 minutes for model loading + downloads. Subsequent runs:
~10 seconds startup.

---

## Install (RunPod, RTX 4090 — CUDA)

Spin up a RunPod Pod with image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
(or any CUDA 12.x Python 3.11 image). Open Jupyter / Web Terminal:

```bash
cd /workspace
git clone <your-repo>   # or: rsync from Mac
cd foto-klass
pip install -r requirements.txt

mkdir -p weights
huggingface-cli download Phips/4xNomosWebPhoto_RealPLKSR \
    4xNomosWebPhoto_RealPLKSR.safetensors --local-dir weights
```

For RunPod you can switch to the larger / higher-quality model variants — edit
`config.yaml`:

```yaml
florence_model: "microsoft/Florence-2-large"
birefnet_model: "ZhengPeng7/BiRefNet_HR-matting"
bg_removal_input_size: 2048
```

---

## Run

```bash
# Mac (MPS) — auto-detects device
python prototype.py --input sample/ --output output/

# Force CPU (slower but works without MPS/CUDA)
python prototype.py --input sample/ --output output/ --device cpu

# RunPod (CUDA)
python prototype.py --input sample/ --output output/ --device cuda

# Process only first 2 images for a quick check
python prototype.py --input sample/ --output output/ --limit 2
```

---

## Output structure

```
output/
├── 1_orig.webp              ← final 1000×1000 WebP q90 (publishable)
├── 3_orig.webp
├── 6_orig.webp
├── ...
├── summary.json             ← aggregate metrics + per-image record
└── debug/
    └── 1_orig/
        ├── 01_loaded.png            full RGB after EXIF strip
        ├── 02_watermark_bboxes.png  Florence-2 bboxes colored by decision:
        │                              red=remove, green=keep_article, yellow=keep_unknown
        ├── 03_watermark_mask.png    union of bboxes marked "remove" (dilated)
        ├── 04_inpainted.png         after LaMa
        ├── 05_upscaled.png          after DAT2 (or skipped)
        ├── 06_alpha_raw.png         BiRefNet alpha mask
        ├── 07_alpha_refined.png     after guided filter
        ├── 08_composed.png          RGB composited on white
        ├── 11_final.png             same as output/1_orig.webp (PNG, lossless for inspection)
        └── metrics.json             per-stage timings, decisions, review flag
```

For drawings: `02_drawing_trimmed.png` instead of the watermark/inpaint chain.

---

## What to check in the results

1. Open `output/<name>.webp` for each sample — does it look publishable?
2. Open `debug/<name>/02_watermark_bboxes.png` — did Florence-2 correctly
   classify ALKO / 308014 / Ø47mm as `keep_article` (green) and only the actual
   watermark/URL/phone as `remove` (red)?
3. Open `debug/<name>/07_alpha_refined.png` — is the edge clean on metal/chrome?
4. Open `summary.json` — what's the average `total_ms`? Anything in `review`?

Tuning knobs live in `config.yaml.example` (copy to `config.yaml`).

---

## Known limitations of Stage 1

- **No SAM 2.1**. Watermark masks are dilated bboxes, not pixel-precise.
  Acceptable for prototype; revisit if LaMa leaves visible patches.
- **No FBA Matting**. Guided filter is much lighter but won't kill every halo
  on chrome. If you see fringing on photos like `1_orig.webp`, we'll bolt FBA
  on for Stage 2.
- **No batching**. One image at a time. Batching comes in Stage 6 per brief.
- **Florence-2-base default**. Smaller, ~1-2s/inference on MPS. Switch to
  `large` on RTX 4090 for better small-text OCR.
- **Drawing classifier is heuristic, not learned**. Tweak thresholds in
  `config.yaml` if classifications look wrong (`drawing_*` keys).

---

## Troubleshooting

**`RuntimeError: MPS backend out of memory`** — drop `bg_removal_input_size`
to 768 in `config.yaml`, or run on CPU once to confirm the pipeline works.

**Florence-2 raises `KeyError` on `<CAPTION_TO_PHRASE_GROUNDING>`** — your
transformers version is too old. `pip install -U transformers>=4.44`.

**`spandrel` complains it can't load the .pth file** — verify the download:
`ls -lh weights/4xRealWebPhoto_v4_dat2.pth` should be ~190 MB. Re-run the
`huggingface-cli download` if smaller.

**`simple-lama-inpainting` slow on Mac** — expected, LaMa runs on CPU on
Apple Silicon (no MPS port). Should be 1-3s per inpaint, not catastrophic.
