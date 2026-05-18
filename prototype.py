#!/usr/bin/env python3
"""
prototype.py — Stage 1 prototype pipeline for foto-klass (car-parts photo cleanup).

Single-file pipeline: load → content-type detect → branch:
  PHOTO   : watermark detect + LaMa inpaint → DAT2 upscale → BiRefNet alpha →
            guided-filter refine → composite on white → crop + pad → resize → WebP
  DRAWING : skip heavy steps; trim white margins → pad → resize → WebP

Output: WebP q90, 1000×1000, RGB(255,255,255) background.

Usage:
    python prototype.py --input sample/ --output output/ \
        [--device auto|mps|cuda|cpu] [--config config.yaml]
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import cv2
import numpy as np
import torch
import yaml
from loguru import logger
from PIL import Image, ImageOps
from tqdm import tqdm

# Silence noisy third-party warnings during prototype runs.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# MPS fallback to CPU for unsupported ops on Apple Silicon.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


# ──────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Pipeline mode
    # - "full":   AI cleanup (watermark inpaint, bg removal, alpha refine) + Nomos SR
    # - "resize": cheap path — content trim + Nomos SR + frame 85% + resize.
    #             Watermarks remain. Chosen for the 13M-photo bulk run.
    mode: Literal["full", "resize"] = "full"

    # Output
    output_size: int = 1000
    webp_quality: int = 90
    product_fill_ratio: float = 0.85
    padding_color: tuple[int, int, int] = (255, 255, 255)

    # Drawing classifier (heuristic).
    # Real drawings in sample/ have near_white > 0.95; photos with watermark
    # patterns can reach ~0.5. Keep threshold high to avoid false-positives.
    drawing_near_white_ratio: float = 0.85
    drawing_max_saturation: float = 10.0
    drawing_min_edge_density: float = 0.005

    # Watermark
    watermark_prompts: list[str] = field(default_factory=lambda: [
        "watermark text",
        "logo overlay covering product",
        "phone number",
        "website url",
        "repeating brand stamp pattern",
    ])
    article_keep_patterns: list[str] = field(default_factory=lambda: [
        r"^[A-Z0-9\-]{4,}$",
        r"^\d+$",
        r"Ø\s*\d+",
        r"\d+\s*mm$",
    ])
    watermark_text_patterns: list[str] = field(default_factory=lambda: [
        r"\+?\d[\d\s\-\(\)]{6,}",
        r"https?://|www\.",
        r"\.(com|ru|ua|net|org|by|kz)",
        r"@",
    ])
    watermark_iou_review_threshold: float = 0.25
    watermark_coverage_review_pct: float = 0.15

    # Upscale skip rule: if input long side already >= upscale_skip_long_side_px,
    # we skip Nomos entirely — Lanczos downscale is lossless from above-target.
    # Threshold == final output size (1000) was validated on 6_orig (2500x812):
    # Nomos took 46s on Mac MPS for zero visible improvement vs Lanczos.
    upscale_skip_long_side_px: int = 1000
    upscale_target_long_side_px: int = 1500

    # Background removal
    bg_removal_input_size: int = 1024

    # Model identifiers
    florence_model: str = "microsoft/Florence-2-base"
    birefnet_model: str = "ZhengPeng7/BiRefNet-matting"
    # Nomos PLKSR x4 — chosen after head-to-head against DAT2, Real-ESRGAN, HAT-L,
    # and the commercial iloveimg.com service. Best quality/speed/license trade-off.
    upscaler_weights_path: str = "weights/4xNomosWebPhoto_RealPLKSR.safetensors"

    @classmethod
    def load(cls, path: Optional[Path]) -> "Config":
        if path is None or not path.exists():
            return cls()
        data = yaml.safe_load(path.read_text()) or {}
        cfg = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        # tuples come back as lists from YAML
        if isinstance(cfg.padding_color, list):
            cfg.padding_color = tuple(cfg.padding_color)  # type: ignore[assignment]
        return cfg


# ──────────────────────────────────────────────────────────────────────────
# Device / model bundle
# ──────────────────────────────────────────────────────────────────────────

def select_device(arg: str) -> torch.device:
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(arg)


@dataclass
class Models:
    device: torch.device
    birefnet_device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    florence_processor: Any = None
    florence_model: Any = None
    birefnet: Any = None
    birefnet_transform: Any = None
    dat2: Any = None
    lama: Any = None


def load_models(cfg: Config, device: torch.device) -> Models:
    """Load models for the current mode. Only what's needed."""
    # BiRefNet uses torchvision.ops.deform_conv2d, which has no MPS implementation
    # and no MPS->CPU fallback as of torch 2.5. Force BiRefNet to CPU on Mac.
    birefnet_device = torch.device("cpu") if device.type == "mps" else device
    mdl = Models(device=device, birefnet_device=birefnet_device)
    dtype = torch.float32

    # Upscaler is shared between modes
    weights = Path(cfg.upscaler_weights_path)
    if weights.exists():
        logger.info(f"Loading upscaler: {weights.name}")
        from spandrel import ImageModelDescriptor, ModelLoader
        desc = ModelLoader().load_from_file(str(weights))
        assert isinstance(desc, ImageModelDescriptor), "Unexpected spandrel descriptor type"
        mdl.dat2 = desc.to(device).eval()
    else:
        logger.warning(
            f"Upscaler weights not found at {weights}. Using bilinear fallback. "
            "Download with: huggingface-cli download Phips/4xNomosWebPhoto_RealPLKSR "
            f"4xNomosWebPhoto_RealPLKSR.safetensors --local-dir {weights.parent}"
        )

    if cfg.mode == "resize":
        # Light mode — only the upscaler is needed.
        return mdl

    # full mode — load everything else
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoModelForImageSegmentation
    import torchvision.transforms as T

    logger.info(f"Loading Florence-2: {cfg.florence_model} (device={device})")
    mdl.florence_processor = AutoProcessor.from_pretrained(cfg.florence_model, trust_remote_code=True)
    mdl.florence_model = AutoModelForCausalLM.from_pretrained(
        cfg.florence_model, trust_remote_code=True, torch_dtype=dtype
    ).to(device).eval()

    logger.info(f"Loading BiRefNet: {cfg.birefnet_model} (device={birefnet_device})")
    mdl.birefnet = AutoModelForImageSegmentation.from_pretrained(
        cfg.birefnet_model, trust_remote_code=True
    ).to(birefnet_device).eval()
    mdl.birefnet_transform = T.Compose([
        T.Resize((cfg.bg_removal_input_size, cfg.bg_removal_input_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    logger.info("Loading LaMa inpainter")
    from simple_lama_inpainting import SimpleLama
    mdl.lama = SimpleLama()

    return mdl


# ──────────────────────────────────────────────────────────────────────────
# Image I/O
# ──────────────────────────────────────────────────────────────────────────

def load_image(path: Path) -> np.ndarray:
    """Load image as RGB uint8 ndarray (H, W, 3); strip EXIF orientation."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # respect orientation, drop EXIF
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def save_image(arr: np.ndarray, path: Path, fmt: str = "PNG", quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr)
    if fmt.upper() == "WEBP":
        img.save(path, "WEBP", quality=quality, method=6)
    else:
        img.save(path, fmt)


# ──────────────────────────────────────────────────────────────────────────
# Stage 1: content type classifier (drawing vs photo)
# ──────────────────────────────────────────────────────────────────────────

def detect_content_type(img: np.ndarray, cfg: Config) -> tuple[Literal["photo", "drawing"], dict]:
    """
    Heuristic on three signals:
      - near-white pixel ratio (drawing has lots of pure white)
      - mean HSV saturation (drawing is mostly grayscale)
      - Canny edge density (drawing has visible strokes, not blank canvas)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    near_white = float(((img > 240).all(axis=-1)).mean())
    saturation = float(hsv[..., 1].mean())
    edges = cv2.Canny(gray, 80, 180)
    edge_density = float((edges > 0).mean())

    is_drawing = (
        near_white >= cfg.drawing_near_white_ratio
        and saturation <= cfg.drawing_max_saturation
        and edge_density >= cfg.drawing_min_edge_density
    )
    return (
        "drawing" if is_drawing else "photo",
        {
            "near_white_ratio": round(near_white, 4),
            "saturation_mean": round(saturation, 2),
            "edge_density": round(edge_density, 4),
        },
    )


# ──────────────────────────────────────────────────────────────────────────
# Stage 2: watermark detection + inpainting (Florence-2 + LaMa)
# ──────────────────────────────────────────────────────────────────────────

def _florence_run(mdl: Models, image_pil: Image.Image, task: str, text_input: Optional[str] = None) -> dict:
    """Generic Florence-2 task runner."""
    prompt = task if text_input is None else f"{task}{text_input}"
    inputs = mdl.florence_processor(text=prompt, images=image_pil, return_tensors="pt").to(mdl.device)
    with torch.inference_mode():
        gen = mdl.florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )
    text = mdl.florence_processor.batch_decode(gen, skip_special_tokens=False)[0]
    parsed = mdl.florence_processor.post_process_generation(
        text, task=task, image_size=(image_pil.width, image_pil.height)
    )
    return parsed.get(task, {})


def detect_watermarks(image_pil: Image.Image, mdl: Models, cfg: Config) -> list[dict]:
    """
    Returns list of candidate watermark regions:
    [{bbox: [x1,y1,x2,y2], label: str, ocr_text: str, decision: "remove"|"keep_article"|"keep_unknown"}, ...]
    Only "remove" entries should be inpainted.
    """
    prompt_text = ". ".join(cfg.watermark_prompts) + "."
    result = _florence_run(mdl, image_pil, "<CAPTION_TO_PHRASE_GROUNDING>", prompt_text)

    bboxes = result.get("bboxes", []) or []
    labels = result.get("labels", []) or []

    article_re = [re.compile(p, re.IGNORECASE) for p in cfg.article_keep_patterns]
    wm_re = [re.compile(p, re.IGNORECASE) for p in cfg.watermark_text_patterns]

    img_area = image_pil.width * image_pil.height
    out: list[dict] = []
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(image_pil.width, x2)
        y2 = min(image_pil.height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        # Guard: whole-image bboxes are always false-positives. Florence sometimes
        # boxes a repeating background pattern as one giant region; inpainting it
        # would erase the entire product (see 3_trim.webp regression).
        bbox_area = (x2 - x1) * (y2 - y1)
        if bbox_area / img_area >= 0.85:
            out.append({
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "ocr_text": "[whole-image bbox — skipped]",
                "decision": "keep_unknown",
            })
            continue
        crop = image_pil.crop((x1, y1, x2, y2))
        ocr_text = ""
        try:
            ocr_res = _florence_run(mdl, crop, "<OCR>")
            if isinstance(ocr_res, str):
                ocr_text = ocr_res
        except Exception as e:  # noqa: BLE001
            logger.debug(f"OCR on bbox failed: {e}")

        ocr_text = ocr_text.strip()

        if any(p.search(ocr_text) for p in article_re):
            decision = "keep_article"
        elif any(p.search(ocr_text) for p in wm_re):
            decision = "remove"
        else:
            # No clear OCR match: decide by label.
            # Trust Florence if label was "phone" / "url" / "watermark", else flag unknown.
            label_lower = label.lower() if isinstance(label, str) else ""
            if any(k in label_lower for k in ("watermark", "phone", "url", "logo", "stamp")):
                decision = "remove"
            else:
                decision = "keep_unknown"

        out.append({
            "bbox": [x1, y1, x2, y2],
            "label": label,
            "ocr_text": ocr_text,
            "decision": decision,
        })
    return out


def build_watermark_mask(regions: list[dict], shape_hw: tuple[int, int], dilate_px: int = 6) -> np.ndarray:
    """Union of all 'remove' bboxes, dilated. Returns H×W uint8 (0/255)."""
    H, W = shape_hw
    mask = np.zeros((H, W), dtype=np.uint8)
    for r in regions:
        if r["decision"] != "remove":
            continue
        x1, y1, x2, y2 = r["bbox"]
        mask[y1:y2, x1:x2] = 255
    if dilate_px > 0 and mask.any():
        k = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
        mask = cv2.dilate(mask, k, iterations=1)
    return mask


def inpaint_watermark(img: np.ndarray, mask: np.ndarray, mdl: Models) -> np.ndarray:
    """Apply LaMa to fill the masked region."""
    pil_img = Image.fromarray(img)
    pil_mask = Image.fromarray(mask, mode="L")
    result = mdl.lama(pil_img, pil_mask)
    if isinstance(result, Image.Image):
        return np.array(result.convert("RGB"))
    return np.array(result)


# ──────────────────────────────────────────────────────────────────────────
# Stage 3: upscale (DAT2 or skip)
# ──────────────────────────────────────────────────────────────────────────

def maybe_upscale(img: np.ndarray, mdl: Models, cfg: Config) -> tuple[np.ndarray, dict]:
    """
    Upscale until long side ≥ upscale_target_long_side_px.
    Skip if long side already ≥ upscale_skip_long_side_px.
    DAT2 is x4; we apply it once if needed, then Lanczos-downscale to target.
    """
    H, W = img.shape[:2]
    long_side = max(H, W)
    info = {"input_long_side": long_side, "applied": False, "factor": 1.0, "fallback": False}

    if long_side >= cfg.upscale_skip_long_side_px:
        return img, info
    if long_side >= cfg.upscale_target_long_side_px:
        return img, info

    if mdl.dat2 is None:
        # Bilinear fallback so the rest of the pipeline still produces output
        target_long = cfg.upscale_target_long_side_px
        scale = target_long / long_side
        new_W, new_H = int(round(W * scale)), int(round(H * scale))
        up = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_LANCZOS4)
        info.update(applied=True, factor=scale, fallback=True)
        return up, info

    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor = tensor.to(mdl.device)
    with torch.inference_mode():
        out = mdl.dat2(tensor)
    out = out.clamp(0, 1)[0].permute(1, 2, 0).cpu().float().numpy()
    up = (out * 255.0 + 0.5).astype(np.uint8)
    factor = up.shape[0] / H  # should be ≈4

    # Downscale if we overshot the target
    new_long = max(up.shape[:2])
    if new_long > cfg.upscale_target_long_side_px:
        scale = cfg.upscale_target_long_side_px / new_long
        new_W = int(round(up.shape[1] * scale))
        new_H = int(round(up.shape[0] * scale))
        up = cv2.resize(up, (new_W, new_H), interpolation=cv2.INTER_LANCZOS4)

    info.update(applied=True, factor=round(factor, 2))
    return up, info


def upscale_batch(imgs: list[np.ndarray], mdl: Models, cfg: Config) -> list[tuple[np.ndarray, dict]]:
    """
    Batched Nomos x4 upscaler.

    Strategy:
      1. Split inputs into 'needs_upscale' (long_side < skip_threshold) and 'skip'.
      2. For 'needs_upscale': pad all to the max H/W in the group, run Nomos once,
         then crop each output back to its true size (×4).
      3. Skipped inputs pass through unchanged.

    All inputs must be uint8 RGB ndarrays. Order of returned list matches input.

    Padding wastes some GPU on smaller images in mixed-size batches. Callers
    should pre-sort by size to minimise waste when possible.
    """
    results: list[Optional[tuple[np.ndarray, dict]]] = [None] * len(imgs)
    needs_idx: list[int] = []
    for i, img in enumerate(imgs):
        H, W = img.shape[:2]
        long_side = max(H, W)
        info_skip = {"input_long_side": long_side, "applied": False, "factor": 1.0, "fallback": False}
        if long_side >= cfg.upscale_skip_long_side_px:
            results[i] = (img, info_skip)
        elif long_side >= cfg.upscale_target_long_side_px:
            results[i] = (img, info_skip)
        else:
            needs_idx.append(i)

    if not needs_idx:
        return [r for r in results if r is not None]  # type: ignore[misc]

    # No Nomos weights → bilinear fallback per item
    if mdl.dat2 is None:
        for i in needs_idx:
            H, W = imgs[i].shape[:2]
            long_side = max(H, W)
            target_long = cfg.upscale_target_long_side_px
            scale = target_long / long_side
            new_W, new_H = int(round(W * scale)), int(round(H * scale))
            up = cv2.resize(imgs[i], (new_W, new_H), interpolation=cv2.INTER_LANCZOS4)
            results[i] = (up, {
                "input_long_side": long_side, "applied": True,
                "factor": scale, "fallback": True,
            })
        return [r for r in results if r is not None]  # type: ignore[misc]

    # Real batched Nomos run
    sub_imgs = [imgs[i] for i in needs_idx]
    max_h = max(im.shape[0] for im in sub_imgs)
    max_w = max(im.shape[1] for im in sub_imgs)
    batch = torch.zeros(
        (len(sub_imgs), 3, max_h, max_w), dtype=torch.float32, device=mdl.device
    )
    for k, im in enumerate(sub_imgs):
        h, w = im.shape[:2]
        t = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
        batch[k, :, :h, :w] = t.to(mdl.device)

    with torch.inference_mode():
        out = mdl.dat2(batch)  # (N, 3, max_h*4, max_w*4)

    out = out.clamp(0, 1).cpu().float().numpy()
    for k, idx in enumerate(needs_idx):
        h, w = sub_imgs[k].shape[:2]
        up = (out[k, :, : h * 4, : w * 4].transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
        new_long = max(up.shape[:2])
        if new_long > cfg.upscale_target_long_side_px:
            scale = cfg.upscale_target_long_side_px / new_long
            up = cv2.resize(
                up,
                (int(round(up.shape[1] * scale)), int(round(up.shape[0] * scale))),
                interpolation=cv2.INTER_LANCZOS4,
            )
        results[idx] = (up, {
            "input_long_side": max(h, w), "applied": True, "factor": 4.0, "fallback": False,
        })

    return [r for r in results if r is not None]  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────────
# Stage 4: background removal (BiRefNet) + refinement
# ──────────────────────────────────────────────────────────────────────────

def predict_alpha(img: np.ndarray, mdl: Models, cfg: Config) -> np.ndarray:
    """Run BiRefNet, return float32 alpha in [0,1] at input resolution."""
    H, W = img.shape[:2]
    pil = Image.fromarray(img)
    inp = mdl.birefnet_transform(pil).unsqueeze(0).to(mdl.birefnet_device)
    with torch.inference_mode():
        preds = mdl.birefnet(inp)
        # BiRefNet returns either a list or a tuple-like; take the last element and sigmoid.
        if isinstance(preds, (list, tuple)):
            pred = preds[-1]
        else:
            pred = preds
        if hasattr(pred, "logits"):
            pred = pred.logits
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
        alpha_low = torch.sigmoid(pred)[0, 0].float().cpu().numpy()
    alpha = cv2.resize(alpha_low, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return np.clip(alpha, 0.0, 1.0)


def guided_filter_refine(rgb: np.ndarray, alpha: np.ndarray, radius: int = 4, eps: float = 1e-3) -> np.ndarray:
    """Edge-aware refinement of alpha using luminance as guide. Returns float32 [0,1]."""
    guide = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    p = alpha.astype(np.float32)
    ksize = (2 * radius + 1, 2 * radius + 1)

    mean_I = cv2.boxFilter(guide, -1, ksize)
    mean_p = cv2.boxFilter(p, -1, ksize)
    corr_Ip = cv2.boxFilter(guide * p, -1, ksize)
    cov_Ip = corr_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(guide * guide, -1, ksize)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, ksize)
    mean_b = cv2.boxFilter(b, -1, ksize)

    refined = mean_a * guide + mean_b
    return np.clip(refined, 0.0, 1.0)


def composite_on_white(rgb: np.ndarray, alpha: np.ndarray, bg=(255, 255, 255)) -> np.ndarray:
    """Soft-alpha composite of rgb foreground onto solid bg."""
    a = alpha[..., None].astype(np.float32)
    bg_arr = np.array(bg, dtype=np.float32).reshape(1, 1, 3)
    out = rgb.astype(np.float32) * a + bg_arr * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────
# Stage 5: framing — crop to content, pad to square, resize
# ──────────────────────────────────────────────────────────────────────────

def content_bbox(img: np.ndarray, alpha: Optional[np.ndarray] = None,
                 white_threshold: int = 245) -> tuple[int, int, int, int]:
    """Bbox (x1, y1, x2, y2) of non-background content."""
    if alpha is not None:
        mask = alpha > 0.05
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray < white_threshold
    if not mask.any():
        H, W = img.shape[:2]
        return 0, 0, W, H
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def frame_to_square(img: np.ndarray, target_size: int, fill_ratio: float,
                    fill_color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Scale longer side to fill_ratio*target_size, pad to target_size square, centered."""
    H, W = img.shape[:2]
    longer = max(H, W)
    scale = (target_size * fill_ratio) / longer
    new_W = max(1, int(round(W * scale)))
    new_H = max(1, int(round(H * scale)))
    resized = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.full((target_size, target_size, 3), fill_color, dtype=np.uint8)
    x_off = (target_size - new_W) // 2
    y_off = (target_size - new_H) // 2
    canvas[y_off:y_off + new_H, x_off:x_off + new_W] = resized
    return canvas


# ──────────────────────────────────────────────────────────────────────────
# Pipeline orchestration
# ──────────────────────────────────────────────────────────────────────────

def _timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    res = fn(*args, **kwargs)
    return res, (time.perf_counter() - t0) * 1000.0


def overlay_bboxes(img: np.ndarray, regions: list[dict]) -> np.ndarray:
    """Draw watermark bboxes on a copy. Color by decision."""
    out = img.copy()
    colors = {
        "remove": (255, 0, 0),
        "keep_article": (0, 200, 0),
        "keep_unknown": (255, 200, 0),
    }
    for r in regions:
        x1, y1, x2, y2 = r["bbox"]
        color = colors.get(r["decision"], (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{r['decision']}: {r['ocr_text'][:30]}"
        cv2.putText(out, label, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return out


def process_one_resize(img_path: Path, output_dir: Path, mdl: Models, cfg: Config) -> dict:
    """Resize-only pipeline (mode B for 13M bulk):
    trim white margins → upscale if <1000 → frame 85% → resize 1000 → WebP.
    No watermark removal, no bg removal, no drawing classifier.
    """
    name = img_path.stem
    debug_dir = output_dir / "debug" / name
    debug_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, Any] = {
        "name": img_path.name,
        "mode": "resize",
        "stage_times_ms": {},
        "review_reason": None,
    }
    total_t0 = time.perf_counter()

    img, t = _timed(load_image, img_path)
    metrics["stage_times_ms"]["load"] = round(t, 1)
    metrics["input_size"] = [int(img.shape[1]), int(img.shape[0])]

    x1, y1, x2, y2 = content_bbox(img, alpha=None)
    cropped = img[y1:y2, x1:x2]
    metrics["content_bbox"] = [x1, y1, x2, y2]

    upscaled, t = _timed(maybe_upscale, cropped, mdl, cfg)
    upscaled, up_info = upscaled
    metrics["stage_times_ms"]["upscale"] = round(t, 1)
    metrics["upscale"] = up_info

    final, t = _timed(frame_to_square, upscaled, cfg.output_size,
                      cfg.product_fill_ratio, cfg.padding_color)
    metrics["stage_times_ms"]["frame"] = round(t, 1)
    metrics["output_size"] = [int(final.shape[1]), int(final.shape[0])]

    out_path = output_dir / f"{name}.webp"
    save_image(final, out_path, fmt="WEBP", quality=cfg.webp_quality)

    metrics["stage_times_ms"]["total"] = round((time.perf_counter() - total_t0) * 1000, 1)
    metrics["output_path"] = str(out_path.relative_to(output_dir.parent))
    (debug_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def process_batch_resize(
    items: list[tuple[Path, str]],
    output_dir: Path,
    mdl: Models,
    cfg: Config,
) -> list[dict]:
    """Batched resize-only pipeline.

    Steps for the whole batch:
      1. Load + EXIF strip (CPU, per-item)
      2. content_bbox + crop (CPU, per-item)
      3. upscale_batch (one GPU call for the whole batch)
      4. frame_to_square + WebP encode (CPU, per-item)

    Returns list of metrics dicts in the same order as `items`.
    """
    if not items:
        return []
    batch_t0 = time.perf_counter()

    # 1+2. Load and trim each image
    loaded: list[dict] = []
    for img_path, sha in items:
        per_t0 = time.perf_counter()
        img = load_image(img_path)
        x1, y1, x2, y2 = content_bbox(img, alpha=None)
        cropped = img[y1:y2, x1:x2]
        loaded.append({
            "path": img_path,
            "sha": sha,
            "input_size": [int(img.shape[1]), int(img.shape[0])],
            "content_bbox": [x1, y1, x2, y2],
            "cropped": cropped,
            "load_trim_ms": (time.perf_counter() - per_t0) * 1000.0,
        })

    # 3. Batched Nomos upscale (one GPU call for the whole batch)
    upscale_t0 = time.perf_counter()
    cropped_list = [item["cropped"] for item in loaded]
    upscaled_with_info = upscale_batch(cropped_list, mdl, cfg)
    upscale_total_ms = (time.perf_counter() - upscale_t0) * 1000.0

    # 4. Frame + save per item
    results: list[dict] = []
    for item, (upscaled, up_info) in zip(loaded, upscaled_with_info):
        per_t0 = time.perf_counter()
        final = frame_to_square(
            upscaled, cfg.output_size, cfg.product_fill_ratio, cfg.padding_color
        )
        out_path = output_dir / f"{item['path'].stem}.webp"
        save_image(final, out_path, fmt="WEBP", quality=cfg.webp_quality)
        frame_ms = (time.perf_counter() - per_t0) * 1000.0

        # Per-image metrics. upscale_share is upscale_total / batch_size — a fair
        # attribution of the shared GPU call across the items in this batch.
        upscale_share_ms = upscale_total_ms / max(1, len(loaded))
        total_ms = item["load_trim_ms"] + upscale_share_ms + frame_ms

        metrics = {
            "name": item["path"].name,
            "mode": "resize",
            "input_size": item["input_size"],
            "output_size": [int(final.shape[1]), int(final.shape[0])],
            "content_bbox": item["content_bbox"],
            "upscale": up_info,
            "stage_times_ms": {
                "load_trim": round(item["load_trim_ms"], 1),
                "upscale_share_of_batch": round(upscale_share_ms, 1),
                "frame": round(frame_ms, 1),
                "total": round(total_ms, 1),
            },
            "batch_size": len(loaded),
            "review_reason": None,
            "input_hash": item["sha"],
            "output_path": str(out_path.relative_to(output_dir.parent)),
        }
        # Per-image metrics on disk
        debug_dir = output_dir / "debug" / item["path"].stem
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False)
        )
        results.append(metrics)

    logger.info(
        f"batch n={len(loaded)} upscale_ms={upscale_total_ms:.0f} "
        f"wall_ms={(time.perf_counter()-batch_t0)*1000:.0f}"
    )
    return results


def process_one(img_path: Path, output_dir: Path, mdl: Models, cfg: Config) -> dict:
    """Dispatch to the right pipeline based on cfg.mode. (Single-image path.)"""
    if cfg.mode == "resize":
        return process_one_resize(img_path, output_dir, mdl, cfg)
    return process_one_full(img_path, output_dir, mdl, cfg)


def process_one_full(img_path: Path, output_dir: Path, mdl: Models, cfg: Config) -> dict:
    """Full AI pipeline: watermark detect+inpaint, bg removal, alpha refine."""
    name = img_path.stem
    debug_dir = output_dir / "debug" / name
    debug_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, Any] = {
        "name": img_path.name,
        "mode": "full",
        "stage_times_ms": {},
        "review_reason": None,
    }
    total_t0 = time.perf_counter()

    # 1. Load
    img, t = _timed(load_image, img_path)
    metrics["stage_times_ms"]["load"] = round(t, 1)
    metrics["input_size"] = [int(img.shape[1]), int(img.shape[0])]
    save_image(img, debug_dir / "01_loaded.png")

    # 2. Content type
    (content_type, ctype_metrics), t = _timed(detect_content_type, img, cfg)
    metrics["stage_times_ms"]["classify"] = round(t, 1)
    metrics["content_type"] = content_type
    metrics["content_type_signals"] = ctype_metrics

    current = img

    if content_type == "photo":
        # 3. Watermark detection
        regions, t = _timed(detect_watermarks, Image.fromarray(current), mdl, cfg)
        metrics["stage_times_ms"]["watermark_detect"] = round(t, 1)
        metrics["watermark_regions"] = regions
        save_image(overlay_bboxes(current, regions),
                   debug_dir / "02_watermark_bboxes.png")

        to_remove = [r for r in regions if r["decision"] == "remove"]
        unknown = [r for r in regions if r["decision"] == "keep_unknown"]
        metrics["watermark_remove_count"] = len(to_remove)
        metrics["watermark_keep_unknown_count"] = len(unknown)

        if to_remove:
            mask, t = _timed(build_watermark_mask, to_remove, current.shape[:2])
            metrics["stage_times_ms"]["watermark_mask"] = round(t, 1)
            save_image(mask, debug_dir / "03_watermark_mask.png")

            mask_coverage = float((mask > 0).mean())
            metrics["watermark_mask_coverage"] = round(mask_coverage, 4)

            # Hard safety bar: if the watermark mask covers more than half of the
            # image, the detector is wrong (no legitimate watermark covers half
            # a product photo). Skip inpaint and flag for review.
            if mask_coverage > 0.5:
                metrics["review_reason"] = "watermark_detection_too_broad"
                logger.warning(
                    f"{img_path.name}: watermark mask covers {mask_coverage:.0%} — skipping inpaint"
                )
            else:
                if mask_coverage > cfg.watermark_coverage_review_pct:
                    metrics["review_reason"] = "watermark_risk"
                inpainted, t = _timed(inpaint_watermark, current, mask, mdl)
                metrics["stage_times_ms"]["inpaint"] = round(t, 1)
                save_image(inpainted, debug_dir / "04_inpainted.png")
                current = inpainted
        else:
            metrics["watermark_mask_coverage"] = 0.0

        # 4. Upscale
        upscaled, t = _timed(maybe_upscale, current, mdl, cfg)
        upscaled, up_info = upscaled
        metrics["stage_times_ms"]["upscale"] = round(t, 1)
        metrics["upscale"] = up_info
        save_image(upscaled, debug_dir / "05_upscaled.png")
        current = upscaled

        # 5. Background removal
        alpha, t = _timed(predict_alpha, current, mdl, cfg)
        metrics["stage_times_ms"]["bg_removal"] = round(t, 1)
        save_image((alpha * 255).astype(np.uint8), debug_dir / "06_alpha_raw.png")

        # 6. Alpha refine
        alpha_ref, t = _timed(guided_filter_refine, current, alpha)
        metrics["stage_times_ms"]["alpha_refine"] = round(t, 1)
        save_image((alpha_ref * 255).astype(np.uint8), debug_dir / "07_alpha_refined.png")

        # Confidence proxy: mean alpha in [0.05, 0.95] band (uncertainty band)
        band = (alpha_ref > 0.05) & (alpha_ref < 0.95)
        band_ratio = float(band.mean()) if alpha_ref.size else 0.0
        metrics["alpha_uncertain_band_ratio"] = round(band_ratio, 4)
        if band_ratio > 0.35:
            metrics["review_reason"] = metrics.get("review_reason") or "low_confidence"

        # 7. Composite
        composed, t = _timed(composite_on_white, current, alpha_ref, cfg.padding_color)
        metrics["stage_times_ms"]["composite"] = round(t, 1)
        save_image(composed, debug_dir / "08_composed.png")

        # 8. Crop to product (use refined alpha so we don't clip to whole canvas)
        x1, y1, x2, y2 = content_bbox(composed, alpha=alpha_ref)
        cropped = composed[y1:y2, x1:x2]

    else:
        # DRAWING branch: trim white margins, no heavy processing.
        x1, y1, x2, y2 = content_bbox(current, alpha=None)
        cropped = current[y1:y2, x1:x2]
        save_image(cropped, debug_dir / "02_drawing_trimmed.png")

    # 9. Frame to square
    final, t = _timed(frame_to_square, cropped, cfg.output_size,
                      cfg.product_fill_ratio, cfg.padding_color)
    metrics["stage_times_ms"]["frame"] = round(t, 1)
    metrics["output_size"] = [int(final.shape[1]), int(final.shape[0])]

    # 10. Save final
    out_path = output_dir / f"{name}.webp"
    save_image(final, out_path, fmt="WEBP", quality=cfg.webp_quality)
    save_image(final, debug_dir / "11_final.png")  # PNG copy for visual debug

    metrics["stage_times_ms"]["total"] = round((time.perf_counter() - total_t0) * 1000, 1)
    metrics["output_path"] = str(out_path.relative_to(output_dir.parent))

    # Persist per-image metrics
    (debug_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def list_inputs(input_dir: Path) -> list[Path]:
    return sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file()])


def file_sha256(path: Path) -> str:
    """SHA256 of file contents (streaming, 64KB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def is_already_done(img_path: Path, output_dir: Path, mode: str) -> tuple[bool, str, Optional[dict]]:
    """Idempotency check via SHA256 cache + output-file existence.

    Cache key includes mode so resize-mode and full-mode outputs don't collide.
    """
    sha = file_sha256(img_path)
    cache_path = output_dir / ".cache" / f"{sha}_{mode}.json"
    out_path = output_dir / f"{img_path.stem}.webp"
    if cache_path.exists() and out_path.exists() and out_path.stat().st_size > 0:
        try:
            cached = json.loads(cache_path.read_text())
            if cached.get("input_hash") == sha and cached.get("mode") == mode:
                return True, sha, cached
        except Exception:  # noqa: BLE001
            pass
    return False, sha, None


def write_cache(output_dir: Path, sha: str, mode: str, metrics: dict) -> None:
    cache_dir = output_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{sha}_{mode}.json").write_text(
        json.dumps({"input_hash": sha, "mode": mode, **metrics}, ensure_ascii=False)
    )


def main() -> int:
    p = argparse.ArgumentParser(description="foto-klass Stage 1 prototype")
    p.add_argument("--input", required=True, type=Path, help="Folder with source images")
    p.add_argument("--output", default=Path("output"), type=Path, help="Output folder")
    p.add_argument("--config", default=Path("config.yaml"), type=Path, help="Config YAML (falls back to defaults)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--limit", type=int, default=None, help="Process only first N images")
    p.add_argument("--force", action="store_true",
                   help="Ignore SHA256 cache; reprocess everything")
    p.add_argument("--mode", choices=["full", "resize"], default=None,
                   help="Pipeline mode (overrides config). 'full' = AI cleanup, "
                        "'resize' = bulk resize+SR only.")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Photos per GPU batch in --mode resize (default 16). "
                        "Set to 1 to disable batching. Ignored in --mode full.")
    args = p.parse_args()

    cfg = Config.load(args.config)
    if args.mode is not None:
        cfg.mode = args.mode
    device = select_device(args.device)
    logger.info(f"Mode: {cfg.mode} | Device: {device}")
    logger.info(f"Output: {args.output}")

    inputs = list_inputs(args.input)
    if args.limit:
        inputs = inputs[:args.limit]
    if not inputs:
        logger.error(f"No images found in {args.input}")
        return 1
    logger.info(f"Found {len(inputs)} images")

    args.output.mkdir(parents=True, exist_ok=True)

    # Pre-flight: split inputs into (cached, todo) so we only load models if needed.
    cached_items: list[dict] = []
    todo: list[tuple[Path, str]] = []
    for img_path in inputs:
        if args.force:
            todo.append((img_path, file_sha256(img_path)))
            continue
        skip, sha, cached_metrics = is_already_done(img_path, args.output, cfg.mode)
        if skip:
            logger.info(f"⏭  {img_path.name} cached (sha256={sha[:8]})")
            cached_items.append({**(cached_metrics or {}), "name": img_path.name, "cached": True})
        else:
            todo.append((img_path, sha))

    summary: list[dict] = list(cached_items)
    if not todo:
        logger.info("Nothing to do — all inputs already processed (use --force to override)")
    else:
        logger.info(
            f"To process: {len(todo)} / {len(inputs)} (cached: {len(cached_items)}) "
            f"mode={cfg.mode} batch_size={args.batch_size}"
        )
        # MPS backend has poor scaling for batched conv operations — batching is
        # actually slower than sequential. CUDA shows the expected 3-5x speedup.
        if device.type == "mps" and args.batch_size > 1 and cfg.mode == "resize":
            logger.warning(
                "Mac MPS detected with --batch-size > 1. On Apple Silicon batching "
                "is currently slower than sequential due to MPS backend limitations. "
                "Consider --batch-size 1 on Mac. On RTX 4090 (CUDA), batch=16 is optimal."
            )
        mdl = load_models(cfg, device)

        if cfg.mode == "resize" and args.batch_size > 1:
            # Sort by image long-side to minimise zero-padding waste inside the
            # batch — Nomos pads all items to the largest H/W in the batch and
            # then crops back, so size-homogeneous batches run faster.
            sized_todo = []
            for img_path, sha in todo:
                try:
                    with Image.open(img_path) as im:
                        long_side = max(im.size)
                except Exception:  # noqa: BLE001
                    long_side = 0
                sized_todo.append((long_side, img_path, sha))
            sized_todo.sort(key=lambda t: t[0])
            sorted_todo = [(p, s) for (_, p, s) in sized_todo]

            # Batched path — one GPU call per `batch_size` photos
            for chunk_start in tqdm(
                range(0, len(sorted_todo), args.batch_size), desc="Batches"
            ):
                chunk = sorted_todo[chunk_start: chunk_start + args.batch_size]
                try:
                    batch_results = process_batch_resize(chunk, args.output, mdl, cfg)
                    for m in batch_results:
                        write_cache(args.output, m["input_hash"], cfg.mode, m)
                        summary.append(m)
                except Exception as e:  # noqa: BLE001
                    logger.exception(f"✗ batch starting at {chunk[0][0].name} failed: {e}")
                    for img_path, _ in chunk:
                        summary.append({"name": img_path.name, "error": str(e)})

                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
        else:
            # Single-image path — used for --mode full or --batch-size=1
            for img_path, sha in tqdm(todo, desc="Processing"):
                try:
                    m = process_one(img_path, args.output, mdl, cfg)
                    m["input_hash"] = sha
                    write_cache(args.output, sha, cfg.mode, m)
                    summary.append(m)
                    logger.info(
                        f"✓ {img_path.name} → {m.get('content_type')}, "
                        f"{m['stage_times_ms']['total']}ms, review={m.get('review_reason')}"
                    )
                except Exception as e:  # noqa: BLE001
                    logger.exception(f"✗ {img_path.name} failed: {e}")
                    summary.append({"name": img_path.name, "error": str(e)})

            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    # Aggregate
    ok = [s for s in summary if "error" not in s]
    review = [s for s in ok if s.get("review_reason")]
    processed = [s for s in ok if not s.get("cached")]
    cached = [s for s in ok if s.get("cached")]
    by_type = {"photo": 0, "drawing": 0}
    for s in ok:
        by_type[s.get("content_type", "photo")] = by_type.get(s.get("content_type", "photo"), 0) + 1
    avg_total = (sum(s["stage_times_ms"]["total"] for s in processed) / len(processed)) if processed else 0.0

    summary_doc = {
        "total": len(summary),
        "ok": len(ok),
        "errors": len(summary) - len(ok),
        "processed": len(processed),
        "cached": len(cached),
        "review_count": len(review),
        "by_content_type": by_type,
        "avg_total_ms": round(avg_total, 1),
        "device": str(device),
        "items": summary,
    }
    (args.output / "summary.json").write_text(json.dumps(summary_doc, indent=2, ensure_ascii=False))
    logger.info(
        f"Done. processed={summary_doc['processed']} cached={summary_doc['cached']} "
        f"errors={summary_doc['errors']} review={summary_doc['review_count']} "
        f"avg {summary_doc['avg_total_ms']}ms/photo"
    )
    return 0 if summary_doc["errors"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
