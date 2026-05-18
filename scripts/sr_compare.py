"""Compare multiple SR models on a single image. Outputs to output/."""
import sys
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image, ImageOps
from spandrel import ModelLoader

SRC = Path("sample/3_orig.webp")
OUT_DIR = Path("output")
TARGET = 1000
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

MODELS = {
    "dat2":     "weights/4xRealWebPhoto_v4_dat2.safetensors",
    "realesr":  "weights/RealESRGAN_x4.pth",
    "nomos":    "weights/4xNomosWebPhoto_RealPLKSR.safetensors",
}


def pad_square(img: Image.Image, color=(255, 255, 255)) -> Image.Image:
    W, H = img.size
    side = max(W, H)
    canvas = Image.new("RGB", (side, side), color)
    canvas.paste(img, ((side - W) // 2, (side - H) // 2))
    return canvas


def upscale(img: Image.Image, weights: str, device: torch.device) -> Image.Image:
    desc = ModelLoader().load_from_file(weights).to(device).eval()
    arr = np.array(img)
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t = t.to(device)
    with torch.inference_mode():
        out = desc(t)
    out = out.clamp(0, 1)[0].permute(1, 2, 0).cpu().float().numpy()
    return Image.fromarray((out * 255 + 0.5).astype(np.uint8))


def main():
    src = Image.open(SRC)
    src = ImageOps.exif_transpose(src).convert("RGB")
    src = pad_square(src)
    print(f"Input (padded): {src.size}, device={DEVICE}")

    for tag, weights in MODELS.items():
        out_path = OUT_DIR / f"3_orig_sr_{tag}.webp"
        try:
            t0 = time.perf_counter()
            up = upscale(src, weights, DEVICE)
            t1 = time.perf_counter()
            print(f"  {tag}: {src.size} -> {up.size} in {(t1-t0)*1000:.0f}ms")
            # Downscale to TARGET if overshoot
            if max(up.size) > TARGET:
                up = up.resize((TARGET, TARGET), Image.LANCZOS)
            up.save(out_path, "WEBP", quality=90, method=6)
            sz = out_path.stat().st_size
            print(f"  saved {out_path} ({sz/1024:.1f} KB)")
        except Exception as e:
            print(f"  {tag} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
