"""Runs on one VPS. End-to-end pipeline:

  1. Lists all 17M keys in R2 under shards 00-ff/unpacked/
  2. For each key (parallel workers): GET → decode WebP → content_bbox →
     frame_to_square(300, fill=0.85) → encode WebP q75
  3. Streams a single GNU tar where each entry is at <archive_id>/<filename>.webp
  4. Splits the tar stream into 100 MB chunks and uploads each as a
     multipart-upload part to Hetzner Object Storage in real time.
  5. Completes the multipart upload at the end → one final tar object.

Output:
  s3://allzap-tecdoc-photos/thumbs_300/listings_300x300.tar  (~72 GB)

Environment expected (from .env on VPS):
  R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_ENDPOINT / R2_BUCKET
  HETZNER_S3_ACCESS_KEY / HETZNER_S3_SECRET_KEY / HETZNER_S3_ENDPOINT
  HETZNER_S3_REGION / HETZNER_S3_BUCKET

Usage:
  python pack_thumbs_to_tar.py                     # full run
  python pack_thumbs_to_tar.py --max-keys 10000    # smoke test
  python pack_thumbs_to_tar.py --threads 64        # tune parallelism
  python pack_thumbs_to_tar.py --resume            # resume after crash
"""
from __future__ import annotations

import argparse
import io
import json
import os
import struct
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue, Empty

import boto3
import cv2
import numpy as np
from botocore.config import Config as BotoConfig
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
def _state_paths(worker_id: int, total_workers: int) -> tuple[Path, Path]:
    if total_workers > 1:
        return (Path(f"/tmp/pack_thumbs_state_w{worker_id}of{total_workers}.json"),
                Path(f"/tmp/pack_thumbs_w{worker_id}of{total_workers}.log"))
    return Path("/tmp/pack_thumbs_state.json"), Path("/tmp/pack_thumbs.log")

# Module-level defaults — overridden in main() after parsing args.
STATE_FILE = Path("/tmp/pack_thumbs_state.json")
LOG_FILE = Path("/tmp/pack_thumbs.log")

PART_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB per multipart part
DEFAULT_OUTPUT_KEY = "thumbs_300/listings_300x300.tar"


def load_env():
    # Look for .env in script dir, parent, and grandparent (handles both repo
    # checkout and standalone deploy on VPS at /opt/fk-unpack/).
    here = Path(__file__).resolve()
    for candidate in (here.parent / ".env", ROOT / ".env", here.parent.parent / ".env"):
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
            return
    raise RuntimeError(f"No .env found in any of: {[str(here.parent/'.env'), str(ROOT/'.env')]}")


def parse_endpoint(raw: str) -> str:
    p = raw.split("://", 1)
    return p[0] + "://" + p[1].split("/", 1)[0]


def r2_client():
    return boto3.client(
        "s3",
        endpoint_url=parse_endpoint(os.environ["R2_ENDPOINT"]),
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=BotoConfig(retries={"max_attempts": 10, "mode": "standard"},
                          max_pool_connections=256),
    )


def hetzner_client():
    ep = os.environ["HETZNER_S3_ENDPOINT"]
    if not ep.startswith("http"):
        ep = "https://" + ep
    return boto3.client(
        "s3",
        endpoint_url=ep,
        aws_access_key_id=os.environ["HETZNER_S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["HETZNER_S3_SECRET_KEY"],
        region_name=os.environ.get("HETZNER_S3_REGION", "fsn1"),
        config=BotoConfig(retries={"max_attempts": 10, "mode": "standard"},
                          max_pool_connections=8),
    )


def content_bbox(img: np.ndarray, white_threshold: int = 230):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray < white_threshold
    if not mask.any():
        H, W = img.shape[:2]; return 0, 0, W, H
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def frame_to_square(img: np.ndarray, target: int, fill_ratio: float) -> np.ndarray:
    H, W = img.shape[:2]
    longer = max(H, W)
    scale = (target * fill_ratio) / longer
    new_W = max(1, int(round(W * scale)))
    new_H = max(1, int(round(H * scale)))
    resized = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.full((target, target, 3), (255, 255, 255), dtype=np.uint8)
    x = (target - new_W) // 2
    y = (target - new_H) // 2
    canvas[y:y + new_H, x:x + new_W] = resized
    return canvas


def make_thumb(raw_bytes: bytes) -> bytes:
    pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    img = np.asarray(pil)
    x1, y1, x2, y2 = content_bbox(img, white_threshold=230)
    cropped = img[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else img
    squared = frame_to_square(cropped, target=300, fill_ratio=0.85)
    buf = io.BytesIO()
    Image.fromarray(squared).save(buf, format="WEBP", quality=75, method=6)
    return buf.getvalue()


def key_to_tar_path(key: str) -> str:
    """R2 key '<hh>/unpacked/<archive_id>/<filename>' → '<archive_id>/<filename>'"""
    parts = key.split("/")
    return f"{parts[2]}/{parts[3]}"


def iter_r2_keys(r2, bucket: str, resume_after: str | None = None,
                 worker_id: int = 0, total_workers: int = 1,
                 start_shard: str | None = None, end_shard: str | None = None):
    """Yield keys under {hh}/unpacked/ filtered by archive_id %% total_workers == worker_id.

    Each key has form '<hh>/unpacked/<archive_id>/<filename>'.
    start_shard/end_shard limit iteration to a hex range, e.g. start='c0' end='ff'.
    """
    start_int = int(start_shard, 16) if start_shard else 0
    end_int = int(end_shard, 16) if end_shard else 255
    for i in range(start_int, end_int + 1):
        shard = f"{i:02x}"
        if resume_after and resume_after < f"{shard}/":
            continue
        token = None
        while True:
            kw = {"Bucket": bucket, "Prefix": f"{shard}/unpacked/", "MaxKeys": 1000}
            if token: kw["ContinuationToken"] = token
            if resume_after and resume_after.startswith(f"{shard}/"):
                kw["StartAfter"] = resume_after
                resume_after = None  # only use once
            resp = r2.list_objects_v2(**kw)
            for o in resp.get("Contents") or []:
                key = o["Key"]
                if total_workers > 1:
                    try:
                        aid = int(key.split("/")[2])
                    except (IndexError, ValueError):
                        continue  # bad key, skip
                    if aid % total_workers != worker_id:
                        continue
                yield key
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")


def write_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def read_state() -> dict | None:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return None


def log(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG_FILE.open("a") as f:
        f.write(line + "\n")


class MultipartTarUploader:
    """Receives raw bytes via .write(), uploads as multipart parts of one S3
    object once PART_SIZE_BYTES is accumulated."""

    def __init__(self, hetzner, bucket: str, key: str, resume_state: dict | None = None):
        self.hetzner = hetzner
        self.bucket = bucket
        self.key = key
        self.buf = io.BytesIO()
        self.parts: list[dict] = []
        self.part_number = 1
        self.total_bytes = 0

        if resume_state and resume_state.get("upload_id"):
            self.upload_id = resume_state["upload_id"]
            self.parts = resume_state.get("parts", [])
            self.part_number = resume_state.get("next_part_number", 1)
            self.total_bytes = resume_state.get("total_bytes", 0)
            log(f"Resuming MPU {self.upload_id}, next part {self.part_number}")
        else:
            r = hetzner.create_multipart_upload(
                Bucket=bucket, Key=key, ContentType="application/x-tar")
            self.upload_id = r["UploadId"]
            log(f"Started MPU {self.upload_id} → s3://{bucket}/{key}")

    def write(self, data: bytes):
        self.buf.write(data)
        if self.buf.tell() >= PART_SIZE_BYTES:
            self._flush_part()

    def _flush_part(self, force: bool = False):
        size = self.buf.tell()
        if size == 0:
            return
        if size < PART_SIZE_BYTES and not force:
            return
        body = self.buf.getvalue()
        self.buf = io.BytesIO()

        attempt = 0
        while True:
            try:
                resp = self.hetzner.upload_part(
                    Bucket=self.bucket, Key=self.key, UploadId=self.upload_id,
                    PartNumber=self.part_number, Body=body)
                break
            except Exception as e:
                attempt += 1
                if attempt > 10:
                    raise
                log(f"upload_part {self.part_number} failed ({e}), retry {attempt}/10")
                time.sleep(2 ** min(attempt, 6))

        self.parts.append({"PartNumber": self.part_number, "ETag": resp["ETag"]})
        self.total_bytes += size
        log(f"part {self.part_number} uploaded ({size/1024**2:.1f} MB, total {self.total_bytes/1024**3:.2f} GB)")
        self.part_number += 1

    def complete(self):
        self._flush_part(force=True)
        log(f"Completing MPU with {len(self.parts)} parts...")
        self.hetzner.complete_multipart_upload(
            Bucket=self.bucket, Key=self.key, UploadId=self.upload_id,
            MultipartUpload={"Parts": self.parts})
        log(f"DONE: s3://{self.bucket}/{self.key}  total {self.total_bytes/1024**3:.2f} GB")

    def abort(self):
        try:
            self.hetzner.abort_multipart_upload(
                Bucket=self.bucket, Key=self.key, UploadId=self.upload_id)
            log(f"Aborted MPU {self.upload_id}")
        except Exception as e:
            log(f"Abort failed: {e}")

    def to_state(self) -> dict:
        return {
            "upload_id": self.upload_id,
            "parts": self.parts,
            "next_part_number": self.part_number,
            "total_bytes": self.total_bytes,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-keys", type=int, default=0, help="0 = process all")
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--prefetch", type=int, default=512, help="in-flight downloads")
    ap.add_argument("--resume", action="store_true", help="resume from STATE_FILE")
    ap.add_argument("--reset", action="store_true", help="abort any active MPU and start fresh")
    ap.add_argument("--worker-id", type=int, default=0, help="0..total-1 (shard by archive_id)")
    ap.add_argument("--total-workers", type=int, default=1, help="total parallel workers")
    ap.add_argument("--output-key", type=str, default=None,
                    help="override Hetzner output key (defaults to thumbs_300/listings_part_<w+1>of<t>.tar)")
    ap.add_argument("--start-shard", type=str, default=None,
                    help="start at shard hex e.g. 'c0' (limits R2 listing range)")
    ap.add_argument("--end-shard", type=str, default=None,
                    help="stop after shard hex e.g. 'bf' (inclusive)")
    args = ap.parse_args()

    # default per-worker output key + per-worker state/log paths
    global STATE_FILE, LOG_FILE
    STATE_FILE, LOG_FILE = _state_paths(args.worker_id, args.total_workers)
    if args.output_key:
        output_key = args.output_key
    elif args.total_workers > 1:
        output_key = f"thumbs_300/listings_part_{args.worker_id+1}of{args.total_workers}.tar"
    else:
        output_key = DEFAULT_OUTPUT_KEY

    load_env()
    r2 = r2_client()
    het = hetzner_client()
    out_bucket = os.environ["HETZNER_S3_BUCKET"]

    # state handling
    state = read_state() if args.resume else None
    if args.reset:
        if state and state.get("upload_id"):
            try:
                het.abort_multipart_upload(Bucket=out_bucket, Key=output_key,
                                           UploadId=state["upload_id"])
                log(f"Aborted old MPU {state['upload_id']}")
            except Exception as e:
                log(f"Abort failed: {e}")
        STATE_FILE.unlink(missing_ok=True)
        state = None

    uploader = MultipartTarUploader(het, out_bucket, output_key, resume_state=state)
    resume_after_key = state.get("last_key") if state else None

    # tar streaming setup
    class _PipeSink:
        def __init__(self, up): self.up = up
        def write(self, b):
            self.up.write(b)
            return len(b)

    tar = tarfile.open(fileobj=_PipeSink(uploader), mode="w|", format=tarfile.GNU_FORMAT)

    # producer-consumer
    processed = state.get("processed", 0) if state else 0
    ok = state.get("ok", 0) if state else 0
    err = state.get("err", 0) if state else 0
    last_key = state.get("last_key", "") if state else ""
    start = time.time()
    last_report = start

    in_flight = {}  # key → Future
    keys_iter = iter_r2_keys(r2, os.environ["R2_BUCKET"], resume_after=resume_after_key,
                             worker_id=args.worker_id, total_workers=args.total_workers,
                             start_shard=args.start_shard, end_shard=args.end_shard)
    log(f"START worker {args.worker_id}/{args.total_workers}  output→ s3://{out_bucket}/{output_key}")
    if args.max_keys:
        from itertools import islice
        keys_iter = islice(keys_iter, args.max_keys)

    def process(key):
        try:
            obj = r2.get_object(Bucket=os.environ["R2_BUCKET"], Key=key)
            raw = obj["Body"].read()
            return key, make_thumb(raw), None
        except Exception as e:
            return key, None, str(e)

    try:
        with ThreadPoolExecutor(max_workers=args.threads) as ex:
            # prime the pipeline
            pending: list = []
            for _ in range(args.prefetch):
                try:
                    pending.append(ex.submit(process, next(keys_iter)))
                except StopIteration:
                    break

            while pending:
                # wait on the OLDEST future to keep tar order deterministic
                f = pending.pop(0)
                key, thumb_bytes, err_msg = f.result()
                last_key = key

                if err_msg:
                    err += 1
                else:
                    tar_path = key_to_tar_path(key)
                    info = tarfile.TarInfo(name=tar_path)
                    info.size = len(thumb_bytes)
                    info.mtime = int(time.time())
                    info.mode = 0o644
                    tar.addfile(info, io.BytesIO(thumb_bytes))
                    ok += 1
                processed += 1

                # schedule next
                try:
                    pending.append(ex.submit(process, next(keys_iter)))
                except StopIteration:
                    pass

                # periodic checkpoint
                now = time.time()
                if now - last_report >= 30:
                    elapsed = now - start
                    rate = processed / elapsed
                    state_blob = uploader.to_state()
                    state_blob.update({"processed": processed, "ok": ok, "err": err,
                                        "last_key": last_key,
                                        "elapsed_s": round(elapsed)})
                    write_state(state_blob)
                    log(f"processed={processed:,} ok={ok:,} err={err}  "
                        f"rate={rate:.1f} fps  elapsed={elapsed/3600:.2f}h  "
                        f"last={last_key}")
                    last_report = now

        tar.close()
        uploader.complete()
        STATE_FILE.unlink(missing_ok=True)
        log(f"FINAL: processed={processed:,} ok={ok:,} err={err}  "
            f"size={uploader.total_bytes/1024**3:.2f} GB  "
            f"wall={(time.time()-start)/3600:.2f}h")
    except KeyboardInterrupt:
        log("Interrupted — state saved, you can --resume")
        state_blob = uploader.to_state()
        state_blob.update({"processed": processed, "ok": ok, "err": err,
                            "last_key": last_key})
        write_state(state_blob)
        raise


if __name__ == "__main__":
    main()
