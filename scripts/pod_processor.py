"""Runs on a RunPod GPU pod. Processes its assigned chunk of the 1.55M
Google Merchant list by:

  1. Fetching dispatch/chunk-{N}.txt from R2 (a list of paths like '30/abc.webp')
  2. Downloading batches of source images from R2 unpacked/ to /tmp/in/
  3. Running prototype.py --mode resize on the batch (Algorithm Dima)
  4. Uploading results to R2 <hh>/processed/<archive_id>/<filename>
  5. Reporting progress every batch to R2 dispatch/progress-{N}.json

Environment expected on pod (set by dispatch_pods.py via env vars):
  R2_ACCESS_KEY_ID
  R2_SECRET_ACCESS_KEY
  R2_ENDPOINT       — e.g. https://<account>.r2.cloudflarestorage.com
  R2_BUCKET         — e.g. allzap-photos

Args:
  --chunk-id   integer 0..N-1
  --batch-size files per prototype.py run (default 5000)
  --threads    parallel R2 download/upload threads (default 32)

Usage on pod:
  python pod_processor.py --chunk-id 0
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess

# Load env from /etc/rp_environment (RunPod populates this) or /workspace/pod_env.sh.
# RunPod processes inherit env vars only from PID 1, not via SSH shell.
def _load_env_from_files():
    for env_file in ("/etc/rp_environment", "/workspace/pod_env.sh"):
        try:
            for line in open(env_file):
                line = line.strip()
                # both formats: "KEY=VALUE" and "export KEY=VALUE"
                if line.startswith("export "):
                    line = line[7:]
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        except (FileNotFoundError, PermissionError):
            pass

_load_env_from_files()
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig

POD_TMP = Path("/tmp/pod_processor")
IN_DIR = POD_TMP / "in"
OUT_DIR = POD_TMP / "out"

# repo root (where prototype.py lives) — pod startup script clones repo here
REPO = Path(os.environ.get("REPO_DIR", "/workspace/foto-klass"))
PROTOTYPE = REPO / "prototype.py"


def parse_endpoint(raw: str) -> str:
    """Strip any bucket suffix from R2_ENDPOINT to get base endpoint."""
    p = raw.split("://", 1)
    return p[0] + "://" + p[1].split("/", 1)[0]


def make_s3():
    return boto3.client(
        "s3",
        endpoint_url=parse_endpoint(os.environ["R2_ENDPOINT"]),
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=BotoConfig(retries={"max_attempts": 10, "mode": "standard"},
                          max_pool_connections=128),
    )


def chunk_key(chunk_id: int) -> str:
    return f"dispatch/chunk-{chunk_id:03d}.txt"


def progress_key(chunk_id: int, start: int = 0, end: int | None = None) -> str:
    if start == 0 and end is None:
        return f"dispatch/progress-{chunk_id:03d}.json"
    return f"dispatch/progress-{chunk_id:03d}_s{start}_e{end if end is not None else 'all'}.json"


def list_to_r2_keys(paths: list[str]) -> list[tuple[str, str, str]]:
    """Convert 'archive_id/filename.webp' → (r2_in_key, archive_id, filename).

    R2 input scheme: <filename[:2]>/unpacked/<archive_id>/<filename>
    """
    result = []
    for p in paths:
        parts = p.strip().strip('"').split("/")
        if len(parts) != 2:
            continue
        archive_id, filename = parts
        shard = filename[:2].lower()
        r2_in_key = f"{shard}/unpacked/{archive_id}/{filename}"
        result.append((r2_in_key, archive_id, filename))
    return result


def fetch_chunk(s3, bucket: str, chunk_id: int) -> list[tuple[str, str, str]]:
    obj = s3.get_object(Bucket=bucket, Key=chunk_key(chunk_id))
    text = obj["Body"].read().decode("utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith('"img"')]
    return list_to_r2_keys(lines)


def download_one(s3, bucket: str, key: str, local_path: Path) -> bool:
    try:
        s3.download_file(bucket, key, str(local_path))
        return True
    except Exception:
        return False


def upload_one(s3, bucket: str, local_path: Path, key: str) -> bool:
    try:
        s3.upload_file(str(local_path), bucket, key,
                       ExtraArgs={"ContentType": "image/webp"})
        return True
    except Exception:
        return False


def report_progress(s3, bucket: str, chunk_id: int, payload: dict,
                    start: int = 0, end: int | None = None):
    body = json.dumps(payload, indent=2).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=progress_key(chunk_id, start, end), Body=body,
                  ContentType="application/json")


def run_prototype(in_dir: Path, out_dir: Path, batch_size: int = 16):
    cmd = [
        sys.executable, str(PROTOTYPE),
        "--input", str(in_dir),
        "--output", str(out_dir),
        "--mode", "resize",
        "--device", "cuda",
        "--batch-size", str(batch_size),
    ]
    subprocess.check_call(cmd, cwd=str(REPO))


def head_exists(s3, bucket: str, key: str) -> bool:
    """Return True if R2 object at key already exists."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def filter_existing(s3, bucket: str, items: list[tuple[str, str, str]],
                    threads: int) -> tuple[list[tuple[str, str, str]], int]:
    """Filter out items whose target R2 processed key already exists.
    Returns (new_items, skipped_count). HEAD is cheap relative to download+process."""
    def check(item):
        _, archive_id, filename = item
        shard = filename[:2].lower()
        out_key = f"{shard}/processed/{archive_id}/{filename}"
        return item, head_exists(s3, bucket, out_key)
    keep: list[tuple[str, str, str]] = []
    skipped = 0
    with ThreadPoolExecutor(max_workers=threads) as ex:
        for item, exists in ex.map(check, items):
            if exists:
                skipped += 1
            else:
                keep.append(item)
    return keep, skipped


def run_prototype_safe(in_dir: Path, out_dir: Path) -> bool:
    """Run prototype.py; on crash split inputs in half and retry each, isolating
    bad files. Returns True if at least some output produced. Files that can't
    be processed are moved aside so subsequent halves can run."""
    inputs = sorted(in_dir.iterdir())
    if not inputs:
        return True
    try:
        run_prototype(in_dir, out_dir, batch_size=16)
        return True
    except subprocess.CalledProcessError:
        if len(inputs) == 1:
            # single file is bad — quarantine and give up on it
            bad = inputs[0]
            print(f"[pod_processor] BAD FILE quarantined: {bad.name}", flush=True)
            try:
                bad.unlink()
            except Exception:
                pass
            return False
        # split into halves, retry each in own temp dirs
        mid = len(inputs) // 2
        success = False
        for half_idx, half in enumerate((inputs[:mid], inputs[mid:])):
            half_in = in_dir.parent / f"in_half_{half_idx}"
            half_out = in_dir.parent / f"out_half_{half_idx}"
            if half_in.exists(): shutil.rmtree(half_in)
            if half_out.exists(): shutil.rmtree(half_out)
            half_in.mkdir(); half_out.mkdir()
            for f in half:
                try:
                    f.rename(half_in / f.name)
                except Exception:
                    pass
            try:
                if run_prototype_safe(half_in, half_out):
                    # move outputs back
                    for f in half_out.iterdir():
                        try: f.rename(out_dir / f.name)
                        except Exception: pass
                    success = True
            finally:
                shutil.rmtree(half_in, ignore_errors=True)
                shutil.rmtree(half_out, ignore_errors=True)
        return success


def process_batch(s3, bucket: str, items: list[tuple[str, str, str]],
                  threads: int, skip_existing: bool = True) -> dict:
    """Process one batch end-to-end: download → prototype → upload.
    If skip_existing=True, HEAD-checks R2 first to skip already-processed files."""
    if IN_DIR.exists(): shutil.rmtree(IN_DIR)
    if OUT_DIR.exists(): shutil.rmtree(OUT_DIR)
    IN_DIR.mkdir(parents=True); OUT_DIR.mkdir(parents=True)

    t_skip0 = time.time()
    skipped_existing = 0
    if skip_existing:
        items, skipped_existing = filter_existing(s3, bucket, items, threads)
    t_skip = time.time() - t_skip0

    if not items:
        return {"ok": 0, "dl_fail": 0, "up_fail": 0, "skipped": skipped_existing,
                "t_skip": t_skip, "t_dl": 0, "t_proc": 0, "t_up": 0}

    t0 = time.time()

    # download in parallel
    def dl(item):
        r2_key, archive_id, filename = item
        return r2_key, download_one(s3, bucket, r2_key, IN_DIR / filename), archive_id, filename

    dl_ok = 0
    dl_fail = 0
    valid_items: list[tuple[str, str, str]] = []
    with ThreadPoolExecutor(max_workers=threads) as ex:
        for r2_key, ok, archive_id, filename in ex.map(dl, items):
            if ok:
                dl_ok += 1
                valid_items.append((r2_key, archive_id, filename))
            else:
                dl_fail += 1
    t_dl = time.time() - t0

    if dl_ok == 0:
        return {"ok": 0, "dl_fail": dl_fail, "up_fail": 0, "skipped": skipped_existing,
                "t_skip": t_skip, "t_dl": t_dl, "t_proc": 0, "t_up": 0}

    # process via prototype.py (Algorithm Dima) with bad-file isolation
    t1 = time.time()
    run_prototype_safe(IN_DIR, OUT_DIR)
    t_proc = time.time() - t1

    # upload results in parallel
    t2 = time.time()

    def up(item):
        _, archive_id, filename = item
        local = OUT_DIR / filename
        if not local.exists():
            return False
        shard = filename[:2].lower()
        out_key = f"{shard}/processed/{archive_id}/{filename}"
        return upload_one(s3, bucket, local, out_key)

    up_ok = 0
    up_fail = 0
    with ThreadPoolExecutor(max_workers=threads) as ex:
        for ok in ex.map(up, valid_items):
            if ok: up_ok += 1
            else: up_fail += 1
    t_up = time.time() - t2

    # cleanup
    shutil.rmtree(IN_DIR, ignore_errors=True)
    shutil.rmtree(OUT_DIR, ignore_errors=True)

    return {
        "ok": up_ok, "dl_fail": dl_fail, "up_fail": up_fail,
        "skipped": skipped_existing,
        "t_skip": t_skip, "t_dl": t_dl, "t_proc": t_proc, "t_up": t_up,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-id", type=int, required=True)
    ap.add_argument("--start", type=int, default=0,
                    help="slice items[start:end] of chunk list (for parallel speedup)")
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=5000,
                    help="files per prototype.py invocation")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--resume-from", type=int, default=0,
                    help="skip first N items (for resume after crash)")
    args = ap.parse_args()

    bucket = os.environ["R2_BUCKET"]
    s3 = make_s3()

    POD_TMP.mkdir(parents=True, exist_ok=True)
    print(f"[chunk-{args.chunk_id}] fetching chunk list...", flush=True)
    items = fetch_chunk(s3, bucket, args.chunk_id)
    print(f"[chunk-{args.chunk_id}] fetched {len(items)} total items", flush=True)
    if args.end is not None or args.start > 0:
        items = items[args.start:args.end]
        print(f"[chunk-{args.chunk_id}] sliced to items[{args.start}:{args.end}] = {len(items)}", flush=True)

    if args.resume_from > 0:
        items = items[args.resume_from:]
        print(f"[chunk-{args.chunk_id}] resuming after {args.resume_from}, {len(items)} left", flush=True)

    total_ok = 0
    total_dl_fail = 0
    total_up_fail = 0
    t0 = time.time()
    start_offset = args.resume_from

    for batch_start in range(0, len(items), args.batch_size):
        batch = items[batch_start:batch_start + args.batch_size]
        bi = batch_start + start_offset
        print(f"[chunk-{args.chunk_id}] batch {bi}-{bi+len(batch)} starting...", flush=True)

        res = process_batch(s3, bucket, batch, args.threads)
        total_ok += res["ok"]
        total_dl_fail += res["dl_fail"]
        total_up_fail += res["up_fail"]

        elapsed = time.time() - t0
        done_now = bi + len(batch)
        rate = total_ok / elapsed if elapsed > 0 else 0
        total_items = len(items) + start_offset
        remaining = total_items - done_now
        eta_s = remaining / rate if rate > 0 else 0

        progress = {
            "chunk_id": args.chunk_id,
            "total": total_items,
            "done": done_now,
            "ok": total_ok,
            "dl_fail": total_dl_fail,
            "up_fail": total_up_fail,
            "rate_fps": round(rate, 2),
            "elapsed_s": round(elapsed),
            "eta_s": round(eta_s),
            "batch_times": {
                "dl_s": round(res["t_dl"], 1),
                "proc_s": round(res["t_proc"], 1),
                "up_s": round(res["t_up"], 1),
            },
        }
        report_progress(s3, bucket, args.chunk_id, progress, args.start, args.end)
        print(f"[chunk-{args.chunk_id}] {json.dumps(progress)}", flush=True)

    # final summary
    final = {
        "chunk_id": args.chunk_id,
        "status": "done",
        "total": len(items) + start_offset,
        "ok": total_ok,
        "dl_fail": total_dl_fail,
        "up_fail": total_up_fail,
        "elapsed_s": round(time.time() - t0),
    }
    report_progress(s3, bucket, args.chunk_id, final, args.start, args.end)
    print(f"[chunk-{args.chunk_id}] DONE: {json.dumps(final)}", flush=True)


if __name__ == "__main__":
    main()
