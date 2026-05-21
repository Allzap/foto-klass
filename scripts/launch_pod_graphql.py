"""Launch RunPod pods via GraphQL (dockerArgs reliably executes vs REST V1).

Usage:
  python scripts/launch_pod_graphql.py --pilot           # launch ONE pilot pod (chunk 0)
  python scripts/launch_pod_graphql.py --pods 20         # launch all 20 chunks
  python scripts/launch_pod_graphql.py --terminate-all   # kill everything
  python scripts/launch_pod_graphql.py --watch           # watch progress files
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
PODS_RECORD = ROOT / ".planning" / "pods.json"
REPO_URL = "https://github.com/Allzap/foto-klass"
REPO_BRANCH = "main"

GRAPHQL = "https://api.runpod.io/graphql"


def load_env():
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def make_docker_args(chunk_id: int) -> str:
    """Minimal dockerArgs: fetches pod_startup.sh from github (committed to main)
    and runs it. CHUNK_ID is passed via env.

    Why minimal: RunPod has an undocumented limit (~800 chars) on dockerArgs that
    silently breaks allocation. Anything longer keeps uptime negative forever.
    """
    return (
        "bash -lc 'curl -fsSL "
        "https://raw.githubusercontent.com/Allzap/foto-klass/main/scripts/pod_startup.sh "
        "| bash'"
    )


def create_pod_gql(chunk_id: int, cloud: str = "SECURE",
                   start: int | None = None, end: int | None = None) -> dict:
    api_key = os.environ["RUNPOD_API_KEY"]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    mutation = """
    mutation deployPod($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id
        imageName
        machineId
        desiredStatus
      }
    }
    """
    name_suffix = ""
    if start is not None or end is not None:
        name_suffix = f"-s{start or 0}-e{end if end is not None else 'all'}"
    env_vars = [
        {"key": "R2_ACCESS_KEY_ID",     "value": os.environ["R2_ACCESS_KEY_ID"]},
        {"key": "R2_SECRET_ACCESS_KEY", "value": os.environ["R2_SECRET_ACCESS_KEY"]},
        {"key": "R2_ENDPOINT",          "value": os.environ["R2_ENDPOINT"]},
        {"key": "R2_BUCKET",            "value": os.environ["R2_BUCKET"]},
        {"key": "CHUNK_ID",             "value": str(chunk_id)},
    ]
    if start is not None:
        env_vars.append({"key": "CHUNK_START", "value": str(start)})
    if end is not None:
        env_vars.append({"key": "CHUNK_END", "value": str(end)})
    variables = {
        "input": {
            "name": f"fk-chunk-{chunk_id:03d}{name_suffix}",
            "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
            "gpuTypeId": "NVIDIA GeForce RTX 4090",
            "gpuCount": 1,
            "cloudType": cloud,
            "containerDiskInGb": 40,
            "volumeInGb": 0,
            "dockerArgs": make_docker_args(chunk_id),
            "env": env_vars,
        }
    }
    with httpx.Client(timeout=60) as c:
        r = c.post(GRAPHQL, headers=headers, json={"query": mutation, "variables": variables})
        j = r.json()
        if j.get("errors"):
            raise RuntimeError(f"GraphQL error: {json.dumps(j['errors'])}")
        return j["data"]["podFindAndDeployOnDemand"]


def terminate_pod_gql(pod_id: str):
    api_key = os.environ["RUNPOD_API_KEY"]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    mutation = """
    mutation($input: PodTerminateInput!) {
      podTerminate(input: $input)
    }
    """
    with httpx.Client(timeout=30) as c:
        c.post(GRAPHQL, headers=headers,
               json={"query": mutation, "variables": {"input": {"podId": pod_id}}})


def cmd_pilot(args):
    print(f"Launching pilot pod for chunk 0 ({args.cloud})...")
    res = create_pod_gql(0, cloud=args.cloud)
    pod_id = res["id"]
    print(f"  pod_id={pod_id}  status={res.get('desiredStatus')}")

    rec = {"pods": {"0": {"pod_id": pod_id, "response": res}},
           "started_at": time.time(), "n": 1, "pilot": True}
    PODS_RECORD.parent.mkdir(parents=True, exist_ok=True)
    PODS_RECORD.write_text(json.dumps(rec, indent=2))
    print(f"Pilot saved. Now run: python scripts/launch_pod_graphql.py --watch")


def cmd_launch(args):
    print(f"Launching {args.pods} pods via GraphQL ({args.cloud})...")
    rec = {"pods": {}, "started_at": time.time(), "n": args.pods}
    PODS_RECORD.parent.mkdir(parents=True, exist_ok=True)
    for i in range(args.pods):
        try:
            r = create_pod_gql(i, cloud=args.cloud)
            rec["pods"][str(i)] = {"pod_id": r["id"], "response": r}
            print(f"  chunk-{i:03d}: {r['id']}")
        except Exception as e:
            print(f"  chunk-{i:03d}: FAILED {str(e)[:200]}")
            rec["pods"][str(i)] = {"error": str(e)}
        PODS_RECORD.write_text(json.dumps(rec, indent=2))
        time.sleep(1)


def parse_spec(spec: str):
    """ '5:0-38760' → (5, 0, 38760)
        '6'         → (6, None, None)
        '5:0-'      → (5, 0, None)        (start only)
    """
    if ":" not in spec:
        return int(spec), None, None
    cid, rng = spec.split(":", 1)
    if "-" not in rng:
        raise ValueError(f"bad spec {spec!r}, expected 'N:start-end'")
    a, b = rng.split("-", 1)
    start = int(a) if a else None
    end = int(b) if b else None
    return int(cid), start, end


def cmd_specs(args):
    specs = [parse_spec(s.strip()) for s in args.chunks.split(",") if s.strip()]
    print(f"Launching {len(specs)} pods on specs={specs} ({args.cloud})...")
    if PODS_RECORD.exists():
        rec = json.loads(PODS_RECORD.read_text())
        if "pods" not in rec: rec["pods"] = {}
    else:
        rec = {"pods": {}, "started_at": time.time()}
    PODS_RECORD.parent.mkdir(parents=True, exist_ok=True)
    for cid, st, en in specs:
        key = f"{cid:03d}" + (f"_s{st or 0}_e{en if en is not None else 'all'}"
                              if (st is not None or en is not None) else "")
        try:
            r = create_pod_gql(cid, cloud=args.cloud, start=st, end=en)
            rec["pods"][key] = {"pod_id": r["id"], "chunk_id": cid,
                                "start": st, "end": en, "response": r}
            print(f"  {key}: {r['id']}")
        except Exception as e:
            print(f"  {key}: FAILED {str(e)[:200]}")
            rec["pods"][key] = {"error": str(e), "chunk_id": cid,
                                "start": st, "end": en}
        PODS_RECORD.write_text(json.dumps(rec, indent=2))
        time.sleep(1)


def cmd_terminate_all(args):
    if not PODS_RECORD.exists():
        print("No pods record."); return
    rec = json.loads(PODS_RECORD.read_text())
    for k, v in rec.get("pods", {}).items():
        pid = v.get("pod_id")
        if pid:
            try:
                terminate_pod_gql(pid)
                print(f"  chunk-{k}: terminated {pid}")
            except Exception as e:
                print(f"  chunk-{k}: FAIL {e}")


def cmd_watch(args):
    """Poll R2 for actual processed/ keys + progress reports."""
    import boto3
    from concurrent.futures import ThreadPoolExecutor
    ep = os.environ["R2_ENDPOINT"].split("://", 1)
    endpoint = ep[0] + "://" + ep[1].split("/", 1)[0]
    s3 = boto3.client("s3", endpoint_url=endpoint,
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"], region_name="auto")
    bucket = os.environ["R2_BUCKET"]

    def count_shard(shard):
        try:
            n = 0
            for page in s3.get_paginator("list_objects_v2").paginate(
                    Bucket=bucket, Prefix=f"{shard}/processed/", MaxKeys=1000):
                n += len(page.get("Contents") or [])
            return n
        except: return 0

    last = None
    while True:
        # progress reports
        running_pods = 0
        total_ok_reports = 0
        for i in range(20):
            try:
                obj = s3.get_object(Bucket=bucket, Key=f"dispatch/progress-{i:03d}.json")
                prog = json.loads(obj["Body"].read())
                total_ok_reports += prog.get("ok", 0)
                running_pods += 1
            except: pass

        # actual processed file count
        with ThreadPoolExecutor(max_workers=32) as ex:
            counts = list(ex.map(count_shard, [f"{i:02x}" for i in range(256)]))
        total_files = sum(counts)

        now = time.strftime("%H:%M:%S")
        line = f"[{now}] pods reporting:{running_pods}/20  reported ok:{total_ok_reports:,}  R2 processed/ keys:{total_files:,}"
        if line != last:
            print(line, flush=True)
            last = line
        time.sleep(30)


def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="launch ONLY chunk 0 as pilot")
    ap.add_argument("--pods", type=int, default=0, help="launch N pods (chunks 0..N-1)")
    ap.add_argument("--chunks", type=str, default="",
                    help="comma-separated chunk specs e.g. '5:0-38760,5:38760-77520,6'")
    ap.add_argument("--terminate-all", action="store_true")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--cloud", default="SECURE", choices=["SECURE", "COMMUNITY"])
    args = ap.parse_args()
    if args.terminate_all: cmd_terminate_all(args)
    elif args.watch: cmd_watch(args)
    elif args.pilot: cmd_pilot(args)
    elif args.chunks: cmd_specs(args)
    elif args.pods: cmd_launch(args)
    else: ap.print_help()


if __name__ == "__main__":
    main()
