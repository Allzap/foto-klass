#!/bin/bash
# Pod startup script — fetched by minimal dockerArgs to avoid RunPod 800-char limit.
# Expects env vars: R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT, R2_BUCKET, CHUNK_ID
set -e
export PYTHONUNBUFFERED=1
cd /workspace
[ -d foto-klass ] || git clone --depth 1 -b main https://github.com/Allzap/foto-klass
cd foto-klass
pip install -q -r requirements.txt boto3 huggingface_hub 2>&1 | tail -3
mkdir -p weights
huggingface-cli download Phips/4xNomosWebPhoto_RealPLKSR \
    4xNomosWebPhoto_RealPLKSR.safetensors --local-dir weights 2>&1 | tail -1
echo "STARTUP DONE — running pod_processor for chunk ${CHUNK_ID}"
python scripts/pod_processor.py --chunk-id "${CHUNK_ID}" 2>&1 | tee /workspace/pod-${CHUNK_ID}.log
echo "POD DONE — terminating self"
runpodctl remove pod $RUNPOD_POD_ID 2>&1 || true
