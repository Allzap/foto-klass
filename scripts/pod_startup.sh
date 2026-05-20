#!/bin/bash
# Pod startup script with progress markers to R2 (lets us monitor setup without SSH)
set -e
export PYTHONUNBUFFERED=1

mark() {
  # write a stage marker to R2 — we can poll this from outside
  python3 -c "
import boto3, os
ep = os.environ['R2_ENDPOINT'].split('://',1)
endpoint = ep[0] + '://' + ep[1].split('/',1)[0]
s3 = boto3.client('s3', endpoint_url=endpoint,
    aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'], region_name='auto')
s3.put_object(Bucket=os.environ['R2_BUCKET'],
    Key='dispatch/pod-${CHUNK_ID}-stage.txt',
    Body=b'$1')" 2>/dev/null || true
}

mark "stage0_starting"
cd /workspace

mark "stage1_git_clone"
[ -d foto-klass ] || git clone --depth 1 -b main https://github.com/Allzap/foto-klass
cd foto-klass

mark "stage2_pip_install"
pip install -q boto3 2>&1 | tail -1
pip install -q -r requirements.txt huggingface_hub 2>&1 | tail -3

mark "stage3_weights_download"
mkdir -p weights
huggingface-cli download Phips/4xNomosWebPhoto_RealPLKSR \
    4xNomosWebPhoto_RealPLKSR.safetensors --local-dir weights 2>&1 | tail -1

mark "stage4_running_processor"
echo "STARTUP DONE — running pod_processor for chunk ${CHUNK_ID}"
python scripts/pod_processor.py --chunk-id "${CHUNK_ID}" 2>&1 | tee /workspace/pod-${CHUNK_ID}.log

mark "stage5_done"
echo "POD DONE — terminating self"
runpodctl remove pod $RUNPOD_POD_ID 2>&1 || true
