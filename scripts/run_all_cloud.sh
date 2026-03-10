#!/bin/bash
# FR-Moshi — Script pipeline complet pour RunPod
# Lance via: nohup bash /runpod-volume/FR-Moshi/scripts/run_all_cloud.sh > /runpod-volume/pipeline.log 2>&1 &
#
# Tout est stocke sur /runpod-volume/ pour persister entre les redemarrages.
set -e
export PYTHONUNBUFFERED=1

# HF_TOKEN must be set as environment variable on the pod
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set. Export it before running this script."
    exit 1
fi
VOLDIR="/runpod-volume"

log() { echo "[$(date +%H:%M:%S)] $1"; }

log "=== [1/7] System deps ==="
apt-get update -qq && apt-get install -y -qq ffmpeg libsndfile1 > /dev/null 2>&1
log "OK"

log "=== [2/7] Clone FR-Moshi ==="
cd "$VOLDIR"
if [ ! -d "FR-Moshi" ]; then
    git clone https://github.com/Imtoocompedidiv/FR-Moshi.git 2>&1
fi
cd FR-Moshi

log "=== [3/7] Clone and install moshi-finetune ==="
if [ ! -d "moshi-finetune" ]; then
    git clone https://github.com/kyutai-labs/moshi-finetune.git 2>&1
fi
cd moshi-finetune
pip install sphn==0.2.1 2>&1 | tail -2
pip install --no-deps -e . 2>&1 | tail -2
pip install moshi 2>&1 | tail -2
cd ..

pip install -q "datasets<4" soundfile librosa whisper-timestamped huggingface_hub wandb torchaudio 2>&1 | tail -3
log "Deps OK"

log "=== [4/7] HuggingFace login ==="
python << 'PYLOGIN'
from huggingface_hub import login
import os
login(token=os.environ['HF_TOKEN'])
print('Logged in')
PYLOGIN

log "=== [5/7] Download Moshiko ==="
if [ ! -f "models/moshiko/model.safetensors" ]; then
    python << 'PYMODEL'
from huggingface_hub import snapshot_download
import os
snapshot_download('kyutai/moshiko-pytorch-bf16', local_dir='models/moshiko', token=os.environ['HF_TOKEN'])
print('Moshiko downloaded')
PYMODEL
else
    log "Moshiko already present"
fi

log "=== [6/7] Prepare SUMM-RE data ==="
if [ ! -f "data/moshi_dataset/train.jsonl" ]; then
    log "  Preparing train split (25h)..."
    python scripts/00_prepare_summ_re.py --split train --output-dir data/moshi_dataset --max-hours 25
    log "  Train data done"
else
    log "  train.jsonl already present"
fi

if [ ! -f "data/moshi_dataset/eval.jsonl" ]; then
    log "  Preparing eval split (3h)..."
    python scripts/00_prepare_summ_re.py --split dev --output-dir data/moshi_dataset --max-hours 3
    log "  Eval data done"
else
    log "  eval.jsonl already present"
fi

python << 'PYSTATS'
import json
for split in ['train', 'eval']:
    path = f'data/moshi_dataset/{split}.jsonl'
    try:
        with open(path) as f:
            entries = [json.loads(l) for l in f]
        hours = sum(e['duration'] for e in entries) / 3600
        print(f'  {split}: {len(entries)} files, {hours:.1f}h')
    except FileNotFoundError:
        print(f'  {split}: not found')
PYSTATS

log "=== [7a/7] Annotation with Whisper ==="
cd moshi-finetune

log "  Annotating train..."
python annotate.py ../data/moshi_dataset/train.jsonl --lang fr --whisper_model medium --local -v 2>&1

log "  Annotating eval..."
python annotate.py ../data/moshi_dataset/eval.jsonl --lang fr --whisper_model medium --local -v 2>&1

cd ..

log "=== [7b/7] Training ==="
cd moshi-finetune
log "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
log "  Config: ../configs/french_lora.yaml"

torchrun --nproc-per-node 1 -m train ../configs/french_lora.yaml 2>&1

cd ..

log "=== PIPELINE COMPLETE ==="
ls -la runs/french_lora/checkpoints/ 2>/dev/null || log "(no checkpoints found)"
log "Done!"
