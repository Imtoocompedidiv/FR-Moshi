#!/bin/bash
set -e
export PYTHONUNBUFFERED=1

cd /workspace/FR-Moshi
log() { echo "[$(date +%H:%M:%S)] $1"; }

log "=== [1/6] SUMM-RE v2 all splits ==="
for SPLIT in train dev test; do
    log "  Processing SUMM-RE split: $SPLIT"
    python scripts/00_prepare_summ_re_v2.py \
        --split "$SPLIT" \
        --output-dir "data/moshi_dataset_v2/split_${SPLIT}" \
        --max-hours 999 \
        --min-quality 0.4 \
        --min-seg 15 \
        --max-seg 120
done
log "SUMM-RE done"

log "=== [2/6] ESLO ==="
python scripts/00_prepare_eslo.py \
    --output-dir data/eslo_dataset \
    --max-hours 300 \
    --min-quality 0.4 \
    --min-seg 15 \
    --max-seg 120
log "ESLO done"

log "=== [3/6] Merge datasets ==="
python scripts/merge_datasets.py
log "Merge done"

log "=== [4/6] Whisper annotation ==="
cd moshi-finetune
python annotate.py ../data/merged_v2/train.jsonl --lang fr --whisper_model medium --local -v 2>&1
log "Train annotated"
python annotate.py ../data/merged_v2/eval.jsonl --lang fr --whisper_model medium --local -v 2>&1
log "Eval annotated"
cd ..

log "=== [5/6] Training ==="
cd moshi-finetune
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
torchrun --nproc-per-node 1 -m train ../configs/french_lora_v2.yaml 2>&1
cd ..

log "=== [6/6] PIPELINE COMPLETE ==="
ls -la runs/french_lora_v2/checkpoints/ 2>/dev/null || log "(no checkpoints)"
log "Done!"
