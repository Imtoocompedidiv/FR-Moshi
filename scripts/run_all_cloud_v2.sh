#!/bin/bash
# FR-Moshi — Run 2 Pipeline pour A6000 48GB
# Data: SUMM-RE v2 (dialogue-quality) + ESLO (interviews conversationnelles)
# Config: rank 128, ft_embed=true, LR 1e-5, duration 90s
#
# Usage: nohup bash /runpod-volume/FR-Moshi/scripts/run_all_cloud_v2.sh > /runpod-volume/pipeline_v2.log 2>&1 &
set -e
export PYTHONUNBUFFERED=1

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set."
    exit 1
fi

VOLDIR="/runpod-volume"
log() { echo "[$(date +%H:%M:%S)] $1"; }

log "=== [1/8] System deps ==="
apt-get update -qq && apt-get install -y -qq ffmpeg libsndfile1 > /dev/null 2>&1
log "OK"

log "=== [2/8] Clone FR-Moshi ==="
cd "$VOLDIR"
if [ ! -d "FR-Moshi" ]; then
    git clone https://github.com/Imtoocompedidiv/FR-Moshi.git 2>&1
else
    cd FR-Moshi && git pull && cd ..
fi
cd FR-Moshi

log "=== [3/8] Clone and install moshi-finetune ==="
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

log "=== [4/8] HuggingFace login ==="
python << 'PYLOGIN'
from huggingface_hub import login
import os
login(token=os.environ['HF_TOKEN'])
print('Logged in')
PYLOGIN

log "=== [5/8] Download Moshiko ==="
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

# ============================================================
# DATA PREPARATION — Two sources in parallel
# ============================================================

log "=== [6a/8] Prepare SUMM-RE v2 (dialogue-quality segments) ==="
# Use ALL splits (train+dev+test) — for fine-tuning data, split boundaries don't matter
if [ ! -f "data/moshi_dataset_v2/train_all.jsonl" ]; then
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

    # Merge all splits into one, then re-split 90/10 for train/eval
    python << 'PYMERGE_SUMM'
import json, random
from pathlib import Path
import shutil

all_entries = []
merged_stereo = Path("data/moshi_dataset_v2/data_stereo")
merged_stereo.mkdir(parents=True, exist_ok=True)

for split in ["train", "dev", "test"]:
    split_dir = Path(f"data/moshi_dataset_v2/split_{split}")
    jsonl = split_dir / "train.jsonl"  # v2 script names output train.jsonl
    if not jsonl.exists():
        jsonl = split_dir / "eval.jsonl"
    if not jsonl.exists():
        continue
    with open(jsonl) as f:
        for line in f:
            entry = json.loads(line)
            # Copy audio to merged stereo dir
            src = split_dir / entry["path"]
            dst = merged_stereo / Path(entry["path"]).name
            if src.exists() and not dst.exists():
                shutil.copy2(str(src), str(dst))
            entry["path"] = f"data_stereo/{Path(entry['path']).name}"
            all_entries.append(entry)

random.seed(42)
random.shuffle(all_entries)
split_idx = int(len(all_entries) * 0.9)
train_entries = all_entries[:split_idx]
eval_entries = all_entries[split_idx:]

out = Path("data/moshi_dataset_v2")
for name, entries in [("train", train_entries), ("eval", eval_entries)]:
    with open(out / f"{name}.jsonl", 'w') as f:
        for e in entries:
            json.dump(e, f, ensure_ascii=False)
            f.write('\n')
    hours = sum(e['duration'] for e in entries) / 3600
    print(f"  SUMM-RE {name}: {len(entries)} segments, {hours:.1f}h")

# Marker file
with open(out / "train_all.jsonl", 'w') as f:
    f.write("merged\n")
print("SUMM-RE merge done!")
PYMERGE_SUMM
    log "  SUMM-RE all splits merged"
else
    log "  SUMM-RE v2 already present"
fi

log "=== [6b/8] Prepare ESLO (interview dialogue segments) ==="
if [ ! -f "data/eslo_dataset/train.jsonl" ]; then
    log "  Processing ESLO conversations..."
    python scripts/00_prepare_eslo.py \
        --output-dir data/eslo_dataset \
        --max-hours 300 \
        --min-quality 0.4 \
        --min-seg 15 \
        --max-seg 120
    log "  ESLO done"
else
    log "  ESLO already present"
fi

log "=== [6c/8] Merge datasets ==="
# Merge SUMM-RE v2 + ESLO into unified dataset
python << 'PYMERGE'
import json
from pathlib import Path
import shutil

merged_dir = Path("data/merged_v2")
merged_stereo = merged_dir / "data_stereo"
merged_stereo.mkdir(parents=True, exist_ok=True)

for split in ['train', 'eval']:
    all_entries = []

    # SUMM-RE v2
    summ_re_path = Path(f"data/moshi_dataset_v2/{split}.jsonl")
    if summ_re_path.exists():
        with open(summ_re_path) as f:
            for line in f:
                entry = json.loads(line)
                # Copy audio file to merged dir
                src = Path("data/moshi_dataset_v2") / entry["path"]
                dst = merged_stereo / Path(entry["path"]).name
                if src.exists() and not dst.exists():
                    shutil.copy2(str(src), str(dst))
                entry["path"] = f"data_stereo/{Path(entry['path']).name}"
                all_entries.append(entry)
        print(f"  SUMM-RE {split}: {len(all_entries)} segments")

    # ESLO
    eslo_path = Path(f"data/eslo_dataset/{split}.jsonl")
    if eslo_path.exists():
        n_before = len(all_entries)
        with open(eslo_path) as f:
            for line in f:
                entry = json.loads(line)
                src = Path("data/eslo_dataset") / entry["path"]
                dst = merged_stereo / Path(entry["path"]).name
                if src.exists() and not dst.exists():
                    shutil.copy2(str(src), str(dst))
                entry["path"] = f"data_stereo/{Path(entry['path']).name}"
                all_entries.append(entry)
        print(f"  ESLO {split}: {len(all_entries) - n_before} segments")

    # Write merged JSONL
    jsonl_path = merged_dir / f"{split}.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    hours = sum(e['duration'] for e in all_entries) / 3600
    durations = [e['duration'] for e in all_entries]
    print(f"  MERGED {split}: {len(all_entries)} segments, {hours:.1f}h")
    if durations:
        print(f"    Duration range: {min(durations):.0f}s - {max(durations):.0f}s, "
              f"avg {sum(durations)/len(durations):.0f}s")

print("Merge done!")
PYMERGE

log "=== [7/8] Annotation with Whisper ==="
cd moshi-finetune

log "  Annotating merged train..."
python annotate.py ../data/merged_v2/train.jsonl --lang fr --whisper_model medium --local -v 2>&1

log "  Annotating merged eval..."
python annotate.py ../data/merged_v2/eval.jsonl --lang fr --whisper_model medium --local -v 2>&1

cd ..

log "=== [8/8] Training v2 ==="
cd moshi-finetune
log "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
log "  Config: ../configs/french_lora_v2.yaml"

torchrun --nproc-per-node 1 -m train ../configs/french_lora_v2.yaml 2>&1

cd ..

log "=== PIPELINE V2 COMPLETE ==="
ls -la runs/french_lora_v2/checkpoints/ 2>/dev/null || log "(no checkpoints found)"
log "Done!"
