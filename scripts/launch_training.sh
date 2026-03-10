#!/bin/bash
# FR-Moshi — Script de lancement complet sur RunPod
# Execute tout le pipeline : setup -> data -> annotation -> training
#
# Usage:
#   export HF_TOKEN='hf_...'
#   bash scripts/launch_training.sh
#
# Budget estime : ~$3-5 pour un run complet sur A6000 48GB

set -e

echo "============================================"
echo "FR-Moshi — Pipeline Complet"
echo "============================================"
echo ""

# ============================================
# ETAPE 0 : Verification de l'environnement
# ============================================
echo "[0/6] Verification de l'environnement..."

if [ -z "$HF_TOKEN" ]; then
    echo "  ERREUR: HF_TOKEN non defini."
    echo "  export HF_TOKEN='hf_...'"
    exit 1
fi

python -c "
import torch
assert torch.cuda.is_available(), 'CUDA non disponible'
gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_mem / 1e9
print(f'  GPU: {gpu} ({vram:.0f} GB)')
assert vram >= 20, f'VRAM insuffisante: {vram:.0f}GB < 20GB minimum'
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.version.cuda}')
"
echo "  OK"

# ============================================
# ETAPE 1 : Installation des dependances
# ============================================
echo ""
echo "[1/6] Installation des dependances..."

apt-get update -qq && apt-get install -y -qq ffmpeg libsndfile1 > /dev/null 2>&1 || true

# moshi-finetune
if [ ! -d "moshi-finetune" ]; then
    git clone https://github.com/kyutai-labs/moshi-finetune.git
    cd moshi-finetune && pip install -e . && cd ..
    echo "  moshi-finetune installe"
else
    echo "  moshi-finetune deja present"
fi

pip install -q \
    datasets \
    soundfile \
    whisper-timestamped \
    sphn \
    huggingface_hub \
    torchaudio \
    wandb 2>/dev/null

huggingface-cli login --token $HF_TOKEN 2>/dev/null || true
echo "  Dependances installees"

# ============================================
# ETAPE 2 : Telecharger le modele de base
# ============================================
echo ""
echo "[2/6] Telechargement de Moshiko..."

if [ ! -d "models/moshiko" ]; then
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('kyutai/moshiko-pytorch-bf16', local_dir='models/moshiko')
print('  Moshiko telecharge')
"
else
    echo "  Moshiko deja present"
fi

# ============================================
# ETAPE 3 : Preparer les donnees SUMM-RE
# ============================================
echo ""
echo "[3/6] Preparation des donnees SUMM-RE..."

# Train : 25h de donnees
if [ ! -f "data/moshi_dataset/train.jsonl" ]; then
    python scripts/00_prepare_summ_re.py \
        --split train \
        --output-dir data/moshi_dataset \
        --max-hours 25
    echo "  Donnees train preparees"
else
    echo "  train.jsonl deja present"
fi

# Eval : 3h de donnees (split dev = manuellement transcrit = meilleure qualite)
if [ ! -f "data/moshi_dataset/eval.jsonl" ]; then
    python scripts/00_prepare_summ_re.py \
        --split dev \
        --output-dir data/moshi_dataset \
        --max-hours 3
    # Renommer si necessaire (le script genere eval.jsonl pour split!=train)
    echo "  Donnees eval preparees"
else
    echo "  eval.jsonl deja present"
fi

# Stats
echo ""
echo "  === Dataset ==="
python -c "
import json
for split in ['train', 'eval']:
    path = f'data/moshi_dataset/{split}.jsonl'
    try:
        with open(path) as f:
            entries = [json.loads(l) for l in f]
        hours = sum(e['duration'] for e in entries) / 3600
        print(f'  {split}: {len(entries)} fichiers, {hours:.1f}h')
    except FileNotFoundError:
        print(f'  {split}: non trouve')
"

# ============================================
# ETAPE 4 : Annotation (whisper_timestamped)
# ============================================
echo ""
echo "[4/6] Annotation avec Whisper (canal gauche seulement)..."

cd moshi-finetune

# Annoter train
echo "  Annotation train.jsonl..."
python annotate.py ../data/moshi_dataset/train.jsonl \
    --lang fr \
    --whisper_model medium \
    --local \
    -v

# Annoter eval
echo "  Annotation eval.jsonl..."
python annotate.py ../data/moshi_dataset/eval.jsonl \
    --lang fr \
    --whisper_model medium \
    --local \
    -v

cd ..

# Verifier les annotations
echo ""
echo "  === Verification annotations ==="
python -c "
import json
from pathlib import Path
for split in ['train', 'eval']:
    jsonl = f'data/moshi_dataset/{split}.jsonl'
    with open(jsonl) as f:
        entries = [json.loads(l) for l in f]
    annotated = 0
    for e in entries:
        json_path = Path('data/moshi_dataset') / e['path'].replace('.wav', '.json')
        if json_path.exists():
            annotated += 1
    print(f'  {split}: {annotated}/{len(entries)} annotes')
"

# ============================================
# ETAPE 5 : Entrainement
# ============================================
echo ""
echo "[5/6] Lancement de l'entrainement..."
echo "  Config : configs/french_lora.yaml"
echo "  GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

cd moshi-finetune

torchrun --nproc-per-node 1 -m train ../configs/french_lora.yaml

cd ..

# ============================================
# ETAPE 6 : Resultats
# ============================================
echo ""
echo "[6/6] Entrainement termine !"
echo ""
echo "============================================"
echo "Resultats dans : runs/french_lora/"
echo ""

ls -la runs/french_lora/checkpoints/ 2>/dev/null || echo "  (pas de checkpoints trouves)"

echo ""
echo "Pour telecharger les poids LoRA :"
echo "  scp -r user@pod:/workspace/runs/french_lora/checkpoints/ ./local_checkpoints/"
echo ""
echo "Pour inference :"
echo "  python -m moshi.server --lora-weight runs/french_lora/checkpoints/best/lora.safetensors"
echo "============================================"
