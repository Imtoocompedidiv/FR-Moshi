#!/bin/bash
# FR-Moshi — Setup rapide sur RunPod
# Installe les dependances SANS lancer le training
# Pour le pipeline complet : bash scripts/launch_training.sh
#
# Usage: bash scripts/05_cloud_setup.sh

set -e

echo "============================================"
echo "FR-Moshi — Setup Cloud"
echo "============================================"

# 1. Dependances systeme
echo "[1/5] Dependances systeme..."
apt-get update -qq && apt-get install -y -qq ffmpeg libsndfile1 > /dev/null 2>&1
echo "  OK"

# 2. moshi-finetune
echo "[2/5] moshi-finetune..."
if [ ! -d "moshi-finetune" ]; then
    git clone https://github.com/kyutai-labs/moshi-finetune.git
    cd moshi-finetune && pip install -e . && cd ..
    echo "  Installe"
else
    echo "  Deja present"
fi

# 3. Dependances Python
echo "[3/5] Dependances Python..."
pip install -q \
    datasets \
    soundfile \
    whisper-timestamped \
    sphn \
    huggingface_hub \
    torchaudio \
    wandb
echo "  OK"

# 4. HuggingFace login
echo "[4/5] HuggingFace..."
if [ -z "$HF_TOKEN" ]; then
    echo "  ATTENTION: HF_TOKEN non defini"
    echo "  export HF_TOKEN='hf_...'"
else
    huggingface-cli login --token $HF_TOKEN
    echo "  Connecte"
fi

# 5. Verification GPU
echo "[5/5] GPU..."
python -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'  {name} ({vram:.0f} GB)')
else:
    print('  PAS DE GPU !')
"

echo ""
echo "============================================"
echo "Setup termine !"
echo ""
echo "Deux options :"
echo ""
echo "  Option A — Pipeline automatique :"
echo "    bash scripts/launch_training.sh"
echo ""
echo "  Option B — Etape par etape :"
echo "    1. python scripts/00_prepare_summ_re.py --max-hours 25"
echo "    2. python scripts/00_prepare_summ_re.py --split dev --max-hours 3"
echo "    3. cd moshi-finetune && python annotate.py ../data/moshi_dataset/train.jsonl --lang fr --local"
echo "    4. python annotate.py ../data/moshi_dataset/eval.jsonl --lang fr --local"
echo "    5. torchrun --nproc-per-node 1 -m train ../configs/french_lora.yaml"
echo "============================================"
