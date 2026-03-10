"""
FR-Moshi — Étape 3 : Annotation / Transcription
Ce script est un WRAPPER autour de annotate.py du repo moshi-finetune.

annotate.py fait tout :
  - Charge chaque WAV stéréo
  - Extrait le canal gauche (canal 0 = Moshi)
  - Resample à 16kHz pour Whisper
  - Transcrit avec whisper_timestamped (PAS WhisperX)
  - Génère un .json par .wav au format :
    {"alignments": [["mot", [start, end], "SPEAKER_MAIN"], ...]}

Usage:
  # Depuis le dossier moshi-finetune :
  python annotate.py ../data/moshi_dataset/train.jsonl --lang fr --local

  # Ou via ce wrapper :
  python scripts/03_transcribe.py

Prérequis:
  - moshi-finetune cloné et installé (pip install -e .)
  - whisper_timestamped installé (pip install whisper-timestamped)
  - GPU avec CUDA
"""

import subprocess
import sys
from pathlib import Path


def run_annotate(
    jsonl_path: str = "data/moshi_dataset/train.jsonl",
    lang: str = "fr",
    whisper_model: str = "medium",
    moshi_finetune_dir: str = "moshi-finetune",
):
    """Lance annotate.py depuis le repo moshi-finetune."""

    jsonl_path = Path(jsonl_path).resolve()
    moshi_dir = Path(moshi_finetune_dir).resolve()
    annotate_script = moshi_dir / "annotate.py"

    # Vérifications
    if not annotate_script.exists():
        print(f"✗ annotate.py introuvable dans {moshi_dir}")
        print(f"  Cloner le repo : git clone https://github.com/kyutai-labs/moshi-finetune")
        sys.exit(1)

    if not jsonl_path.exists():
        print(f"✗ JSONL introuvable : {jsonl_path}")
        print(f"  Exécuter d'abord : python scripts/03_prepare_dataset.py")
        sys.exit(1)

    # Vérifier que whisper_timestamped est installé
    try:
        import whisper_timestamped
        print(f"  ✓ whisper_timestamped {whisper_timestamped.__version__}")
    except ImportError:
        print("✗ whisper_timestamped non installé")
        print("  pip install whisper-timestamped")
        sys.exit(1)

    print(f"=== FR-Moshi — Annotation avec Whisper ({whisper_model}) ===")
    print(f"  JSONL    : {jsonl_path}")
    print(f"  Langue   : {lang}")
    print(f"  Modèle   : {whisper_model}")
    print(f"  Script   : {annotate_script}")
    print()

    # IMPORTANT : --local pour exécuter localement (sans SLURM)
    # --whisper_model medium : recommandé pour stéréo avec VAD
    cmd = [
        sys.executable,
        str(annotate_script),
        str(jsonl_path),
        "--lang", lang,
        "--whisper_model", whisper_model,
        "--local",
    ]

    print(f"Commande : {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=str(moshi_dir))

    if result.returncode == 0:
        print(f"\n✓ Annotation terminée.")
        print(f"  Les fichiers .json ont été créés à côté de chaque .wav")
        print(f"\nProchaine étape :")
        print(f"  torchrun --nproc-per-node 1 -m train ../configs/french_lora.yaml")
    else:
        print(f"\n✗ Erreur lors de l'annotation (code {result.returncode})")
        sys.exit(result.returncode)


# Aussi exécuter annotate.py sur eval.jsonl si il existe
def run_annotate_all(
    dataset_dir: str = "data/moshi_dataset",
    lang: str = "fr",
    whisper_model: str = "medium",
    moshi_finetune_dir: str = "moshi-finetune",
):
    """Annote train.jsonl et eval.jsonl."""
    dataset_dir = Path(dataset_dir)

    for split in ["train.jsonl", "eval.jsonl"]:
        jsonl_path = dataset_dir / split
        if jsonl_path.exists():
            print(f"\n{'='*60}")
            print(f"Annotation de {split}...")
            print(f"{'='*60}")
            run_annotate(
                str(jsonl_path), lang, whisper_model, moshi_finetune_dir
            )
        else:
            print(f"  ⏭ {split} non trouvé, ignoré")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="FR-Moshi — Wrapper pour annotate.py (moshi-finetune)"
    )
    parser.add_argument("--dataset-dir", default="data/moshi_dataset",
                        help="Répertoire du dataset")
    parser.add_argument("--lang", default="fr", help="Langue (défaut: fr)")
    parser.add_argument("--whisper-model", default="medium",
                        help="Modèle Whisper (défaut: medium, recommandé pour stéréo)")
    parser.add_argument("--moshi-finetune-dir", default="moshi-finetune",
                        help="Chemin vers le repo moshi-finetune")
    args = parser.parse_args()

    run_annotate_all(
        args.dataset_dir, args.lang, args.whisper_model, args.moshi_finetune_dir
    )
