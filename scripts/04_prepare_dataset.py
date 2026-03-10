"""
FR-Moshi — Étape 4 : Préparation du dataset au format moshi-finetune
Valide les WAV stéréo et génère le JSONL manifest.

IMPORTANT : Cette étape se fait AVANT annotate.py !
Le JSONL ne contient que {"path": "...", "duration": ...}
Les fichiers .json de transcription seront générés par annotate.py ensuite.

Pipeline :
  1. scripts/01_download_podcasts.py  → data/raw_audio/
  2. scripts/02_diarize_stereo.py     → data/stereo/
  3. scripts/04_prepare_dataset.py    → data/moshi_dataset/ (ce script)
  4. scripts/03_transcribe.py         → .json annotations (via annotate.py)
  5. torchrun -m train config.yaml    → fine-tuning
"""

import json
import shutil
import random
import argparse
from pathlib import Path


def validate_wav(wav_path: Path, min_duration: float = 10.0, max_duration: float = 300.0):
    """Vérifie qu'un fichier WAV est valide pour Moshi."""
    import wave
    try:
        with wave.open(str(wav_path), 'r') as w:
            sr = w.getframerate()
            channels = w.getnchannels()
            duration = w.getnframes() / sr

            if channels != 2:
                return False, f"channels {channels} != 2 (stéréo requis)"
            if duration < min_duration:
                return False, f"trop court ({duration:.0f}s < {min_duration:.0f}s)"
            if duration > max_duration:
                return False, f"trop long ({duration:.0f}s > {max_duration:.0f}s)"

            return True, {"duration": duration, "sample_rate": sr}
    except Exception as e:
        return False, str(e)


def resample_wav(wav_path: Path, output_path: Path, target_sr: int = 24000):
    """Resample un WAV stéréo à 24kHz si nécessaire."""
    import subprocess
    cmd = [
        "ffmpeg", "-i", str(wav_path),
        "-ar", str(target_sr),
        "-ac", "2",  # Garder stéréo !
        "-y",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def prepare_dataset(
    stereo_dir: str = "data/stereo",
    output_dir: str = "data/moshi_dataset",
    eval_ratio: float = 0.05,
    min_duration: float = 10.0,
    max_duration: float = 300.0,
    seed: int = 42,
):
    """Prépare le dataset final au format moshi-finetune."""
    stereo_dir = Path(stereo_dir)
    output_dir = Path(output_dir)
    output_stereo = output_dir / "data_stereo"
    output_stereo.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(stereo_dir.glob("*.wav"))
    print(f"Validation de {len(wav_files)} fichiers...")

    valid_entries = []
    resampled_count = 0

    for wav in wav_files:
        is_valid, result = validate_wav(wav, min_duration, max_duration)
        if not is_valid:
            print(f"  ⏭ {wav.name}: {result}")
            continue

        info = result
        dest_wav = output_stereo / wav.name

        # Resampler à 24kHz si nécessaire
        if info["sample_rate"] != 24000:
            if not dest_wav.exists():
                print(f"  >> {wav.name}: resample {info['sample_rate']} -> 24000 Hz")
                if not resample_wav(wav, dest_wav, 24000):
                    print(f"  ✗ {wav.name}: échec resample")
                    continue
                resampled_count += 1

            # Recalculer la durée après resample
            import wave
            with wave.open(str(dest_wav), 'r') as w:
                info["duration"] = w.getnframes() / w.getframerate()
        else:
            if not dest_wav.exists():
                shutil.copy2(wav, dest_wav)

        # Le path dans le JSONL est RELATIF au répertoire contenant le JSONL
        valid_entries.append({
            "path": f"data_stereo/{wav.name}",
            "duration": round(info["duration"], 6),
        })

    print(f"\n{len(valid_entries)} fichiers valides sur {len(wav_files)}")
    if resampled_count:
        print(f"  ({resampled_count} fichiers resampleés à 24kHz)")

    if not valid_entries:
        print("✗ Aucun fichier valide trouvé.")
        return

    # Split train/eval
    random.seed(seed)
    random.shuffle(valid_entries)
    n_eval = max(1, int(len(valid_entries) * eval_ratio))
    eval_entries = valid_entries[:n_eval]
    train_entries = valid_entries[n_eval:]

    # Écrire les JSONL au format moshi-finetune :
    # Chaque ligne : {"path": "data_stereo/X.wav", "duration": 24.52}
    for name, entries in [("train.jsonl", train_entries),
                          ("eval.jsonl", eval_entries)]:
        path = output_dir / name
        with open(path, 'w', encoding='utf-8') as f:
            for entry in entries:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')

    # Résumé
    total_hours = sum(e["duration"] for e in valid_entries) / 3600
    train_hours = sum(e["duration"] for e in train_entries) / 3600
    eval_hours = sum(e["duration"] for e in eval_entries) / 3600

    print(f"\n{'='*60}")
    print(f"Dataset préparé dans : {output_dir}")
    print(f"  Total : {len(valid_entries)} fichiers, {total_hours:.1f}h")
    print(f"  Train : {len(train_entries)} fichiers, {train_hours:.1f}h")
    print(f"  Eval  : {len(eval_entries)} fichiers, {eval_hours:.1f}h")
    print(f"{'='*60}")
    print(f"\nFormat JSONL (vérifié compatible moshi-finetune) :")
    print(f'  {json.dumps(valid_entries[0], ensure_ascii=False)}')
    print(f"\nProchaine étape :")
    print(f"  python scripts/03_transcribe.py")
    print(f"  (lance annotate.py avec --lang fr pour générer les .json)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Préparer le dataset moshi-finetune")
    parser.add_argument("--stereo-dir", default="data/stereo")
    parser.add_argument("--output-dir", default="data/moshi_dataset")
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--min-duration", type=float, default=10.0)
    parser.add_argument("--max-duration", type=float, default=300.0)
    args = parser.parse_args()

    prepare_dataset(
        args.stereo_dir, args.output_dir,
        args.eval_ratio, args.min_duration, args.max_duration,
    )
