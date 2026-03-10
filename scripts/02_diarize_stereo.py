"""
FR-Moshi — Étape 2 : Diarisation et conversion stéréo
Sépare les locuteurs et crée des fichiers stéréo (format Moshi).

Canal gauche = Moshi (locuteur sélectionné)
Canal droit  = Utilisateur (autre locuteur)

Prérequis:
  pip install pyannote.audio soundfile numpy torch
  Token HuggingFace avec accès à pyannote/speaker-diarization-3.1
"""

import json
import argparse
import numpy as np
from pathlib import Path

# Essayer d'importer les dépendances GPU
try:
    import soundfile as sf
    SOUNDFILE_OK = True
except ImportError:
    SOUNDFILE_OK = False
    print("⚠ soundfile non installé: pip install soundfile")

try:
    from pyannote.audio import Pipeline
    PYANNOTE_OK = True
except ImportError:
    PYANNOTE_OK = False
    print("⚠ pyannote non installé: pip install pyannote.audio")


def diarize_to_stereo(
    input_path: str,
    output_dir: Path,
    pipeline,
    target_sr: int = 24000,
    min_duration: float = 30.0,
):
    """
    Diarise un fichier audio mono et crée un fichier stéréo.

    Args:
        input_path: Chemin vers le fichier audio mono
        output_dir: Répertoire de sortie
        pipeline: Pipeline pyannote de diarisation
        target_sr: Sample rate cible (24000 pour Mimi)
        min_duration: Durée minimum en secondes
    """
    input_path = Path(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Charger l'audio
    audio, sr = sf.read(str(input_path))
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)  # Convertir en mono si nécessaire

    duration = len(audio) / sr
    if duration < min_duration:
        print(f"  ⏭ {input_path.name}: trop court ({duration:.0f}s < {min_duration:.0f}s)")
        return None

    # Resampler à 24kHz si nécessaire
    if sr != target_sr:
        import torchaudio
        import torch
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio_tensor = resampler(audio_tensor)
        audio = audio_tensor.squeeze(0).numpy()
        sr = target_sr

    # Diarisation
    print(f"  🔊 Diarisation de {input_path.name} ({duration:.0f}s)...")
    diarization = pipeline(str(input_path))

    # Identifier les 2 locuteurs principaux
    speaker_durations = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        dur = turn.end - turn.start
        speaker_durations[speaker] = speaker_durations.get(speaker, 0) + dur

    if len(speaker_durations) < 2:
        print(f"  ⏭ {input_path.name}: moins de 2 locuteurs détectés")
        return None

    # Trier par durée de parole
    sorted_speakers = sorted(speaker_durations.items(), key=lambda x: -x[1])
    moshi_speaker = sorted_speakers[0][0]   # Plus gros locuteur = Moshi
    user_speaker = sorted_speakers[1][0]    # Deuxième = Utilisateur

    print(f"    Moshi ({moshi_speaker}): {speaker_durations[moshi_speaker]:.0f}s")
    print(f"    User  ({user_speaker}): {speaker_durations[user_speaker]:.0f}s")

    # Créer les canaux stéréo
    left_channel = np.zeros_like(audio)   # Moshi
    right_channel = np.zeros_like(audio)  # Utilisateur

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_sample = int(turn.start * sr)
        end_sample = min(int(turn.end * sr), len(audio))

        if speaker == moshi_speaker:
            left_channel[start_sample:end_sample] = audio[start_sample:end_sample]
        elif speaker == user_speaker:
            right_channel[start_sample:end_sample] = audio[start_sample:end_sample]
        # Ignorer les autres locuteurs

    # Écrire le fichier stéréo
    stereo = np.stack([left_channel, right_channel], axis=-1)
    output_path = output_dir / input_path.with_suffix('.wav').name
    sf.write(str(output_path), stereo, sr)

    # Métadonnées
    meta = {
        "source": str(input_path),
        "output": str(output_path),
        "duration": duration,
        "sample_rate": sr,
        "moshi_speaker": moshi_speaker,
        "moshi_duration": speaker_durations[moshi_speaker],
        "user_speaker": user_speaker,
        "user_duration": speaker_durations[user_speaker],
        "num_speakers_detected": len(speaker_durations),
    }

    print(f"  ✓ {output_path.name} ({duration:.0f}s)")
    return meta


def process_batch(
    manifest_path: str,
    output_dir: str = "data/stereo",
    hf_token: str = None,
    max_files: int = None,
):
    """Traite un lot de fichiers audio."""
    if not PYANNOTE_OK or not SOUNDFILE_OK:
        print("✗ Dépendances manquantes. Installer avec:")
        print("  pip install pyannote.audio soundfile torch torchaudio")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Charger le pipeline de diarisation
    print("Chargement du modèle pyannote...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    # GPU si disponible
    import torch
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        print("  → GPU détecté, utilisation CUDA")
    else:
        print("  → Pas de GPU, utilisation CPU (plus lent)")

    # Charger le manifeste
    entries = []
    with open(manifest_path, 'r') as f:
        for line in f:
            entries.append(json.loads(line.strip()))

    if max_files:
        entries = entries[:max_files]

    print(f"\nTraitement de {len(entries)} fichiers...")

    # Traiter chaque fichier
    results = []
    for i, entry in enumerate(entries):
        print(f"\n[{i+1}/{len(entries)}] {entry.get('filename', entry['path'])}")
        meta = diarize_to_stereo(
            entry["path"],
            output_dir,
            pipeline,
        )
        if meta:
            results.append(meta)

    # Sauvegarder les métadonnées
    meta_path = output_dir / "diarization_results.jsonl"
    with open(meta_path, 'w', encoding='utf-8') as f:
        for r in results:
            json.dump(r, f, ensure_ascii=False)
            f.write('\n')

    total_hours = sum(r["duration"] for r in results) / 3600
    print(f"\n{'='*60}")
    print(f"Diarisation terminée:")
    print(f"  {len(results)}/{len(entries)} fichiers traités")
    print(f"  {total_hours:.1f} heures de stéréo généré")
    print(f"  Résultats: {meta_path}")
    print(f"{'='*60}")
    print(f"\nProchaine étape: python scripts/04_prepare_dataset.py --stereo-dir {output_dir}")
    print(f"  (puis: python scripts/03_transcribe.py pour annotation via annotate.py)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diarisation → stéréo")
    parser.add_argument("manifest", help="Chemin vers le manifeste JSONL")
    parser.add_argument("--output-dir", default="data/stereo", help="Répertoire de sortie")
    parser.add_argument("--hf-token", default=None, help="Token HuggingFace")
    parser.add_argument("--max-files", type=int, default=None, help="Limiter le nombre de fichiers")
    args = parser.parse_args()

    process_batch(args.manifest, args.output_dir, args.hf_token, args.max_files)
