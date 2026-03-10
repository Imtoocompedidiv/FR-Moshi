"""
FR-Moshi — Etape 0 : Conversion SUMM-RE -> format moshi-finetune

Telecharge le dataset SUMM-RE (Linagora, CC BY-SA 4.0) depuis HuggingFace
et le convertit en fichiers stereo 24kHz WAV + JSONL manifest.

SUMM-RE contient des reunions francaises avec micro separe par locuteur.
On prend les 2 locuteurs les plus actifs par reunion -> stereo.

Canal gauche (0) = Moshi (locuteur principal)
Canal droit  (1) = Utilisateur (second locuteur)

Usage:
  python scripts/00_prepare_summ_re.py --max-hours 30
  python scripts/00_prepare_summ_re.py --split dev --max-hours 10  # Pour eval

Prereqs:
  pip install datasets soundfile numpy torchaudio
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


def resample_audio(audio_array, orig_sr, target_sr=24000):
    """Resample audio numpy array to target sample rate."""
    if orig_sr == target_sr:
        return audio_array
    try:
        import torchaudio
        import torch
        tensor = torch.from_numpy(audio_array).float().unsqueeze(0)
        resampled = torchaudio.functional.resample(tensor, orig_sr, target_sr)
        return resampled.squeeze(0).numpy()
    except ImportError:
        # Fallback: simple decimation (less quality but no dependency)
        ratio = orig_sr / target_sr
        n_out = int(len(audio_array) / ratio)
        indices = (np.arange(n_out) * ratio).astype(int)
        indices = np.clip(indices, 0, len(audio_array) - 1)
        return audio_array[indices]


def clean_transcript_word(word):
    """Nettoie un mot des conventions SUMM-RE (SPPAS format)."""
    if not word or word in ('*', '@', '+', '#'):
        return None
    # Retirer les annotations entre crochets/accolades
    if word.startswith('{') or word.startswith('['):
        return None
    # Retirer les parentheses d'elision
    word = word.replace('(', '').replace(')', '')
    # Garder les mots tronques (ex-) tels quels
    return word.strip() if word.strip() else None


def process_meeting_pair(speaker1_data, speaker2_data, output_dir, meeting_id,
                         target_sr=24000, min_duration=10.0, max_duration=300.0):
    """
    Combine 2 pistes mono en un fichier stereo 24kHz.
    Retourne une liste d'entries JSONL ou [] si invalide.
    """
    import soundfile as sf

    s1_audio = speaker1_data['audio']['array']
    s1_sr = speaker1_data['audio']['sampling_rate']
    s2_audio = speaker2_data['audio']['array']
    s2_sr = speaker2_data['audio']['sampling_rate']

    # Resample to target SR
    s1_resampled = resample_audio(s1_audio, s1_sr, target_sr)
    s2_resampled = resample_audio(s2_audio, s2_sr, target_sr)

    # Aligner les longueurs (padding avec des zeros)
    max_len = max(len(s1_resampled), len(s2_resampled))
    if len(s1_resampled) < max_len:
        s1_resampled = np.pad(s1_resampled, (0, max_len - len(s1_resampled)))
    if len(s2_resampled) < max_len:
        s2_resampled = np.pad(s2_resampled, (0, max_len - len(s2_resampled)))

    duration = max_len / target_sr

    if duration < min_duration:
        return []
    if duration > max_duration:
        # Tronquer a max_duration
        max_samples = int(max_duration * target_sr)
        s1_resampled = s1_resampled[:max_samples]
        s2_resampled = s2_resampled[:max_samples]
        duration = max_duration

    # Normaliser les niveaux (eviter le clipping)
    for arr in [s1_resampled, s2_resampled]:
        peak = np.abs(arr).max()
        if peak > 0.95:
            arr *= 0.9 / peak

    # Creer le fichier stereo
    stereo = np.stack([s1_resampled, s2_resampled], axis=-1).astype(np.float32)
    s1_id = speaker1_data['speaker_id']
    s2_id = speaker2_data['speaker_id']
    filename = f"{meeting_id}_{s1_id}_{s2_id}.wav"
    wav_path = output_dir / filename

    sf.write(str(wav_path), stereo, target_sr)

    return [{
        "path": f"data_stereo/{filename}",
        "duration": round(duration, 6),
    }]


def get_speaker_duration(speaker_data):
    """Calcule la duree totale de parole d'un locuteur."""
    total = 0
    for seg in speaker_data.get('segments', []):
        total += seg['end'] - seg['start']
    return total


def convert_summ_re(
    split="train",
    output_dir="data/moshi_dataset",
    max_hours=30.0,
    target_sr=24000,
    min_duration=10.0,
    max_duration=300.0,
):
    """Convertit SUMM-RE en format moshi-finetune."""
    from datasets import load_dataset

    output_dir = Path(output_dir)
    stereo_dir = output_dir / "data_stereo"
    stereo_dir.mkdir(parents=True, exist_ok=True)

    print(f"Chargement de SUMM-RE (split={split})...")
    print(f"  Cible : {max_hours}h maximum")
    print(f"  Output : {output_dir}")

    # Charger en streaming pour eviter de telecharger tout
    ds = load_dataset("linagora/SUMM-RE", split=split, streaming=True)

    # Grouper par meeting_id
    meetings = defaultdict(list)
    print("Indexation des locuteurs par reunion...")

    for sample in ds:
        mid = sample['meeting_id']
        meetings[mid].append(sample)

        # Log de progression
        if len(meetings) % 10 == 0 and len(meetings[mid]) == 1:
            total_speakers = sum(len(v) for v in meetings.values())
            print(f"  {len(meetings)} reunions, {total_speakers} pistes...")

    print(f"  Total : {len(meetings)} reunions")

    # Traiter chaque reunion
    all_entries = []
    total_hours = 0

    for meeting_id, speakers in sorted(meetings.items()):
        if total_hours >= max_hours:
            print(f"\n  Limite de {max_hours}h atteinte.")
            break

        if len(speakers) < 2:
            continue

        # Trouver les 2 locuteurs les plus actifs
        speakers_with_duration = [
            (s, get_speaker_duration(s)) for s in speakers
        ]
        speakers_with_duration.sort(key=lambda x: -x[1])
        speaker1, dur1 = speakers_with_duration[0]
        speaker2, dur2 = speakers_with_duration[1]

        # Verifier qu'ils parlent suffisamment
        if dur1 < 30 or dur2 < 30:
            continue

        print(f"\n[{meeting_id}] {speaker1['speaker_id']} ({dur1:.0f}s) + "
              f"{speaker2['speaker_id']} ({dur2:.0f}s)")

        entries = process_meeting_pair(
            speaker1, speaker2, stereo_dir, meeting_id,
            target_sr, min_duration, max_duration,
        )

        for entry in entries:
            all_entries.append(entry)
            total_hours += entry["duration"] / 3600
            print(f"  -> {entry['path']} ({entry['duration']:.0f}s) "
                  f"[total: {total_hours:.1f}h]")

    if not all_entries:
        print("\nAucun fichier valide genere.")
        return

    # Ecrire le JSONL
    jsonl_name = "train.jsonl" if split == "train" else "eval.jsonl"
    jsonl_path = output_dir / jsonl_name
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    print(f"\n{'='*60}")
    print(f"Conversion terminee !")
    print(f"  Fichiers : {len(all_entries)}")
    print(f"  Duree    : {total_hours:.1f}h")
    print(f"  JSONL    : {jsonl_path}")
    print(f"  Stereo   : {stereo_dir}")
    print(f"{'='*60}")
    print(f"\nProchaine etape :")
    print(f"  cd moshi-finetune")
    print(f"  python annotate.py ../{jsonl_path} --lang fr --local")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convertir SUMM-RE en format moshi-finetune"
    )
    parser.add_argument("--split", default="train",
                        help="Split HuggingFace (train/dev/test)")
    parser.add_argument("--output-dir", default="data/moshi_dataset",
                        help="Repertoire de sortie")
    parser.add_argument("--max-hours", type=float, default=30.0,
                        help="Heures maximum a traiter")
    parser.add_argument("--min-duration", type=float, default=10.0,
                        help="Duree minimum par fichier (secondes)")
    parser.add_argument("--max-duration", type=float, default=300.0,
                        help="Duree maximum par fichier (secondes)")
    args = parser.parse_args()

    convert_summ_re(
        split=args.split,
        output_dir=args.output_dir,
        max_hours=args.max_hours,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )
