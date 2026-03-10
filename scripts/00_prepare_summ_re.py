"""
FR-Moshi — Etape 0 : Conversion SUMM-RE -> format moshi-finetune

Telecharge le dataset SUMM-RE (Linagora, CC BY-SA 4.0) depuis HuggingFace
et le convertit en fichiers stereo 24kHz WAV + JSONL manifest.

SUMM-RE contient des reunions francaises avec micro separe par locuteur.
On prend les 2 locuteurs les plus actifs par reunion -> stereo.

Canal gauche (0) = Moshi (locuteur principal)
Canal droit  (1) = Utilisateur (second locuteur)

Two-pass approach to avoid loading all audio into memory:
  Pass 1: Stream metadata only (no audio) to select speaker pairs
  Pass 2: Stream again, only decode audio for selected pairs

Usage:
  python scripts/00_prepare_summ_re.py --max-hours 30
  python scripts/00_prepare_summ_re.py --split dev --max-hours 3  # Pour eval

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
        ratio = orig_sr / target_sr
        n_out = int(len(audio_array) / ratio)
        indices = (np.arange(n_out) * ratio).astype(int)
        indices = np.clip(indices, 0, len(audio_array) - 1)
        return audio_array[indices]


def get_speaker_duration(segments):
    """Calcule la duree totale de parole d'un locuteur a partir des segments."""
    total = 0
    for seg in segments:
        total += seg['end'] - seg['start']
    return total


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


def convert_summ_re(
    split="train",
    output_dir="data/moshi_dataset",
    max_hours=30.0,
    target_sr=24000,
    min_duration=10.0,
    max_duration=300.0,
):
    """Convertit SUMM-RE en format moshi-finetune (2-pass approach)."""
    from datasets import load_dataset

    output_dir = Path(output_dir)
    stereo_dir = output_dir / "data_stereo"
    stereo_dir.mkdir(parents=True, exist_ok=True)

    print(f"Chargement de SUMM-RE (split={split})...")
    print(f"  Cible : {max_hours}h maximum")
    print(f"  Output : {output_dir}")

    # ========================================
    # PASS 1: Metadata only — select best speaker pairs
    # No audio decoding, very fast and memory-efficient
    # ========================================
    print("\n--- Pass 1: Indexation des metadonnees (sans audio) ---")

    ds_meta = load_dataset(
        "linagora/SUMM-RE", split=split, streaming=True
    ).remove_columns(["audio"])

    meetings_meta = defaultdict(list)
    sample_count = 0

    for sample in ds_meta:
        mid = sample['meeting_id']
        meetings_meta[mid].append({
            'meeting_id': mid,
            'speaker_id': sample['speaker_id'],
            'segments': sample.get('segments', []),
        })
        sample_count += 1

        if sample_count % 50 == 0:
            print(f"  {sample_count} pistes, {len(meetings_meta)} reunions...")

    print(f"  Total : {sample_count} pistes, {len(meetings_meta)} reunions")

    # Select best 2 speakers per meeting
    selected_pairs = {}  # meeting_id -> (speaker1_id, speaker2_id)

    for mid, speakers in sorted(meetings_meta.items()):
        if len(speakers) < 2:
            continue

        speakers_with_dur = [
            (s, get_speaker_duration(s['segments'])) for s in speakers
        ]
        speakers_with_dur.sort(key=lambda x: -x[1])
        s1, dur1 = speakers_with_dur[0]
        s2, dur2 = speakers_with_dur[1]

        if dur1 < 30 or dur2 < 30:
            continue

        selected_pairs[mid] = (s1['speaker_id'], s2['speaker_id'])
        print(f"  [{mid}] {s1['speaker_id']} ({dur1:.0f}s) + {s2['speaker_id']} ({dur2:.0f}s)")

    print(f"\n  {len(selected_pairs)} reunions selectionnees")

    if not selected_pairs:
        print("Aucune reunion valide trouvee.")
        return

    # ========================================
    # PASS 2: Stream with audio — process only selected pairs
    # Only 2 audio tracks per meeting are kept in memory at a time
    # ========================================
    print("\n--- Pass 2: Conversion audio (paires selectionnees uniquement) ---")

    ds_audio = load_dataset("linagora/SUMM-RE", split=split, streaming=True)

    # Buffer: meeting_id -> {speaker_id: sample_data}
    meeting_buffer = defaultdict(dict)
    all_entries = []
    total_hours = 0
    processed_meetings = set()
    sample_count = 0

    for sample in ds_audio:
        sample_count += 1
        mid = sample['meeting_id']
        sid = sample['speaker_id']

        if mid in processed_meetings:
            continue

        if total_hours >= max_hours:
            print(f"\n  Limite de {max_hours}h atteinte.")
            break

        if mid not in selected_pairs:
            continue

        s1_id, s2_id = selected_pairs[mid]
        if sid not in (s1_id, s2_id):
            continue

        # Store this speaker's data
        meeting_buffer[mid][sid] = sample

        if sample_count % 50 == 0:
            print(f"  {sample_count} pistes lues, {len(all_entries)} fichiers generes, {total_hours:.1f}h...")

        # Check if both speakers are now available
        if s1_id in meeting_buffer[mid] and s2_id in meeting_buffer[mid]:
            speaker1 = meeting_buffer[mid][s1_id]
            speaker2 = meeting_buffer[mid][s2_id]

            entries = process_meeting_pair(
                speaker1, speaker2, stereo_dir, mid,
                target_sr, min_duration, max_duration,
            )

            for entry in entries:
                all_entries.append(entry)
                total_hours += entry["duration"] / 3600
                print(f"  -> {entry['path']} ({entry['duration']:.0f}s) "
                      f"[total: {total_hours:.1f}h]")

            # Free memory
            del meeting_buffer[mid]
            processed_meetings.add(mid)

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
