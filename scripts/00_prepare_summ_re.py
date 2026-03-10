"""
FR-Moshi — Etape 0 : Conversion SUMM-RE -> format moshi-finetune

Downloads SUMM-RE (Linagora, CC BY-SA 4.0) from HuggingFace and converts
to stereo 24kHz WAV + JSONL manifest for moshi-finetune.

Uses non-streaming download with HF cache for faster parallel downloads.
Single-pass: samples from the same meeting are contiguous in the dataset,
so we buffer per-meeting and process as soon as both speakers are available.

Canal gauche (0) = Moshi (locuteur principal)
Canal droit  (1) = Utilisateur (second locuteur)

Usage:
  python scripts/00_prepare_summ_re.py --max-hours 25
  python scripts/00_prepare_summ_re.py --split dev --max-hours 3
"""

import json
import argparse
import sys
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
    """Calcule la duree totale de parole d'un locuteur."""
    total = 0
    for seg in segments:
        total += seg['end'] - seg['start']
    return total


def process_meeting_pair(speaker1_data, speaker2_data, output_dir, meeting_id,
                         target_sr=24000, min_duration=10.0, max_duration=300.0):
    """Combine 2 pistes mono en un fichier stereo 24kHz."""
    import soundfile as sf

    s1_audio = speaker1_data['audio']['array']
    s1_sr = speaker1_data['audio']['sampling_rate']
    s2_audio = speaker2_data['audio']['array']
    s2_sr = speaker2_data['audio']['sampling_rate']

    s1_resampled = resample_audio(s1_audio, s1_sr, target_sr)
    s2_resampled = resample_audio(s2_audio, s2_sr, target_sr)

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

    for arr in [s1_resampled, s2_resampled]:
        peak = np.abs(arr).max()
        if peak > 0.95:
            arr *= 0.9 / peak

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


def select_best_pair(speakers):
    """Select the 2 speakers with the most speech in a meeting."""
    speakers_with_dur = [
        (s, get_speaker_duration(s.get('segments', []))) for s in speakers
    ]
    speakers_with_dur.sort(key=lambda x: -x[1])

    if len(speakers_with_dur) < 2:
        return None, None
    s1, dur1 = speakers_with_dur[0]
    s2, dur2 = speakers_with_dur[1]
    if dur1 < 30 or dur2 < 30:
        return None, None
    return s1, s2


def convert_summ_re(
    split="train",
    output_dir="data/moshi_dataset",
    max_hours=30.0,
    target_sr=24000,
    min_duration=10.0,
    max_duration=300.0,
):
    """Convertit SUMM-RE en format moshi-finetune."""
    import os
    from datasets import load_dataset

    output_dir = Path(output_dir)
    stereo_dir = output_dir / "data_stereo"
    stereo_dir.mkdir(parents=True, exist_ok=True)

    # Use network volume for HF cache if available
    if os.path.isdir("/runpod-volume"):
        cache_dir = "/runpod-volume/.hf_cache"
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = cache_dir
        print(f"Using cache: {cache_dir}")

    print(f"Chargement de SUMM-RE (split={split})...")
    print(f"  Cible : {max_hours}h maximum")
    print(f"  Output : {output_dir}")
    sys.stdout.flush()

    # Non-streaming: downloads in parallel, caches to disk
    print("Downloading dataset (parallel, cached)...")
    sys.stdout.flush()
    ds = load_dataset("linagora/SUMM-RE", split=split)
    print(f"  {len(ds)} samples loaded")
    sys.stdout.flush()

    # Single pass: group by meeting, process each meeting
    all_entries = []
    total_hours = 0
    current_meeting = None
    meeting_speakers = []

    def process_meeting(meeting_id, speakers):
        """Process a complete meeting's speakers."""
        nonlocal total_hours
        if total_hours >= max_hours:
            return

        s1, s2 = select_best_pair(speakers)
        if s1 is None:
            return

        dur1 = get_speaker_duration(s1.get('segments', []))
        dur2 = get_speaker_duration(s2.get('segments', []))
        print(f"  [{meeting_id}] {s1['speaker_id']} ({dur1:.0f}s) + "
              f"{s2['speaker_id']} ({dur2:.0f}s)")
        sys.stdout.flush()

        entries = process_meeting_pair(
            s1, s2, stereo_dir, meeting_id,
            target_sr, min_duration, max_duration,
        )
        for entry in entries:
            all_entries.append(entry)
            total_hours += entry["duration"] / 3600
            print(f"    -> {entry['path']} ({entry['duration']:.0f}s) "
                  f"[total: {total_hours:.1f}h]")
            sys.stdout.flush()

    for i, sample in enumerate(ds):
        if total_hours >= max_hours:
            print(f"\n  Limite de {max_hours}h atteinte.")
            break

        mid = sample['meeting_id']

        if mid != current_meeting and current_meeting is not None:
            # New meeting started, process the previous one
            process_meeting(current_meeting, meeting_speakers)
            meeting_speakers = []

        current_meeting = mid
        meeting_speakers.append(sample)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(ds)} samples, {len(all_entries)} files, {total_hours:.1f}h")
            sys.stdout.flush()

    # Process the last meeting
    if meeting_speakers and total_hours < max_hours:
        process_meeting(current_meeting, meeting_speakers)

    if not all_entries:
        print("\nAucun fichier valide genere.")
        return

    # Write JSONL
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
    sys.stdout.flush()


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
