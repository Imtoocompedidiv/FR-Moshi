"""
FR-Moshi — Etape 0 : Conversion SUMM-RE -> format moshi-finetune

Streams SUMM-RE from HuggingFace and converts to stereo 24kHz WAV.
Uses streaming mode to avoid downloading the full dataset to disk.

Key insight: samples from the same meeting are contiguous in SUMM-RE.
We buffer one meeting at a time, process when meeting_id changes,
then free memory immediately.

Canal gauche (0) = Moshi (locuteur principal)
Canal droit  (1) = Utilisateur (second locuteur)
"""

import json
import argparse
import sys
import gc
import numpy as np
from pathlib import Path


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
    return sum(seg['end'] - seg['start'] for seg in segments)


def process_meeting(speakers, output_dir, target_sr=24000,
                    min_duration=10.0, max_duration=300.0):
    """
    Process a meeting's speakers: select best 2, create stereo WAV.
    Returns list of JSONL entries.
    """
    import soundfile as sf

    if len(speakers) < 2:
        return []

    # Find 2 most active speakers
    speakers_with_dur = [
        (s, get_speaker_duration(s.get('segments', [])))
        for s in speakers
    ]
    speakers_with_dur.sort(key=lambda x: -x[1])
    s1, dur1 = speakers_with_dur[0]
    s2, dur2 = speakers_with_dur[1]

    if dur1 < 30 or dur2 < 30:
        return []

    meeting_id = s1['meeting_id']
    print(f"  [{meeting_id}] {s1['speaker_id']} ({dur1:.0f}s) + "
          f"{s2['speaker_id']} ({dur2:.0f}s)", flush=True)

    # Resample both channels to target SR
    s1_audio = resample_audio(s1['audio']['array'], s1['audio']['sampling_rate'], target_sr)
    s2_audio = resample_audio(s2['audio']['array'], s2['audio']['sampling_rate'], target_sr)

    # Align lengths
    max_len = max(len(s1_audio), len(s2_audio))
    if len(s1_audio) < max_len:
        s1_audio = np.pad(s1_audio, (0, max_len - len(s1_audio)))
    if len(s2_audio) < max_len:
        s2_audio = np.pad(s2_audio, (0, max_len - len(s2_audio)))

    duration = max_len / target_sr
    if duration < min_duration:
        return []
    if duration > max_duration:
        max_samples = int(max_duration * target_sr)
        s1_audio = s1_audio[:max_samples]
        s2_audio = s2_audio[:max_samples]
        duration = max_duration

    # Normalize
    for arr in [s1_audio, s2_audio]:
        peak = np.abs(arr).max()
        if peak > 0.95:
            arr *= 0.9 / peak

    # Write stereo WAV
    stereo = np.stack([s1_audio, s2_audio], axis=-1).astype(np.float32)
    filename = f"{meeting_id}_{s1['speaker_id']}_{s2['speaker_id']}.wav"
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
    """Convertit SUMM-RE en format moshi-finetune (streaming, 1 meeting at a time)."""
    from datasets import load_dataset

    output_dir = Path(output_dir)
    stereo_dir = output_dir / "data_stereo"
    stereo_dir.mkdir(parents=True, exist_ok=True)

    print(f"Chargement de SUMM-RE (split={split})...", flush=True)
    print(f"  Cible : {max_hours}h maximum", flush=True)
    print(f"  Output : {output_dir}", flush=True)
    print(f"  Mode : streaming (1 meeting buffered at a time)", flush=True)

    ds = load_dataset("linagora/SUMM-RE", split=split, streaming=True)

    all_entries = []
    total_hours = 0
    current_meeting = None
    meeting_buffer = []
    sample_count = 0

    for sample in ds:
        sample_count += 1
        mid = sample['meeting_id']

        # When meeting changes, process the previous meeting
        if mid != current_meeting and current_meeting is not None:
            entries = process_meeting(
                meeting_buffer, stereo_dir, target_sr, min_duration, max_duration
            )
            for entry in entries:
                all_entries.append(entry)
                total_hours += entry["duration"] / 3600
                print(f"    -> {entry['path']} ({entry['duration']:.0f}s) "
                      f"[total: {total_hours:.1f}h]", flush=True)

            # Free memory
            meeting_buffer = []
            gc.collect()

            if total_hours >= max_hours:
                print(f"\n  Limite de {max_hours}h atteinte.", flush=True)
                break

        current_meeting = mid
        meeting_buffer.append(sample)

        if sample_count % 10 == 0:
            print(f"  {sample_count} samples streamed, "
                  f"{len(all_entries)} files, {total_hours:.1f}h", flush=True)

    # Process the last meeting
    if meeting_buffer and total_hours < max_hours:
        entries = process_meeting(
            meeting_buffer, stereo_dir, target_sr, min_duration, max_duration
        )
        for entry in entries:
            all_entries.append(entry)
            total_hours += entry["duration"] / 3600
            print(f"    -> {entry['path']} ({entry['duration']:.0f}s) "
                  f"[total: {total_hours:.1f}h]", flush=True)

    if not all_entries:
        print("\nAucun fichier valide genere.", flush=True)
        return

    # Write JSONL
    jsonl_name = "train.jsonl" if split == "train" else "eval.jsonl"
    jsonl_path = output_dir / jsonl_name
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    print(f"\n{'='*60}", flush=True)
    print(f"Conversion terminee !", flush=True)
    print(f"  Fichiers : {len(all_entries)}", flush=True)
    print(f"  Duree    : {total_hours:.1f}h", flush=True)
    print(f"  JSONL    : {jsonl_path}", flush=True)
    print(f"  Stereo   : {stereo_dir}", flush=True)
    print(f"{'='*60}", flush=True)


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
