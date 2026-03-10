"""
FR-Moshi — Étape 0 : Conversion ESLO -> format moshi-finetune
Dialogue-quality extraction from sociolinguistic interviews.

ESLO (Enquêtes Sociolinguistiques à Orléans) — 700h of real French conversations.
Dataset is pre-segmented with speaker labels and timestamps.

Strategy:
1. Group segments by conversation (file field)
2. Identify 2 main speakers per conversation
3. Reconstruct full conversation audio by placing segments on timeline
4. Extract dialogue-quality windows (both speakers active, good turn-taking)
5. Output as stereo WAV: L=Moshi (speaker 1), R=User (speaker 2)

Canal gauche (0) = Moshi (locuteur principal)
Canal droit  (1) = Utilisateur (second locuteur)
"""

import json
import argparse
import sys
import gc
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class DialogueSegment:
    """A segment of active dialogue between 2 speakers."""
    start_sec: float
    end_sec: float
    s1_speech_sec: float
    s2_speech_sec: float
    n_turns: int
    silence_ratio: float

    @property
    def duration(self):
        return self.end_sec - self.start_sec

    @property
    def speech_ratio(self):
        if self.duration <= 0:
            return 0
        return (self.s1_speech_sec + self.s2_speech_sec) / self.duration

    @property
    def balance(self):
        total = self.s1_speech_sec + self.s2_speech_sec
        if total <= 0:
            return 0
        minority = min(self.s1_speech_sec, self.s2_speech_sec)
        return minority / (total / 2)

    @property
    def quality_score(self):
        turn_score = min(1.0, self.n_turns / 6)
        return (self.speech_ratio * 0.3 +
                self.balance * 0.3 +
                turn_score * 0.3 +
                (1 - self.silence_ratio) * 0.1)


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


def find_dialogue_windows(s1_timeline, s2_timeline, resolution=0.1,
                          min_seg=15.0, max_seg=120.0, max_gap=3.0,
                          min_speech_ratio=0.3, min_quality=0.3):
    """Find windows of active dialogue between 2 speakers."""
    n_bins = len(s1_timeline)
    gap_bins = int(max_gap / resolution)

    either_speaking = s1_timeline | s2_timeline

    segments = []
    in_segment = False
    seg_start = 0
    gap_count = 0

    for i in range(n_bins):
        if either_speaking[i]:
            if not in_segment:
                seg_start = i
                in_segment = True
            gap_count = 0
        else:
            if in_segment:
                gap_count += 1
                if gap_count > gap_bins:
                    seg_end = i - gap_count
                    seg_dur = (seg_end - seg_start) * resolution
                    if seg_dur >= min_seg:
                        segments.append((seg_start, seg_end))
                    in_segment = False
                    gap_count = 0

    if in_segment:
        seg_end = n_bins
        seg_dur = (seg_end - seg_start) * resolution
        if seg_dur >= min_seg:
            segments.append((seg_start, seg_end))

    # Split long segments
    final_segments = []
    for start, end in segments:
        seg_dur = (end - start) * resolution
        if seg_dur <= max_seg:
            final_segments.append((start, end))
        else:
            pos = start
            while pos < end:
                chunk_end = min(pos + int(max_seg / resolution), end)
                if chunk_end < end:
                    search_start = max(pos + int(min_seg / resolution),
                                       chunk_end - int(10 / resolution))
                    best_cut = chunk_end
                    best_silence = 0
                    for j in range(search_start, chunk_end):
                        if not s1_timeline[j] and not s2_timeline[j]:
                            silence = 0
                            while (j + silence < chunk_end and
                                   not s1_timeline[j + silence] and
                                   not s2_timeline[j + silence]):
                                silence += 1
                            if silence > best_silence:
                                best_silence = silence
                                best_cut = j
                    chunk_end = best_cut
                chunk_dur = (chunk_end - pos) * resolution
                if chunk_dur >= min_seg:
                    final_segments.append((pos, chunk_end))
                pos = chunk_end

    # Score each segment
    dialogue_segments = []
    for start, end in final_segments:
        s1_speech = np.sum(s1_timeline[start:end]) * resolution
        s2_speech = np.sum(s2_timeline[start:end]) * resolution
        duration = (end - start) * resolution
        silence = duration - s1_speech - s2_speech

        n_turns = 0
        last_speaker = None
        for i in range(start, end):
            if s1_timeline[i] and not s2_timeline[i]:
                if last_speaker != 1:
                    n_turns += 1
                    last_speaker = 1
            elif s2_timeline[i] and not s1_timeline[i]:
                if last_speaker != 2:
                    n_turns += 1
                    last_speaker = 2

        seg = DialogueSegment(
            start_sec=start * resolution,
            end_sec=end * resolution,
            s1_speech_sec=s1_speech,
            s2_speech_sec=s2_speech,
            n_turns=n_turns,
            silence_ratio=max(0, silence / duration) if duration > 0 else 1,
        )

        if seg.speech_ratio >= min_speech_ratio and seg.quality_score >= min_quality:
            dialogue_segments.append(seg)

    return dialogue_segments


def process_conversation(file_id, segments_by_speaker, output_dir, target_sr=24000,
                         min_seg=15.0, max_seg=120.0, min_quality=0.3):
    """
    Process one conversation from ESLO.
    segments_by_speaker: dict of speaker_id -> list of (start, end, audio_array, sr)
    Returns list of JSONL entries.
    """
    import soundfile as sf

    if len(segments_by_speaker) < 2:
        return []

    # Compute total speech per speaker
    speaker_stats = {}
    for spk, segs in segments_by_speaker.items():
        total_speech = sum(end - start for start, end, _, _ in segs)
        speaker_stats[spk] = total_speech

    # Sort by speech duration, pick top 2
    sorted_speakers = sorted(speaker_stats.items(), key=lambda x: -x[1])
    s1_id = sorted_speakers[0][0]
    s2_id = sorted_speakers[1][0]
    s1_speech = sorted_speakers[0][1]
    s2_speech = sorted_speakers[1][1]

    # Both speakers need meaningful speech (>20s each)
    if s1_speech < 20 or s2_speech < 20:
        return []

    # Find total conversation duration
    all_ends = []
    for spk, segs in segments_by_speaker.items():
        for start, end, _, _ in segs:
            all_ends.append(end)
    total_dur = max(all_ends) if all_ends else 0

    if total_dur < min_seg:
        return []

    # Build speech timelines at 100ms resolution
    resolution = 0.1
    n_bins = int(total_dur / resolution) + 1

    s1_timeline = np.zeros(n_bins, dtype=bool)
    s2_timeline = np.zeros(n_bins, dtype=bool)

    for start, end, _, _ in segments_by_speaker[s1_id]:
        start_bin = max(0, int(start / resolution))
        end_bin = min(n_bins, int(end / resolution))
        s1_timeline[start_bin:end_bin] = True

    for start, end, _, _ in segments_by_speaker[s2_id]:
        start_bin = max(0, int(start / resolution))
        end_bin = min(n_bins, int(end / resolution))
        s2_timeline[start_bin:end_bin] = True

    # Find dialogue windows
    windows = find_dialogue_windows(
        s1_timeline, s2_timeline,
        resolution=resolution,
        min_seg=min_seg,
        max_seg=max_seg,
        min_quality=min_quality,
    )

    if not windows:
        return []

    # Build full audio channels by placing segments on timeline
    total_samples = int(total_dur * target_sr) + target_sr  # +1s buffer
    s1_audio = np.zeros(total_samples, dtype=np.float32)
    s2_audio = np.zeros(total_samples, dtype=np.float32)

    for start, end, audio_arr, sr in segments_by_speaker[s1_id]:
        resampled = resample_audio(audio_arr, sr, target_sr).astype(np.float32)
        start_sample = int(start * target_sr)
        end_sample = start_sample + len(resampled)
        end_sample = min(end_sample, total_samples)
        n_copy = end_sample - start_sample
        if n_copy > 0:
            s1_audio[start_sample:end_sample] = resampled[:n_copy]

    for start, end, audio_arr, sr in segments_by_speaker[s2_id]:
        resampled = resample_audio(audio_arr, sr, target_sr).astype(np.float32)
        start_sample = int(start * target_sr)
        end_sample = start_sample + len(resampled)
        end_sample = min(end_sample, total_samples)
        n_copy = end_sample - start_sample
        if n_copy > 0:
            s2_audio[start_sample:end_sample] = resampled[:n_copy]

    # Extract dialogue segments as stereo WAVs
    entries = []
    for idx, win in enumerate(windows):
        start_sample = int(win.start_sec * target_sr)
        end_sample = int(win.end_sec * target_sr)
        end_sample = min(end_sample, total_samples)

        s1_chunk = s1_audio[start_sample:end_sample].copy()
        s2_chunk = s2_audio[start_sample:end_sample].copy()

        # Normalize each channel
        for arr in [s1_chunk, s2_chunk]:
            peak = np.abs(arr).max()
            if peak > 0.01:
                arr *= 0.9 / max(peak, 0.95)

        stereo = np.stack([s1_chunk, s2_chunk], axis=-1).astype(np.float32)

        # Clean filename
        safe_file_id = file_id.replace("/", "_").replace("\\", "_")
        filename = f"eslo_{safe_file_id}_{s1_id}_{s2_id}_{idx:02d}.wav"
        wav_path = output_dir / filename
        sf.write(str(wav_path), stereo, target_sr)

        duration = win.duration
        entries.append({
            "path": f"data_stereo/{filename}",
            "duration": round(duration, 6),
        })

        print(f"    -> {filename} ({duration:.0f}s, "
              f"q={win.quality_score:.2f}, "
              f"turns={win.n_turns}, "
              f"speech={win.speech_ratio:.0%}, "
              f"bal={win.balance:.0%})", flush=True)

    return entries


def convert_eslo(
    output_dir="data/eslo_dataset",
    max_hours=999.0,
    target_sr=24000,
    min_seg=15.0,
    max_seg=120.0,
    min_quality=0.3,
    max_conversations=None,
):
    """Convert ESLO to moshi-finetune format with dialogue quality extraction."""
    from datasets import load_dataset

    output_dir = Path(output_dir)
    stereo_dir = output_dir / "data_stereo"
    stereo_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== ESLO -> moshi-finetune (dialogue extraction) ===", flush=True)
    print(f"  Max hours       : {max_hours}", flush=True)
    print(f"  Segments        : {min_seg}-{max_seg}s", flush=True)
    print(f"  Min quality     : {min_quality}", flush=True)
    print(f"  Output          : {output_dir}", flush=True)
    print(flush=True)

    print("Loading ESLO dataset (streaming)...", flush=True)
    ds = load_dataset("illuin/ESLO", streaming=True, split="train")

    all_entries = []
    total_hours = 0
    current_file = None
    conv_buffer = defaultdict(list)  # speaker -> list of (start, end, audio, sr)
    conv_count = 0
    skipped_conv = 0
    sample_count = 0

    for sample in ds:
        sample_count += 1

        file_id = sample.get("file", sample.get("file_id", f"unknown_{sample_count}"))
        speaker = sample.get("speaker", "unknown")
        start_ts = sample.get("start_timestamp", sample.get("start", 0))
        end_ts = sample.get("end_timestamp", sample.get("end", 0))

        # Get audio
        audio_data = sample.get("audio", {})
        if isinstance(audio_data, dict):
            audio_arr = np.array(audio_data.get("array", []), dtype=np.float32)
            sr = audio_data.get("sampling_rate", 16000)
        else:
            continue

        if len(audio_arr) == 0:
            continue

        # Normalize int16 to float if needed
        if audio_arr.dtype == np.int16 or np.abs(audio_arr).max() > 2.0:
            audio_arr = audio_arr.astype(np.float32) / 32768.0

        # New conversation?
        if file_id != current_file and current_file is not None:
            conv_count += 1
            n_speakers = len(conv_buffer)
            total_segs = sum(len(v) for v in conv_buffer.values())
            print(f"\n[{conv_count}] {current_file} "
                  f"({n_speakers} speakers, {total_segs} segments)", flush=True)

            entries = process_conversation(
                current_file, dict(conv_buffer), stereo_dir, target_sr,
                min_seg, max_seg, min_quality
            )

            if entries:
                for entry in entries:
                    all_entries.append(entry)
                    total_hours += entry["duration"] / 3600
                print(f"  => {len(entries)} segments, "
                      f"total: {total_hours:.1f}h", flush=True)
            else:
                skipped_conv += 1
                print(f"  => SKIPPED", flush=True)

            conv_buffer.clear()
            gc.collect()

            if total_hours >= max_hours:
                print(f"\n  Limite de {max_hours}h atteinte.", flush=True)
                break

            if max_conversations and conv_count >= max_conversations:
                print(f"\n  Limite de {max_conversations} conversations atteinte.",
                      flush=True)
                break

        current_file = file_id
        conv_buffer[speaker].append((start_ts, end_ts, audio_arr, sr))

        if sample_count % 500 == 0:
            print(f"  ... {sample_count} segments streamed, "
                  f"{conv_count} conversations, {total_hours:.1f}h", flush=True)

    # Process last conversation
    if conv_buffer and total_hours < max_hours:
        conv_count += 1
        n_speakers = len(conv_buffer)
        print(f"\n[{conv_count}] {current_file} ({n_speakers} speakers)", flush=True)
        entries = process_conversation(
            current_file, dict(conv_buffer), stereo_dir, target_sr,
            min_seg, max_seg, min_quality
        )
        if entries:
            for entry in entries:
                all_entries.append(entry)
                total_hours += entry["duration"] / 3600
            print(f"  => {len(entries)} segments, total: {total_hours:.1f}h", flush=True)
        else:
            skipped_conv += 1

    if not all_entries:
        print("\nAucun segment de dialogue valide.", flush=True)
        return

    # Split 90/10 train/eval
    np.random.seed(42)
    indices = np.random.permutation(len(all_entries))
    split_idx = int(len(all_entries) * 0.9)
    train_indices = indices[:split_idx]
    eval_indices = indices[split_idx:]

    for split_name, split_indices in [("train", train_indices), ("eval", eval_indices)]:
        jsonl_path = output_dir / f"{split_name}.jsonl"
        split_entries = [all_entries[i] for i in split_indices]
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for entry in split_entries:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        split_hours = sum(e['duration'] for e in split_entries) / 3600
        print(f"  {split_name}: {len(split_entries)} segments, {split_hours:.1f}h",
              flush=True)

    # Print stats
    durations = [e['duration'] for e in all_entries]
    print(f"\n{'='*60}", flush=True)
    print(f"Conversion ESLO terminee !", flush=True)
    print(f"  Conversations traitees : {conv_count}", flush=True)
    print(f"  Conversations utilisees: {conv_count - skipped_conv}", flush=True)
    print(f"  Conversations ignorees : {skipped_conv}", flush=True)
    print(f"  Segments generes       : {len(all_entries)}", flush=True)
    print(f"  Duree totale           : {total_hours:.1f}h", flush=True)
    print(f"  Duree min/moy/max      : {min(durations):.0f}s / "
          f"{sum(durations)/len(durations):.0f}s / {max(durations):.0f}s", flush=True)
    print(f"  Train: {len(train_indices)} segments", flush=True)
    print(f"  Eval : {len(eval_indices)} segments", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convertir ESLO en format moshi-finetune (dialogue quality)"
    )
    parser.add_argument("--output-dir", default="data/eslo_dataset",
                        help="Repertoire de sortie")
    parser.add_argument("--max-hours", type=float, default=999.0,
                        help="Heures maximum (default: no limit)")
    parser.add_argument("--min-seg", type=float, default=15.0,
                        help="Duree minimum par segment (secondes)")
    parser.add_argument("--max-seg", type=float, default=120.0,
                        help="Duree maximum par segment (secondes)")
    parser.add_argument("--min-quality", type=float, default=0.3,
                        help="Score qualite minimum (0-1)")
    parser.add_argument("--max-conversations", type=int, default=None,
                        help="Nombre max de conversations a traiter")
    args = parser.parse_args()

    convert_eslo(
        output_dir=args.output_dir,
        max_hours=args.max_hours,
        min_seg=args.min_seg,
        max_seg=args.max_seg,
        min_quality=args.min_quality,
        max_conversations=args.max_conversations,
    )
