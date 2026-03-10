"""
FR-Moshi — Etape 0 v2 : Conversion SUMM-RE -> format moshi-finetune
DIALOGUE-QUALITY extraction from meeting recordings.

Key differences from v1:
- Extracts DIALOGUE SEGMENTS, not full meetings
- Uses speech timestamps to find windows where both speakers interact
- Filters out dead time (3rd/4th speaker talking, long silences)
- Produces varied-length files (15-120s) with active conversation
- Scores each segment by dialogue quality (alternation, speech ratio)

Canal gauche (0) = Moshi (locuteur principal)
Canal droit  (1) = Utilisateur (second locuteur)
"""

import json
import argparse
import sys
import gc
import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class DialogueSegment:
    """A segment of active dialogue between 2 speakers."""
    start_sec: float
    end_sec: float
    s1_speech_sec: float  # speech duration of speaker 1 in this segment
    s2_speech_sec: float  # speech duration of speaker 2 in this segment
    n_turns: int          # number of speaker alternations
    silence_ratio: float  # ratio of silence in this segment

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
        """How balanced the speakers are. 1.0 = perfectly balanced."""
        total = self.s1_speech_sec + self.s2_speech_sec
        if total <= 0:
            return 0
        minority = min(self.s1_speech_sec, self.s2_speech_sec)
        return minority / (total / 2)  # 1.0 = equal, 0.0 = one speaker only

    @property
    def quality_score(self):
        """Overall dialogue quality score (0-1)."""
        turn_score = min(1.0, self.n_turns / 6)  # 6+ turns = max score
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


def get_speech_timeline(segments, total_duration, resolution=0.1):
    """Convert segment list to a binary speech timeline at given resolution."""
    n_bins = int(total_duration / resolution) + 1
    timeline = np.zeros(n_bins, dtype=bool)
    for seg in segments:
        start_bin = int(seg['start'] / resolution)
        end_bin = int(seg['end'] / resolution)
        start_bin = max(0, min(start_bin, n_bins - 1))
        end_bin = max(0, min(end_bin, n_bins))
        timeline[start_bin:end_bin] = True
    return timeline


def find_dialogue_segments(s1_timeline, s2_timeline, other_timelines,
                           resolution=0.1, min_seg=15.0, max_seg=120.0,
                           max_gap=3.0, min_speech_ratio=0.3,
                           min_quality=0.3):
    """
    Find segments of active dialogue between s1 and s2.
    Avoids regions where other speakers are talking.
    """
    n_bins = len(s1_timeline)
    gap_bins = int(max_gap / resolution)

    # Combined "other speakers" timeline
    if other_timelines:
        others = np.any(np.stack(other_timelines), axis=0)
    else:
        others = np.zeros(n_bins, dtype=bool)

    # "Good" bins: at least one of our speakers is talking, AND no other speaker
    either_speaking = s1_timeline | s2_timeline
    good = either_speaking & ~others

    # Find contiguous regions of "good" bins, allowing small gaps
    segments = []
    in_segment = False
    seg_start = 0
    gap_count = 0

    for i in range(n_bins):
        if good[i]:
            if not in_segment:
                seg_start = i
                in_segment = True
            gap_count = 0
        else:
            if in_segment:
                gap_count += 1
                if gap_count > gap_bins:
                    # End segment
                    seg_end = i - gap_count
                    seg_dur = (seg_end - seg_start) * resolution
                    if seg_dur >= min_seg:
                        segments.append((seg_start, seg_end))
                    in_segment = False
                    gap_count = 0

    # Close last segment
    if in_segment:
        seg_end = n_bins
        seg_dur = (seg_end - seg_start) * resolution
        if seg_dur >= min_seg:
            segments.append((seg_start, seg_end))

    # Split segments that are too long
    final_segments = []
    for start, end in segments:
        seg_dur = (end - start) * resolution
        if seg_dur <= max_seg:
            final_segments.append((start, end))
        else:
            # Split into chunks at natural silence points
            pos = start
            while pos < end:
                chunk_end = min(pos + int(max_seg / resolution), end)
                # Try to find a silence point near the end for a clean cut
                if chunk_end < end:
                    search_start = max(pos + int(min_seg / resolution), chunk_end - int(10 / resolution))
                    best_cut = chunk_end
                    best_silence = 0
                    for j in range(search_start, chunk_end):
                        if not s1_timeline[j] and not s2_timeline[j]:
                            # Count consecutive silence
                            silence = 0
                            while j + silence < chunk_end and not s1_timeline[j + silence] and not s2_timeline[j + silence]:
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

        # Count turns (alternations between speakers)
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


def process_meeting_v2(speakers, output_dir, target_sr=24000,
                       min_seg=15.0, max_seg=120.0, min_quality=0.3):
    """
    Process a meeting: extract dialogue-quality segments.
    Returns list of JSONL entries.
    """
    import soundfile as sf

    if len(speakers) < 2:
        return []

    # Compute speech duration per speaker
    speakers_scored = []
    for s in speakers:
        segs = s.get('segments', [])
        speech_dur = sum(seg['end'] - seg['start'] for seg in segs)
        audio_dur = len(s['audio']['array']) / s['audio']['sampling_rate']
        speakers_scored.append((s, speech_dur, audio_dur))

    # Sort by speech duration
    speakers_scored.sort(key=lambda x: -x[1])

    # Select 2 most active speakers
    s1, s1_speech, s1_dur = speakers_scored[0]
    s2, s2_speech, s2_dur = speakers_scored[1]

    # Both need meaningful speech
    if s1_speech < 20 or s2_speech < 20:
        return []

    meeting_id = s1['meeting_id']
    total_dur = max(s1_dur, s2_dur)

    # Build speech timelines
    resolution = 0.1  # 100ms resolution
    s1_timeline = get_speech_timeline(s1.get('segments', []), total_dur, resolution)
    s2_timeline = get_speech_timeline(s2.get('segments', []), total_dur, resolution)

    # Build "other speakers" timelines
    other_timelines = []
    for s, speech_dur, audio_dur in speakers_scored[2:]:
        if speech_dur > 5:  # Only count speakers with >5s of speech
            other_timelines.append(
                get_speech_timeline(s.get('segments', []), total_dur, resolution)
            )

    # Find dialogue segments
    segments = find_dialogue_segments(
        s1_timeline, s2_timeline, other_timelines,
        resolution=resolution,
        min_seg=min_seg,
        max_seg=max_seg,
        min_quality=min_quality,
    )

    if not segments:
        return []

    # Resample audio
    s1_audio = resample_audio(s1['audio']['array'], s1['audio']['sampling_rate'], target_sr)
    s2_audio = resample_audio(s2['audio']['array'], s2['audio']['sampling_rate'], target_sr)

    # Align lengths
    max_len = max(len(s1_audio), len(s2_audio))
    if len(s1_audio) < max_len:
        s1_audio = np.pad(s1_audio, (0, max_len - len(s1_audio)))
    if len(s2_audio) < max_len:
        s2_audio = np.pad(s2_audio, (0, max_len - len(s2_audio)))

    entries = []
    for idx, seg in enumerate(segments):
        start_sample = int(seg.start_sec * target_sr)
        end_sample = int(seg.end_sec * target_sr)
        end_sample = min(end_sample, max_len)

        s1_chunk = s1_audio[start_sample:end_sample].copy()
        s2_chunk = s2_audio[start_sample:end_sample].copy()

        # Normalize
        for arr in [s1_chunk, s2_chunk]:
            peak = np.abs(arr).max()
            if peak > 0.01:  # Only normalize if there's actual audio
                arr *= 0.9 / max(peak, 0.95)

        # Write stereo WAV
        stereo = np.stack([s1_chunk, s2_chunk], axis=-1).astype(np.float32)
        filename = f"{meeting_id}_{s1['speaker_id']}_{s2['speaker_id']}_{idx:02d}.wav"
        wav_path = output_dir / filename
        sf.write(str(wav_path), stereo, target_sr)

        duration = seg.duration
        entries.append({
            "path": f"data_stereo/{filename}",
            "duration": round(duration, 6),
        })

        print(f"    -> {filename} ({duration:.0f}s, "
              f"quality={seg.quality_score:.2f}, "
              f"turns={seg.n_turns}, "
              f"speech={seg.speech_ratio:.0%}, "
              f"balance={seg.balance:.0%})", flush=True)

    return entries


def convert_summ_re_v2(
    split="train",
    output_dir="data/moshi_dataset",
    max_hours=999.0,
    target_sr=24000,
    min_seg=15.0,
    max_seg=120.0,
    min_quality=0.3,
):
    """Convertit SUMM-RE en format moshi-finetune v2 (dialogue-quality segments)."""
    from datasets import load_dataset

    output_dir = Path(output_dir)
    stereo_dir = output_dir / "data_stereo"
    stereo_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== SUMM-RE -> moshi-finetune v2 (dialogue extraction) ===", flush=True)
    print(f"  Split     : {split}", flush=True)
    print(f"  Max hours : {max_hours}", flush=True)
    print(f"  Segments  : {min_seg}-{max_seg}s", flush=True)
    print(f"  Min quality: {min_quality}", flush=True)
    print(f"  Output    : {output_dir}", flush=True)
    print(flush=True)

    ds = load_dataset("linagora/SUMM-RE", split=split, streaming=True)

    all_entries = []
    total_hours = 0
    current_meeting = None
    meeting_buffer = []
    sample_count = 0
    meeting_count = 0
    skipped_meetings = 0

    for sample in ds:
        sample_count += 1
        mid = sample['meeting_id']

        if mid != current_meeting and current_meeting is not None:
            meeting_count += 1
            n_speakers = len(meeting_buffer)
            print(f"\n[{meeting_count}] {current_meeting} ({n_speakers} speakers)", flush=True)

            entries = process_meeting_v2(
                meeting_buffer, stereo_dir, target_sr, min_seg, max_seg, min_quality
            )
            if entries:
                for entry in entries:
                    all_entries.append(entry)
                    total_hours += entry["duration"] / 3600
                print(f"  => {len(entries)} segments, "
                      f"total: {total_hours:.1f}h", flush=True)
            else:
                skipped_meetings += 1
                print(f"  => SKIPPED (no quality dialogue found)", flush=True)

            meeting_buffer = []
            gc.collect()

            if total_hours >= max_hours:
                print(f"\n  Limite de {max_hours}h atteinte.", flush=True)
                break

        current_meeting = mid
        meeting_buffer.append(sample)

        if sample_count % 20 == 0:
            print(f"  ... {sample_count} samples streamed", flush=True)

    # Process last meeting
    if meeting_buffer and total_hours < max_hours:
        meeting_count += 1
        print(f"\n[{meeting_count}] {current_meeting} ({len(meeting_buffer)} speakers)", flush=True)
        entries = process_meeting_v2(
            meeting_buffer, stereo_dir, target_sr, min_seg, max_seg, min_quality
        )
        if entries:
            for entry in entries:
                all_entries.append(entry)
                total_hours += entry["duration"] / 3600
            print(f"  => {len(entries)} segments, total: {total_hours:.1f}h", flush=True)
        else:
            skipped_meetings += 1

    if not all_entries:
        print("\nAucun segment de dialogue valide genere.", flush=True)
        return

    # Write JSONL
    jsonl_name = "train.jsonl" if split == "train" else "eval.jsonl"
    jsonl_path = output_dir / jsonl_name
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    # Print stats
    durations = [e['duration'] for e in all_entries]
    print(f"\n{'='*60}", flush=True)
    print(f"Conversion v2 terminee !", flush=True)
    print(f"  Meetings traites  : {meeting_count}", flush=True)
    print(f"  Meetings utilises : {meeting_count - skipped_meetings}", flush=True)
    print(f"  Meetings ignores  : {skipped_meetings}", flush=True)
    print(f"  Segments generes  : {len(all_entries)}", flush=True)
    print(f"  Duree totale      : {total_hours:.1f}h", flush=True)
    print(f"  Duree min/moy/max : {min(durations):.0f}s / "
          f"{sum(durations)/len(durations):.0f}s / {max(durations):.0f}s", flush=True)
    print(f"  JSONL             : {jsonl_path}", flush=True)
    print(f"  Stereo            : {stereo_dir}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convertir SUMM-RE en format moshi-finetune v2 (dialogue quality)"
    )
    parser.add_argument("--split", default="train",
                        help="Split HuggingFace (train/dev/test)")
    parser.add_argument("--output-dir", default="data/moshi_dataset",
                        help="Repertoire de sortie")
    parser.add_argument("--max-hours", type=float, default=999.0,
                        help="Heures maximum a traiter (default: no limit)")
    parser.add_argument("--min-seg", type=float, default=15.0,
                        help="Duree minimum par segment (secondes)")
    parser.add_argument("--max-seg", type=float, default=120.0,
                        help="Duree maximum par segment (secondes)")
    parser.add_argument("--min-quality", type=float, default=0.3,
                        help="Score qualite minimum (0-1)")
    args = parser.parse_args()

    convert_summ_re_v2(
        split=args.split,
        output_dir=args.output_dir,
        max_hours=args.max_hours,
        min_seg=args.min_seg,
        max_seg=args.max_seg,
        min_quality=args.min_quality,
    )
