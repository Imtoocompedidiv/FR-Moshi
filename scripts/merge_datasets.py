"""Merge SUMM-RE v2 + ESLO into unified dataset for training."""
import json
import random
import shutil
from pathlib import Path


def merge():
    merged_dir = Path("data/merged_v2")
    merged_stereo = merged_dir / "data_stereo"
    merged_stereo.mkdir(parents=True, exist_ok=True)

    all_entries = []

    # SUMM-RE v2 all splits
    for split in ["train", "dev", "test"]:
        split_dir = Path(f"data/moshi_dataset_v2/split_{split}")
        for jsonl_name in ["train.jsonl", "eval.jsonl"]:
            jsonl = split_dir / jsonl_name
            if not jsonl.exists():
                continue
            with open(jsonl) as f:
                for line in f:
                    entry = json.loads(line)
                    src = split_dir / entry["path"]
                    dst = merged_stereo / Path(entry["path"]).name
                    if src.exists() and not dst.exists():
                        shutil.copy2(str(src), str(dst))
                    entry["path"] = "data_stereo/" + Path(entry["path"]).name
                    all_entries.append(entry)

    n_summ = len(all_entries)
    print(f"  SUMM-RE total: {n_summ} segments")

    # ESLO
    for jsonl_name in ["train.jsonl", "eval.jsonl"]:
        jsonl = Path(f"data/eslo_dataset/{jsonl_name}")
        if not jsonl.exists():
            continue
        with open(jsonl) as f:
            for line in f:
                entry = json.loads(line)
                src = Path("data/eslo_dataset") / entry["path"]
                dst = merged_stereo / Path(entry["path"]).name
                if src.exists() and not dst.exists():
                    shutil.copy2(str(src), str(dst))
                entry["path"] = "data_stereo/" + Path(entry["path"]).name
                all_entries.append(entry)

    n_eslo = len(all_entries) - n_summ
    print(f"  ESLO: {n_eslo} segments")
    print(f"  TOTAL: {len(all_entries)} segments")

    if not all_entries:
        print("ERROR: No entries to merge!")
        return

    hours = sum(e["duration"] for e in all_entries) / 3600
    print(f"  Total hours: {hours:.1f}h")

    random.seed(42)
    random.shuffle(all_entries)
    split_idx = int(len(all_entries) * 0.9)

    for name, entries in [("train", all_entries[:split_idx]),
                          ("eval", all_entries[split_idx:])]:
        with open(merged_dir / f"{name}.jsonl", "w") as f:
            for e in entries:
                json.dump(e, f, ensure_ascii=False)
                f.write("\n")
        h = sum(e["duration"] for e in entries) / 3600
        print(f"  {name}: {len(entries)} segments, {h:.1f}h")


if __name__ == "__main__":
    merge()
