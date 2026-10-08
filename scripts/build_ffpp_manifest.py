#!/usr/bin/env python3
"""Create train/val/test JSONL manifests from extracted DeepfakeBench FF++ frames.

Expected common layout: ROOT/{original_sequences,manipulated_sequences}/.../c23/<video>/<frame>.png.
It deliberately discovers files recursively, so it also works with DFB's cropped-face exports.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def group_key(path: Path):
    return str(path.parent)


def class_for(path: Path, types):
    parts = {p.lower() for p in path.parts}
    if "original_sequences" in parts or "original" in parts or "real" in parts:
        return 0, -1
    for i, name in enumerate(types):
        if name.lower() in parts:
            return 1, i
    return None


def read_official_splits(path):
    """Read standard FF++ train/val/test JSON files (lists of source-id pairs)."""
    result = {}
    for split in ("train", "val", "test"):
        candidate = Path(path) / f"{split}.json"
        if not candidate.exists():
            continue
        pairs = json.loads(candidate.read_text())
        ids = set()
        for a, b in pairs:
            a, b = str(a).zfill(3), str(b).zfill(3)
            ids.update((a, b, f"{a}_{b}", f"{b}_{a}"))
        result[split] = ids
    if set(result) != {"train", "val", "test"}:
        raise ValueError("--official-splits must contain train.json, val.json and test.json")
    return result


def official_split(video_id, split_ids):
    name = Path(video_id).name
    for split, ids in split_ids.items():
        if name in ids:
            return split
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="FF++/DeepfakeBench extracted-frame root")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--types", nargs="+", default=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"])
    p.add_argument("--frames-per-video", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--official-splits", help="Directory with FF++ train.json, val.json, test.json; preferred for benchmark results")
    args = p.parse_args()
    rng = random.Random(args.seed)
    groups = defaultdict(list)
    for path in Path(args.root).rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        label = class_for(path, args.types)
        if label is not None:
            groups[(label, group_key(path))].append(path)
    if not groups:
        raise SystemExit("No labelled images found. Check --root and --types.")
    records = []
    for ((label, type_label), vid), paths in groups.items():
        chosen = rng.sample(paths, min(args.frames_per_video, len(paths)))
        records.extend({"path": str(x), "label": label, "type_label": type_label, "video_id": vid} for x in chosen)
    # Prefer the canonical FF++ split files. Otherwise make a reproducible, video-level split.
    video_rows = defaultdict(list)
    for r in records:
        video_rows[r["video_id"]].append(r)
    splits = {"train": [], "val": [], "test": []}
    if args.official_splits:
        split_ids = read_official_splits(args.official_splits)
        unmatched = 0
        for vid, rows in video_rows.items():
            split = official_split(vid, split_ids)
            if split is None: unmatched += 1
            else: splits[split].extend(rows)
        if unmatched:
            print(f"warning: {unmatched} video folders did not match the official split names")
    else:
        strata = defaultdict(list)
        for vid, rows in video_rows.items():
            strata[(rows[0]["label"], rows[0]["type_label"])].append(vid)
        for vids in strata.values():
            rng.shuffle(vids)
            n = len(vids)
            a, b = max(1, int(.8 * n)), max(1, int(.9 * n))
            for split, subset in (("train", vids[:a]), ("val", vids[a:b]), ("test", vids[b:])):
                for vid in subset: splits[split].extend(video_rows[vid])
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        with (out / f"ffpp_{split}.jsonl").open("w") as f:
            for row in rows: f.write(json.dumps(row) + "\n")
        print(split, len(rows), "frames", len({r['video_id'] for r in rows}), "videos")


if __name__ == "__main__":
    main()
