#!/usr/bin/env python3
"""
Merge OBB datasets into a combined detect dataset.

Converts OBB labels (class x1 y1 x2 y2 x3 y3 x4 y4) to detect format
(class cx cy w h) and copies images with a unique prefix to avoid
name collisions.

Usage:
    python scripts/merge_obb_to_det.py \
        --sources obb_cyl_4/yolo_obb_dataset \
        --dst data/combined_det_cyl \
        --prefix ds5
"""

import argparse
import shutil
from pathlib import Path


def obb_to_detect(line: str) -> str | None:
    """Convert 'cls x1 y1 x2 y2 x3 y3 x4 y4' -> 'cls cx cy w h'"""
    parts = line.strip().split()
    if len(parts) != 9:
        return None
    cls = parts[0]
    xs = [float(parts[i]) for i in range(1, 9, 2)]
    ys = [float(parts[i]) for i in range(2, 9, 2)]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def merge_source(src: Path, dst: Path, prefix: str):
    added = {"train": 0, "val": 0}

    for split in ["train", "val"]:
        img_src = src / split / "images"
        lbl_src = src / split / "labels"
        img_dst = dst / split / "images"
        lbl_dst = dst / split / "labels"

        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        if not img_src.exists():
            continue

        for img_file in sorted(img_src.iterdir()):
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            stem = img_file.stem
            lbl_file = lbl_src / f"{stem}.txt"

            new_name = f"{prefix}_{stem}"
            dst_img = img_dst / f"{new_name}{img_file.suffix}"
            dst_lbl = lbl_dst / f"{new_name}.txt"

            if dst_img.exists():
                print(f"  Skipping (exists): {dst_img.name}")
                continue

            shutil.copy2(img_file, dst_img)

            if lbl_file.exists():
                lines = lbl_file.read_text().strip().splitlines()
                det_lines = [l for l in (obb_to_detect(ln) for ln in lines) if l]
                dst_lbl.write_text("\n".join(det_lines) + "\n" if det_lines else "")
            else:
                dst_lbl.write_text("")

            added[split] += 1

    return added


def main():
    ap = argparse.ArgumentParser(description="Merge OBB dataset into combined detect dataset")
    ap.add_argument("--sources", nargs="+", required=True,
                     help="OBB dataset dirs (with train/val/images/labels)")
    ap.add_argument("--dst", default="data/combined_det_cyl",
                     help="Output combined detect dataset")
    ap.add_argument("--prefix", default=None,
                     help="Prefix for file names (auto-detected if omitted)")
    args = ap.parse_args()

    dst = Path(args.dst)

    for i, src_path in enumerate(args.sources):
        src = Path(src_path)
        if not src.exists():
            print(f"Source not found: {src}")
            continue

        prefix = args.prefix if args.prefix else f"ds{i+1}"
        print(f"Merging {src} with prefix '{prefix}'...")
        added = merge_source(src, dst, prefix)
        print(f"  Added {added['train']} train, {added['val']} val")

    # Show totals
    for split in ["train", "val"]:
        img_dir = dst / split / "images"
        if img_dir.exists():
            n = len(list(img_dir.iterdir()))
            print(f"Total {split}: {n} images")


if __name__ == "__main__":
    main()
