#!/usr/bin/env python3
"""Build a 2-class (Mouse + SideCyl) mirror-augmented detect dataset.

VertCyl (class 2) is DROPPED here -- we don't detect the vertical cylinder (it's
placed from the curriculum, and the white disk is just background for this task).
The SAM3 source labels it, but this step filters it out. Classes: 0=Mouse, 1=SideCyl.

For every TRAIN image, write 4 copies: original + horizontal flip + vertical flip +
both (180). Labels flip accordingly (cx->1-cx for H, cy->1-cy for V) so objects are
seen equally on every side (no left/right bias). VAL/TEST: images copied, labels
filtered to drop VertCyl.

Source: data/combined_det_3class   ->   Output: data/combined_det_2class_mirror
"""

import shutil
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "combined_det_3class"
DST = ROOT / "data" / "combined_det_2class_mirror"
NAMES = {0: "Mouse", 1: "SideCyl"}
KEEP = {0, 1}   # drop class 2 (VertCyl)


def read_labels(path):
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) == 5 and int(p[0]) in KEEP:
            out.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return out


def write_labels(path, boxes):
    path.write_text("\n".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                              for (c, cx, cy, w, h) in boxes) + ("\n" if boxes else ""))


def flip_boxes(boxes, hflip, vflip):
    out = []
    for (c, cx, cy, w, h) in boxes:
        if hflip:
            cx = 1.0 - cx
        if vflip:
            cy = 1.0 - cy
        out.append((c, cx, cy, w, h))
    return out


def main():
    if DST.exists():
        shutil.rmtree(DST)
    for split in ("train", "val", "test"):
        (DST / split / "images").mkdir(parents=True, exist_ok=True)
        (DST / split / "labels").mkdir(parents=True, exist_ok=True)

    # VAL/TEST: copy images unchanged, filter labels to drop VertCyl
    for split in ("val", "test"):
        sdir = SRC / split
        if not sdir.is_dir():
            continue
        for img in sorted((sdir / "images").glob("*.jpg")):
            shutil.copy2(img, DST / split / "images" / img.name)
            write_labels(DST / split / "labels" / f"{img.stem}.txt",
                         read_labels(sdir / "labels" / f"{img.stem}.txt"))

    # TRAIN: 4-way mirror
    img_dir = SRC / "train" / "images"
    lbl_dir = SRC / "train" / "labels"
    variants = [("", None, False, False), ("_h", 1, True, False),
                ("_v", 0, False, True), ("_hv", -1, True, True)]
    n = 0
    cls_counts = {0: 0, 1: 0}
    for img_path in sorted(img_dir.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        boxes = read_labels(lbl_dir / f"{img_path.stem}.txt")
        for suf, code, hf, vf in variants:
            out_img = img if code is None else cv2.flip(img, code)
            cv2.imwrite(str(DST / "train" / "images" / f"{img_path.stem}{suf}.jpg"),
                        out_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            fb = flip_boxes(boxes, hf, vf)
            write_labels(DST / "train" / "labels" / f"{img_path.stem}{suf}.txt", fb)
            for (c, *_) in fb:
                cls_counts[c] += 1
            n += 1

    names_block = "\n".join(f"  {k}: {v}" for k, v in NAMES.items())
    (DST / "data.yaml").write_text(
        "# 2-class (Mouse+SideCyl) mirror-augmented detect dataset\n"
        f"path: {DST.resolve()}\n"
        "train: train/images\nval: val/images\ntest: test/images\n\n"
        f"nc: 2\n\nnames:\n{names_block}\n")

    print(f"Wrote {n} train images (4x mirrored). "
          f"Instances: Mouse={cls_counts[0]} SideCyl={cls_counts[1]}")


if __name__ == "__main__":
    main()
