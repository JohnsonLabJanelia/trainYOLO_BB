#!/usr/bin/env python3
"""Convert the OBB cylinder-only datasets into axis-aligned SideCyl detect labels.

These datasets (2cyl_small, small_cyl, obb_cyl_4, obb_cyl_5) are human-annotated
OBB (8-corner) boxes of the white square cube, all labeled class 0 = "class_0".
In OUR 2-class scheme class 0 is Mouse, so we MUST remap: the cube -> SideCyl (1).

They add SideCyl from VARIED positions/sessions -- the antidote to the SAM3 static
cube always sitting in the same spot. VertCyl is not a class (2-class network), so
the white disk that appears in some frames is just background -- nothing to add.

Output: data/cyl_extra/{train,val,test}/{images,labels}  (merged by build script)
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "cyl_extra"
SOURCES = {
    "2cyl_small": "2cyl_small/export",
    "small_cyl":  "small_cyl/export",
    "obb_cyl_4":  "obb_cyl_4/yolo_obb_dataset",
    "obb_cyl_5":  "obb_cyl_5/yolo_obb_dataset",
}
SPLITS = ["train", "val", "test"]


def obb_to_bbox(corners):
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    clamp = lambda v: min(max(v, 0.0), 1.0)
    return clamp(cx), clamp(cy), clamp(w), clamp(h)


def read_obb_sidecyl(lbl_path):
    """Deduped SideCyl bboxes '1 cx cy w h' from an OBB label file (class 0 -> 1)."""
    if not lbl_path.is_file():
        return []
    out, seen = [], set()
    for line in lbl_path.read_text().splitlines():
        p = line.split()
        if len(p) != 9 or line in seen:
            continue
        seen.add(line)
        corners = [(float(p[1 + 2 * i]), float(p[2 + 2 * i])) for i in range(4)]
        cx, cy, w, h = obb_to_bbox(corners)
        if w > 0 and h > 0:
            out.append((1, cx, cy, w, h))
    return out


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for sp in SPLITS:
        (OUT / sp / "images").mkdir(parents=True, exist_ok=True)
        (OUT / sp / "labels").mkdir(parents=True, exist_ok=True)

    n_img, n_side = 0, 0
    for name, base in SOURCES.items():
        for sp in SPLITS:
            idir = ROOT / base / sp / "images"
            ldir = ROOT / base / sp / "labels"
            if not idir.is_dir():
                continue
            for ip in sorted(idir.glob("*.jpg")):
                boxes = read_obb_sidecyl(ldir / f"{ip.stem}.txt")
                if not boxes:
                    continue
                out_stem = f"{name}_{ip.stem}"
                shutil.copy2(ip, OUT / sp / "images" / f"{out_stem}.jpg")
                (OUT / sp / "labels" / f"{out_stem}.txt").write_text(
                    "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for (c, cx, cy, w, h) in boxes))
                n_img += 1
                n_side += len(boxes)

    print(f"Converted {n_img} images, {n_side} SideCyl boxes -> {OUT}")


if __name__ == "__main__":
    main()
