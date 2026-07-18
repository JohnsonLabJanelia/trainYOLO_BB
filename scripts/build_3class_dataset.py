#!/usr/bin/env python3
"""Build the combined 3-class detect dataset from the *_bb export folders.

Sources: remus_bb, kimchi_bb, wasabi_bb. Each already exports YOLO detect
format (images + `class cx cy w h` labels) with the class scheme:
    0 = Mouse, 1 = SideCyl, 2 = VertCyl

All source images are named Cam2005325_<frame>.jpg, so files are prefixed with
the source name to avoid collisions. Label files contain triplicate duplicate
lines (one per export pass) which are de-duplicated here.

Output: data/combined_det_3class/{train,val,test}/{images,labels} + data.yaml
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ["remus_bb", "kimchi_bb", "wasabi_bb", "mouseCylCircleArea", "tata_cyl_bb", "rat_bb_cyl",
           "blossom_bb_cyl", "blossom_cyl_2", "tata_bb_cyl_2"]
SPLITS = ["train", "val", "test"]
DST = ROOT / "data" / "combined_det_3class"
NAMES = {0: "Mouse", 1: "SideCyl", 2: "VertCyl"}


def dedupe_label(text: str) -> str:
    seen = []
    for line in text.splitlines():
        line = line.strip()
        if line and line not in seen:
            seen.append(line)
    return "\n".join(seen) + ("\n" if seen else "")


def main():
    if DST.exists():
        shutil.rmtree(DST)
    for split in SPLITS:
        (DST / split / "images").mkdir(parents=True, exist_ok=True)
        (DST / split / "labels").mkdir(parents=True, exist_ok=True)

    # Frames the user flagged as mislabeled in the label browser (purged reproducibly).
    excl_file = ROOT / "data" / "flagged_exclude.txt"
    EXCLUDE = set()
    if excl_file.is_file():
        EXCLUDE = {l.strip() for l in excl_file.read_text().splitlines() if l.strip()}
    print(f"Excluding {len(EXCLUDE)} user-flagged frames")

    counts = {s: 0 for s in SPLITS}
    cls_counts = {0: 0, 1: 0, 2: 0}
    n_excluded = 0
    for src in SOURCES:
        for split in SPLITS:
            img_dir = ROOT / src / "export" / split / "images"
            lbl_dir = ROOT / src / "export" / split / "labels"
            if not img_dir.is_dir():
                continue
            for img in sorted(img_dir.glob("*.jpg")):
                stem = img.stem
                lbl = lbl_dir / f"{stem}.txt"
                out_stem = f"{src}_{stem}"
                if out_stem in EXCLUDE:
                    n_excluded += 1
                    continue
                shutil.copy2(img, DST / split / "images" / f"{out_stem}.jpg")
                if lbl.is_file():
                    deduped = dedupe_label(lbl.read_text())
                    (DST / split / "labels" / f"{out_stem}.txt").write_text(deduped)
                    for line in deduped.splitlines():
                        c = int(line.split()[0])
                        cls_counts[c] = cls_counts.get(c, 0) + 1
                else:
                    # negative/background image — empty label
                    (DST / split / "labels" / f"{out_stem}.txt").write_text("")
                counts[split] += 1

    # Extra source: SAM3 auto-labeled data (data/sam3_mouse/{train,val}). The mouse
    # MOVES (diverse) but the cylinders are STATIC (same spot for ~800 frames/session),
    # so keeping every cylinder frame both over-represents SideCyl and pins it to a
    # fixed position. Fix: keep EVERY mouse frame, but temporally subsample the
    # non-mouse (cylinder/negative) frames to a cap per session.
    MAX_NONMOUSE_PER_SESSION = 80
    sam3 = ROOT / "data" / "sam3_mouse"
    if sam3.is_dir():
        for split in ("train", "val"):
            img_dir = sam3 / split / "images"
            lbl_dir = sam3 / split / "labels"
            if not img_dir.is_dir():
                continue
            imgs = sorted(img_dir.glob("*.jpg"))
            labels = {}
            for img in imgs:
                lbl = lbl_dir / f"{img.stem}.txt"
                labels[img.stem] = dedupe_label(lbl.read_text()) if lbl.is_file() else ""

            def has_mouse(stem):
                return any(line.split()[0] == "0" for line in labels[stem].splitlines() if line.strip())

            def session_of(stem):  # strip trailing _fNNNNNN
                return stem.rsplit("_f", 1)[0]

            # non-mouse frames grouped by session, temporally strided to the cap
            nonmouse_by_sess = {}
            for img in imgs:
                if not has_mouse(img.stem):
                    nonmouse_by_sess.setdefault(session_of(img.stem), []).append(img)
            keep_nonmouse = set()
            for sess, lst in nonmouse_by_sess.items():
                if len(lst) <= MAX_NONMOUSE_PER_SESSION:
                    keep_nonmouse.update(lst)
                else:
                    stride = len(lst) / MAX_NONMOUSE_PER_SESSION
                    keep_nonmouse.update(lst[int(i * stride)] for i in range(MAX_NONMOUSE_PER_SESSION))

            for img in imgs:
                if not has_mouse(img.stem) and img not in keep_nonmouse:
                    continue
                out_stem = f"sam3_{img.stem}"
                if out_stem in EXCLUDE:
                    n_excluded += 1
                    continue
                shutil.copy2(img, DST / split / "images" / f"{out_stem}.jpg")
                (DST / split / "labels" / f"{out_stem}.txt").write_text(labels[img.stem])
                for line in labels[img.stem].splitlines():
                    if line.strip():
                        cls_counts[int(line.split()[0])] += 1
                counts[split] += 1

    # Extra source: converted OBB cylinder datasets (SideCyl from varied positions).
    cyl = ROOT / "data" / "cyl_extra"
    if cyl.is_dir():
        for split in SPLITS:
            img_dir = cyl / split / "images"
            lbl_dir = cyl / split / "labels"
            if not img_dir.is_dir():
                continue
            for img in sorted(img_dir.glob("*.jpg")):
                out_stem = f"cylx_{img.stem}"
                if out_stem in EXCLUDE:
                    n_excluded += 1
                    continue
                shutil.copy2(img, DST / split / "images" / f"{out_stem}.jpg")
                lbl = lbl_dir / f"{img.stem}.txt"
                deduped = dedupe_label(lbl.read_text()) if lbl.is_file() else ""
                (DST / split / "labels" / f"{out_stem}.txt").write_text(deduped)
                for line in deduped.splitlines():
                    if line.strip():
                        cls_counts[int(line.split()[0])] += 1
                counts[split] += 1

    names_block = "\n".join(f"  {k}: {v}" for k, v in NAMES.items())
    yaml = (
        "# Combined 3-class detect dataset (remus_bb + kimchi_bb + wasabi_bb)\n"
        f"path: {DST.resolve()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n\n"
        "nc: 3\n\n"
        "names:\n"
        f"{names_block}\n"
    )
    (DST / "data.yaml").write_text(yaml)

    print("Built dataset at", DST, f"(excluded {n_excluded} flagged frames)")
    for split in SPLITS:
        print(f"  {split}: {counts[split]} images")
    print("Class instance counts (deduped):")
    for c, n in cls_counts.items():
        print(f"  {c} ({NAMES[c]}): {n}")
    print("\ndata.yaml:\n" + yaml)


if __name__ == "__main__":
    main()
