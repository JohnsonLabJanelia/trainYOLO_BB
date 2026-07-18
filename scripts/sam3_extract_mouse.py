#!/usr/bin/env python3
"""Auto-label 3-class detect data (Mouse + SideCyl + VertCyl) from circular-arena
climb videos using SAM3.

For each CIRCULAR-arena session we sample frames and prompt SAM3 with three text
concepts -- "mouse", the white square block (SideCyl), the white round disk
(VertCyl) -- keep ONLY high-confidence, sensibly-sized detections, and write every
confident object into one YOLO label file so nothing on screen is left unlabeled
(an unlabeled cylinder would teach the model that cylinders are background).

Square-table sessions are SKIPPED: their wall reflections create bright white
artifacts that get confused with the white cylinders. Only circular arenas are used.

Class map (matches the deployed convention):
    0 = Mouse
    1 = SideCyl   (cylinder on its side  -> white square/rectangle from top)
    2 = VertCyl   (cylinder standing up  -> white circle/disk   from top)

Requires HuggingFace auth + accepted license for the gated facebook/sam3 repo.

Run:
    python3 scripts/sam3_extract_mouse.py                 # all circular sessions
    python3 scripts/sam3_extract_mouse.py --limit 40      # smoke test
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sam3_mouse"           # YOLO source dir (train/ val/)
CLIMB = Path("/home/ratan/orange_data/exp/climb")
CAM = "Cam2005325.mp4"                        # the single top-cam file

# Only CIRCULAR-arena sessions (square-table reflections confuse cylinder detection).
# Classified by eyeballing one mid-frame per session (sam3_sessions_overview.jpg):
# all the square-table 06_15 recordings + both blossom sessions are dropped.
CIRCLE_SESSIONS = {
    "romulus": {"2026_07_08_14_01_31", "2026_07_13_13_40_40"},
    "remus":   {"2026_07_08_14_51_52", "2026_07_13_14_37_01"},
    "kimchi":  {"2026_07_14_13_48_38", "2026_07_14_14_07_13"},
    "wasabi":  {"2026_07_14_14_47_38"},
}

# (text prompt, class id, short label). One SAM3 pass per prompt per frame.
PROMPTS = [
    ("mouse",              0, "mouse"),
    ("white square block", 1, "sidecyl"),
    ("white round disk",   2, "vertcyl"),
]

CONF = 0.70          # cylinder confidence floor -- static white objects detect strongly.
CONF_MOUSE = 0.78    # mouse floor is higher: drops faint rim/edge detections that are
                     # ambiguous with dark wall shadow (calibration: real mice 0.66-0.91).
MOUSE_GRAY_LO = 0.40 # mouse gray zone: if the top "mouse" score is in [0.40, CONF_MOUSE)
                     # the frame is ambiguous -- a real-but-faint mouse might be present.
                     # Drop the whole frame rather than risk an UNLABELED mouse (which
                     # would teach the model to miss faint mice -- the live failure mode).
MAX_BOX_FRAC = 0.35  # no single object (mouse/cyl) fills a third of the frame ->
                     # rejects the whole white arena matching "white ... disk"
MIN_BOX_FRAC = 0.0004 # ...nor a few-pixel speck. The SideCyl cube is genuinely small
                      # (~0.04x0.06 = 0.0024 of frame), so this floor must stay well below it.
SAMPLE_EVERY = 60    # sample 1 frame per this many (30fps -> ~1 every 2s)
MAX_PER_VIDEO = 800  # cap so one long session can't dominate the set
VAL_FRAC = 0.10      # last 10% of kept frames per session -> val
MAX_SIDE = 1280      # save frames resized (aspect kept, NO padding) to this max side


def find_sessions():
    """Return [(animal, video_path)] for the circular-arena single-cam sessions."""
    sessions = []
    for animal, keep in CIRCLE_SESSIONS.items():
        adir = CLIMB / animal
        if not adir.is_dir():
            print(f"  [skip] no dir for {animal}: {adir}")
            continue
        for sess_name in sorted(keep):
            vid = adir / sess_name / CAM
            if vid.is_file():
                sessions.append((animal, vid))
            else:
                print(f"  [skip] missing {vid}")
    return sessions


def resize_keep_aspect(img, max_side):
    """Plain resize to a max side, preserving aspect. NO padding -- normalized YOLO
    labels stay valid (they are invariant to axis-uniform scaling). Letterbox padding
    would shift boxes vertically and misalign every label; ultralytics letterboxes
    internally at train time, so we must NOT pad here."""
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1.0:
        return img
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


SAM3_SRC = "/home/ratan/src/sam3"   # local package dir; `pip install .` didn't register it


def build_sam3():
    if SAM3_SRC not in sys.path:
        sys.path.insert(0, SAM3_SRC)
    import torch
    # SAM3 weights are bf16; every example runs inference under a global autocast.
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model()
    proc = Sam3Processor(model, confidence_threshold=CONF)
    return proc


def _to_np(t):
    return t.detach().float().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def detect_objects(proc, bgr):
    """Run every PROMPT on a BGR frame. Returns (dets, drop) where dets is a list of
    (cls, cx,cy,bw,bh, score) normalized boxes (best confident+sized box per prompt),
    and drop=True means the frame is ambiguous for the mouse class and must be skipped
    entirely (a faint mouse may be present but below threshold -> don't leave it
    unlabeled)."""
    H, W = bgr.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    st = proc.set_image(pil)                 # embed once, reuse across prompts
    dets = []
    for text, cls, _ in PROMPTS:
        proc.reset_all_prompts(st)
        st = proc.set_text_prompt(state=st, prompt=text)
        boxes, scores = st.get("boxes"), st.get("scores")
        if boxes is None or len(boxes) == 0:
            continue
        boxes = _to_np(boxes).astype(float)
        scores = _to_np(scores).astype(float).reshape(-1)
        i = int(scores.argmax())
        top = float(scores[i])
        floor = CONF_MOUSE if cls == 0 else CONF
        # Mouse gray zone -> ambiguous frame, signal drop.
        if cls == 0 and MOUSE_GRAY_LO <= top < floor:
            return [], True
        if top < floor:
            continue
        x1, y1, x2, y2 = boxes[i]
        bw, bh = abs(x2 - x1) / W, abs(y2 - y1) / H
        if bw > MAX_BOX_FRAC or bh > MAX_BOX_FRAC or bw * bh < MIN_BOX_FRAC:
            continue
        cx = min(max(((x1 + x2) / 2) / W, 0), 1)
        cy = min(max(((y1 + y2) / 2) / H, 0), 1)
        dets.append((cls, cx, cy, min(bw, 1), min(bh, 1), top))
    return dets, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="if >0, sample at most this many frames per video (smoke test)")
    args = ap.parse_args()

    sessions = find_sessions()
    print(f"Found {len(sessions)} circular-arena sessions:")
    for a, v in sessions:
        print(f"  {a}: {v}")
    if not sessions:
        sys.exit("No sessions found -- check CIRCLE_SESSIONS / CLIMB path.")

    for split in ("train", "val"):
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)

    print("\nBuilding SAM3 (downloads gated checkpoint on first run)...")
    proc = build_sam3()
    print("SAM3 ready.\n")

    total_kept = 0
    cls_counts = {0: 0, 1: 0, 2: 0}
    for animal, vpath in sessions:
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"  [skip] cannot open {vpath}")
            continue
        stem = f"{animal}_{vpath.parent.name}"
        kept = []   # (out_img, label_text, frame_idx)
        idx, sampled = 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % SAMPLE_EVERY == 0:
                sampled += 1
                dets, drop = detect_objects(proc, frame)
                if drop:
                    idx += 1
                    continue
                if dets:
                    lines = "".join(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
                                    for (c, cx, cy, bw, bh, _s) in dets)
                    out_img = resize_keep_aspect(frame, MAX_SIDE)
                    kept.append((out_img, lines, idx))
                    for (c, *_r) in dets:
                        cls_counts[c] += 1
                if args.limit and sampled >= args.limit:
                    break
                if len(kept) >= MAX_PER_VIDEO:
                    break
            idx += 1
        cap.release()

        n_val = int(len(kept) * VAL_FRAC)
        for j, (out_img, lines, fidx) in enumerate(kept):
            split = "val" if j >= len(kept) - n_val else "train"
            name = f"{stem}_f{fidx:06d}"
            cv2.imwrite(str(OUT / split / "images" / f"{name}.jpg"),
                        out_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (OUT / split / "labels" / f"{name}.txt").write_text(lines)
        total_kept += len(kept)
        print(f"  {stem}: sampled {sampled}, kept {len(kept)} frames")

    (OUT / "data.yaml").write_text(
        "# SAM3 auto-labeled 3-class source (circular arenas only)\n"
        f"path: {OUT.resolve()}\n"
        "train: train/images\nval: val/images\n\n"
        "nc: 3\n\nnames:\n  0: Mouse\n  1: SideCyl\n  2: VertCyl\n")

    print(f"\nDONE. {total_kept} frames kept.")
    print(f"Instances: Mouse={cls_counts[0]} SideCyl={cls_counts[1]} VertCyl={cls_counts[2]}")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
