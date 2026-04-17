#!/usr/bin/env python3
"""
Evaluate YOLO-Seg model for OBB fitting.

Runs seg inference on val images, extracts masks, fits minAreaRect
to each mask, and compares with ground truth OBB labels.
Saves a visualization image.

Usage:
    python scripts/eval_seg_obb.py [--model runs/segment/seg_cyl/weights/best.pt]
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def mask_to_obb(mask_xy):
    """Convert a polygon mask (Nx2 array) to a RotatedRect (cx, cy, w, h, angle)."""
    pts = mask_xy.astype(np.float32)
    if len(pts) < 5:
        rect = cv2.minAreaRect(pts)
    else:
        rect = cv2.minAreaRect(pts)
    return rect  # ((cx,cy), (w,h), angle)


def draw_rotated_rect(img, rect, color=(0, 255, 0), thickness=2):
    box = cv2.boxPoints(rect).astype(int)
    cv2.drawContours(img, [box], 0, color, thickness)
    return box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/segment/seg_cyl/weights/best.pt")
    ap.add_argument("--data", default="data/combined_seg_cyl/val/images")
    ap.add_argument("--out", default="seg_obb_eval.jpg")
    ap.add_argument("--max-imgs", type=int, default=12)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    model = YOLO(args.model, task="segment")
    img_dir = Path(args.data)
    img_files = sorted(img_dir.glob("*.jpg"))[:args.max_imgs]

    panels = []
    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        results = model.predict(img, conf=args.conf, verbose=False)
        vis = img.copy()

        if results and results[0].masks is not None:
            masks = results[0].masks
            for i, mask_xy in enumerate(masks.xy):
                if len(mask_xy) < 3:
                    continue
                # Fit OBB from the predicted mask
                rect = mask_to_obb(mask_xy)
                box = draw_rotated_rect(vis, rect, color=(0, 0, 255), thickness=2)

                # Draw mask outline
                pts = mask_xy.astype(int).reshape((-1, 1, 2))
                cv2.polylines(vis, [pts], True, (0, 255, 0), 1)

                # Print info
                (cx, cy), (w, h), angle = rect
                aspect = max(w, h) / max(min(w, h), 1)
                print(f"  {img_path.stem}: OBB {w:.0f}x{h:.0f} aspect={aspect:.2f} angle={angle:.1f}°")

        panels.append(vis)

    # Arrange in grid
    if not panels:
        print("No results!")
        return

    cols = min(4, len(panels))
    rows = (len(panels) + cols - 1) // cols
    h, w = panels[0].shape[:2]
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, panel in enumerate(panels):
        r, c = divmod(i, cols)
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = panel

    cv2.imwrite(args.out, grid)
    print(f"\nSaved {args.out} ({rows}x{cols} grid)")


if __name__ == "__main__":
    main()
