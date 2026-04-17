#!/usr/bin/env python3
"""
Generate YOLO-Seg polygon labels by auto-thresholding bright objects.

Takes a detect dataset (images + cx/cy/w/h labels) and creates a
segmentation dataset with precise polygon masks by thresholding the
brightest pixels within each detection box.

Usage:
    python scripts/gen_seg_labels.py \
        --src data/combined_det_cyl \
        --dst data/combined_seg_cyl \
        --percentile 85
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path


def generate_seg_labels(src: Path, dst: Path, percentile: float = 85):
    src_splits = {}
    for split in ["train", "val"]:
        img_dir = src / split / "images"
        lbl_dir = src / split / "labels"
        if img_dir.exists() and lbl_dir.exists():
            src_splits[split] = (img_dir, lbl_dir)

    if not src_splits:
        raise FileNotFoundError(f"No train/val splits found in {src}")

    stats = {}
    for split, (img_dir, lbl_dir) in src_splits.items():
        dst_img = dst / split / "images"
        dst_lbl = dst / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        n_imgs, n_objs, n_fail = 0, 0, 0

        for img_file in sorted(img_dir.iterdir()):
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl_file = lbl_dir / f"{img_file.stem}.txt"
            if not lbl_file.exists():
                continue

            img = cv2.imread(str(img_file))
            if img is None:
                continue
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            seg_lines = []

            for line in lbl_file.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = parts[0]
                cx_n = float(parts[1])
                cy_n = float(parts[2])
                bw_n = float(parts[3])
                bh_n = float(parts[4])

                # Expand crop by 2x for background context
                crop_w = int(bw_n * w * 2.0)
                crop_h = int(bh_n * h * 2.0)
                cx_px = int(cx_n * w)
                cy_px = int(cy_n * h)

                x1 = max(0, cx_px - crop_w // 2)
                y1 = max(0, cy_px - crop_h // 2)
                x2 = min(w, cx_px + crop_w // 2)
                y2 = min(h, cy_px + crop_h // 2)

                if x2 <= x1 or y2 <= y1:
                    n_fail += 1
                    continue

                patch = gray[y1:y2, x1:x2]
                blur = cv2.GaussianBlur(patch, (3, 3), 0)

                # Percentile threshold — isolate brightest pixels
                thresh_val = int(np.percentile(blur, percentile))
                _, mask = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not contours:
                    n_fail += 1
                    continue

                # Pick contour closest to patch center
                patch_cx = (x2 - x1) / 2.0
                patch_cy = (y2 - y1) / 2.0
                best_cnt = None
                best_dist = float("inf")
                for cnt in contours:
                    if cv2.contourArea(cnt) < 30:
                        continue
                    M = cv2.moments(cnt)
                    if M["m00"] < 1:
                        continue
                    ccx = M["m10"] / M["m00"]
                    ccy = M["m01"] / M["m00"]
                    d = np.sqrt((ccx - patch_cx) ** 2 + (ccy - patch_cy) ** 2)
                    if d < best_dist:
                        best_dist = d
                        best_cnt = cnt

                if best_cnt is None or len(best_cnt) < 3:
                    n_fail += 1
                    continue

                # Simplify contour
                peri = cv2.arcLength(best_cnt, True)
                approx = cv2.approxPolyDP(best_cnt, 0.01 * peri, True)
                if len(approx) < 3:
                    approx = best_cnt

                # Convert to normalized polygon label
                poly_parts = [cls]
                for pt in approx:
                    px = (pt[0][0] + x1) / w
                    py = (pt[0][1] + y1) / h
                    poly_parts.append(f"{px:.6f}")
                    poly_parts.append(f"{py:.6f}")

                seg_lines.append(" ".join(poly_parts))
                n_objs += 1

            if seg_lines:
                # Symlink image
                link = dst_img / img_file.name
                if not link.exists():
                    os.symlink(img_file.resolve(), link)
                # Write seg label
                (dst_lbl / f"{img_file.stem}.txt").write_text(
                    "\n".join(seg_lines) + "\n"
                )
                n_imgs += 1

        stats[split] = (n_imgs, n_objs, n_fail)

    # Write data.yaml
    yaml_content = f"""names:
- class_0
nc: 1
path: {dst.resolve()}
train: train/images
val: val/images
"""
    (dst / "data.yaml").write_text(yaml_content)

    print("=== Seg dataset created ===")
    for split, (ni, no, nf) in stats.items():
        print(f"  {split}: {ni} images, {no} objects, {nf} failed")


def main():
    ap = argparse.ArgumentParser(
        description="Generate YOLO-Seg labels from detect dataset via brightness thresholding"
    )
    ap.add_argument(
        "--src",
        default="data/combined_det_cyl",
        help="Source detect dataset (with train/val splits)",
    )
    ap.add_argument(
        "--dst",
        default="data/combined_seg_cyl",
        help="Output seg dataset directory",
    )
    ap.add_argument(
        "--percentile",
        type=float,
        default=85,
        help="Brightness percentile for thresholding (default: 85)",
    )
    args = ap.parse_args()
    generate_seg_labels(Path(args.src), Path(args.dst), args.percentile)


if __name__ == "__main__":
    main()
