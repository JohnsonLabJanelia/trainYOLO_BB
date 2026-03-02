#!/usr/bin/env python3
"""
Quick parity check between detect .pt and TensorRT .engine outputs.

Compares detection count/class/confidence on the same images.
"""

import argparse
import os
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def letterbox_to_bin(img_path: str, out_path: str, imgsz: int = 640) -> None:
    im = cv2.imread(img_path)
    if im is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    h, w = im.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = imgsz - nw, imgsz - nh
    top, bottom = int(round(dh / 2 - 0.1)), int(round(dh / 2 + 0.1))
    left, right = int(round(dw / 2 - 0.1)), int(round(dw / 2 + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    arr = im[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    arr = np.expand_dims(np.ascontiguousarray(arr), 0)
    arr.tofile(out_path)


def parse_engine_rows(trt_output: str) -> list[list[float]]:
    m = re.search(r"output0: \(1x(\d+)x6\)\n\[[^\]]+\] \[I\] (.+?)\n&&&& PASSED", trt_output, re.S)
    if not m:
        return []
    vals = [float(x) for x in m.group(2).split()]
    return [vals[i : i + 6] for i in range(0, len(vals), 6) if len(vals[i : i + 6]) == 6]


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare .pt vs .engine detect outputs")
    ap.add_argument("--pt", required=True, help="Path to detect .pt")
    ap.add_argument("--engine", required=True, help="Path to TensorRT .engine")
    ap.add_argument("--images", nargs="+", required=True, help="Image paths to compare")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for counts")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    ap.add_argument("--device", default="0", help="GPU device id for PT predict")
    ap.add_argument("--trtexec", default="/home/ratan/nvidia/TensorRT/bin/trtexec", help="Path to trtexec")
    ap.add_argument("--trt-lib", default="/home/ratan/nvidia/TensorRT/lib", help="TensorRT lib path")
    args = ap.parse_args()

    pt = YOLO(args.pt, task="detect")
    tmp_dir = Path("/tmp/trt_cmp_inputs")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for img in args.images:
        r = pt.predict(img, imgsz=args.imgsz, device=args.device, conf=args.conf, iou=0.7, verbose=False)[0]
        pt_rows = []
        if r.boxes is not None and len(r.boxes) > 0:
            xy = r.boxes.xyxy.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            cl = r.boxes.cls.cpu().numpy()
            for i in range(len(r.boxes)):
                if cf[i] >= args.conf:
                    pt_rows.append([xy[i][0], xy[i][1], xy[i][2], xy[i][3], float(cf[i]), float(cl[i])])

        bin_path = str(tmp_dir / (Path(img).name + ".bin"))
        letterbox_to_bin(img, bin_path, imgsz=args.imgsz)
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = args.trt_lib + (":" + env.get("LD_LIBRARY_PATH", "") if env.get("LD_LIBRARY_PATH") else "")
        cmd = [
            args.trtexec,
            f"--loadEngine={os.path.abspath(args.engine)}",
            f"--loadInputs=images:{bin_path}",
            "--warmUp=0",
            "--iterations=1",
            "--duration=0",
            "--dumpOutput",
            "--avgRuns=1",
        ]
        p = subprocess.run(cmd, text=True, capture_output=True, env=env)
        engine_rows = [row for row in parse_engine_rows(p.stdout + p.stderr) if row[4] >= args.conf]

        print(f"\nIMAGE {img}")
        print("PT ", len(pt_rows), [[round(float(v), 4) for v in row] for row in pt_rows[:5]])
        print("ENG", len(engine_rows), [[round(float(v), 4) for v in row] for row in engine_rows[:5]])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
