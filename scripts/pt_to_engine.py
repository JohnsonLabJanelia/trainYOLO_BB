#!/usr/bin/env python3
"""
Convert a YOLO detect .pt model to TensorRT .engine.

Flow:
1) PT -> ONNX (Ultralytics)
2) ONNX -> ENGINE (trtexec)
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


def find_trtexec(explicit_path: str | None) -> str:
    if explicit_path:
        p = Path(explicit_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"trtexec not found: {p}")
        return str(p)

    preferred = Path("/home/ratan/nvidia/TensorRT/bin/trtexec")
    if preferred.is_file():
        return str(preferred)

    p = shutil.which("trtexec")
    if p:
        return p

    raise FileNotFoundError("Could not find trtexec. Set --trtexec.")


def run_cmd(cmd: list[str], env: dict[str, str]) -> None:
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert YOLO detect .pt -> TensorRT .engine")
    ap.add_argument("--pt", required=True, help="Path to input .pt model")
    ap.add_argument("--out", required=True, help="Path to output .engine model")
    ap.add_argument("--device", default="0", help="GPU device id (default: 0)")
    ap.add_argument("--imgsz", type=int, default=640, help="Export image size (default: 640)")
    ap.add_argument("--trtexec", default=None, help="Path to trtexec binary")
    ap.add_argument("--trt-lib", default="/home/ratan/nvidia/TensorRT/lib", help="TensorRT lib path")
    ap.add_argument("--builder-opt-level", type=int, default=0, help="trtexec builder opt level (default: 0)")
    ap.add_argument("--max-aux-streams", type=int, default=0, help="trtexec max aux streams (default: 0)")
    ap.add_argument("--fp32", action="store_true", help="Build FP32 engine (default: FP16)")
    ap.add_argument("--keep-onnx", action="store_true", help="Keep generated ONNX file")
    args = ap.parse_args()

    pt_path = Path(args.pt).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    if not pt_path.is_file():
        raise FileNotFoundError(f"PT model not found: {pt_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        backup = out_path.with_name(f"{out_path.name}.bak_{ts}")
        shutil.copy2(out_path, backup)
        print(f"Backed up existing engine to: {backup}")

    trtexec = find_trtexec(args.trtexec)

    print(f"Exporting ONNX from: {pt_path}")
    model = YOLO(str(pt_path), task="detect")
    # Export with NMS so engine output is final detect boxes (Nx6), not raw head tensors.
    onnx_out = model.export(
        format="onnx",
        imgsz=args.imgsz,
        simplify=True,
        opset=20,
        device=args.device,
        nms=True,
    )
    onnx_path = Path(str(onnx_out)).expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX export failed: {onnx_path}")
    print(f"ONNX exported: {onnx_path}")

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={out_path}",
        f"--device={args.device}",
        f"--builderOptimizationLevel={args.builder_opt_level}",
        f"--maxAuxStreams={args.max_aux_streams}",
        "--skipInference",
    ]
    if not args.fp32:
        cmd.append("--fp16")

    env = dict(os.environ)
    old_ld = env.get("LD_LIBRARY_PATH", "")
    trt_lib = str(Path(args.trt_lib).expanduser().resolve())
    env["LD_LIBRARY_PATH"] = f"{trt_lib}:{old_ld}" if old_ld else trt_lib
    run_cmd(cmd, env)

    if not out_path.is_file():
        raise FileNotFoundError(f"Engine build reported success but file is missing: {out_path}")

    print(f"Engine created: {out_path} ({out_path.stat().st_size / (1024*1024):.2f} MB)")

    if not args.keep_onnx:
        try:
            onnx_path.unlink()
            print(f"Removed ONNX: {onnx_path}")
        except OSError:
            print(f"Warning: could not remove ONNX: {onnx_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
