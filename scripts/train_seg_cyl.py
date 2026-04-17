#!/usr/bin/env python3
"""
Train YOLO-Seg model for cylinder segmentation.

Usage:
    python scripts/train_seg_cyl.py
    python scripts/train_seg_cyl.py --epochs 500 --resume
"""

import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description="Train YOLO-Seg for cylinder segmentation")
    ap.add_argument("--model", default="weights/yolo11s-seg.pt", help="Base model")
    ap.add_argument("--data", default="data/combined_seg_cyl/data.yaml", help="Dataset config")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.resume:
        model = YOLO("runs/segment/seg_cyl/weights/last.pt", task="segment")
    else:
        model = YOLO(args.model, task="segment")

    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=8,
        project="runs/segment",
        name="seg_cyl",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        seed=0,
        deterministic=True,
        close_mosaic=10,
        amp=True,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        degrees=45.0,
        translate=0.1,
        scale=0.5,
        shear=5.0,
        flipud=0.2,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.15,
        auto_augment="randaugment",
        erasing=0.4,
        plots=True,
        val=True,
    )
    print("Seg training complete!")


if __name__ == "__main__":
    main()
