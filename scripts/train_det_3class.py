#!/usr/bin/env python3
"""Train the 3-class detect model (Mouse / SideCyl / VertCyl).

Dataset: data/combined_det_3class (remus_bb + kimchi_bb + wasabi_bb).
Output: runs/detect/det_3class/weights/best.pt

device="0" is the RTX A6000 (torch default fastest-first ordering).
"""

from ultralytics import YOLO


def main():
    model = YOLO("yolo11m.pt", task="detect")   # bigger backbone (was yolo11s)
    model.train(
        data="data/combined_det_2class_mirror/data.yaml",   # 2-class (Mouse+SideCyl), 4-way mirrored
        epochs=800,
        patience=100,
        batch=24,           # yolo11m @ 960 on the 49 GB A6000 (device 0 in-process) -> ~27 GB, fast
        imgsz=960,          # 2.25x mouse pixels vs 640 -> the big lever for small-mouse detection
        device="0",         # torch CUDA:0 == the RTX A6000 (49 GB) in this process
        workers=12,
        project="runs/detect",
        name="det_2class_960_m",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        seed=0,
        deterministic=True,
        close_mosaic=10,
        amp=True,
        lr0=0.01, lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5, cls=0.5, dfl=1.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=45.0,
        translate=0.1,
        scale=0.5,          # full aug -> most robust real-world DETECTION (what we want)
        shear=5.0,
        flipud=0.2, fliplr=0.5,
        mosaic=0.5,
        mixup=0.15,
        auto_augment="randaugment",
        erasing=0.4,
        plots=True,
        val=True,
    )


if __name__ == "__main__":
    main()
