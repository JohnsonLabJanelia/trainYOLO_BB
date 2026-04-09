# TrainYOLO — Cylinder Detection for Lime

Train a YOLO **detect** model for cylinder bounding-box detection, then convert to a TensorRT engine for use in [lime](../lime). Lime runs a two-stage pipeline: this YOLO detect model provides axis-aligned boxes, then an OBB refinement step adds orientation.

## Requirements

- Python 3.10+, [Ultralytics](https://github.com/ultralytics/ultralytics), PyTorch, OpenCV
- TensorRT (installed at `~/nvidia/TensorRT/`)
- GPU (trained/converted on NVIDIA RTX A6000)

```bash
conda create -n yolo python=3.10 && conda activate yolo && pip install ultralytics
```

## Project layout

```
TrainYOLO/
├── README.md
├── DETECT_ENGINE_CONVERSION.md   # TensorRT conversion details
├── scripts/
│   ├── pt_to_engine.py           # Convert .pt → .engine (via ONNX + trtexec)
│   └── compare_pt_engine.py      # Verify .pt vs .engine outputs match
├── weights/                      # Base pretrained models (yolo11s.pt, etc.)
├── obb_cyl/yolo_obb_dataset/     # Source dataset 1 (29 train / 8 val)
├── obb_cyl_2/yolo_obb_dataset/   # Source dataset 2 (17 train / 5 val)
├── obb_cyl_4/yolo_obb_dataset/   # Source dataset 4 (24 train / 7 val)
├── data/
│   └── combined_det_cyl/         # Merged training dataset (all sources, detect format)
├── runs/detect/det_cyl/          # Training output (best.pt, last.pt, results.csv)
└── engine/
    └── obb.engine                # Latest TensorRT engine
```

## End-to-end workflow

### 1. Add new annotations

Label new data with the Red annotation tool. Export produces a YOLO OBB dataset at `obb_cyl_N/yolo_obb_dataset/` with this structure:

```
obb_cyl_N/yolo_obb_dataset/
├── train/images/, train/labels/
├── val/images/, val/labels/
└── data.yaml
```

Labels are in **OBB format**: `class_id x1 y1 x2 y2 x3 y3 x4 y4` (4 normalized corners).

### 2. Convert OBB labels to detect format and merge

The training task is `detect` (axis-aligned boxes), so OBB labels must be converted to `class_id cx cy w h` format. Then copy into `data/combined_det_cyl/` with a unique prefix to avoid name collisions.

```python
# Convert one OBB label line to detect format
# "cls x1 y1 x2 y2 x3 y3 x4 y4" → "cls cx cy w h"
xs = [x1, x2, x3, x4]
ys = [y1, y2, y3, y4]
cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
w, h   = max(xs)-min(xs), max(ys)-min(ys)
```

Copy images and converted labels into `data/combined_det_cyl/{train,val}/{images,labels}/` with prefix `dsN_` (check existing prefixes first — `ds1`, `ds2`, `ds3_*`, `ds4_*`, `ds5`, etc.).

Verify the merge:
```bash
# Check all prefixes are present
ls data/combined_det_cyl/train/images/ | sed 's/_Cam.*//' | sort | uniq -c
```

### 3. Train

```python
from ultralytics import YOLO

model = YOLO("weights/yolo11s.pt", task="detect")
model.train(
    data="data/combined_det_cyl/data.yaml",
    epochs=800,
    patience=100,       # early stopping
    batch=16,
    imgsz=640,
    device="0",
    workers=8,
    project="runs/detect",
    name="det_cyl",
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
    scale=0.5,
    shear=5.0,
    flipud=0.2, fliplr=0.5,
    mosaic=0.5,
    mixup=0.15,
    auto_augment="randaugment",
    erasing=0.4,
    plots=True,
    val=True,
)
```

Output: `runs/detect/det_cyl/weights/best.pt`

Training typically early-stops around epoch 200-300 (~20-30 min on A6000).

### 4. Convert to TensorRT engine

```bash
python3 scripts/pt_to_engine.py \
    --pt runs/detect/det_cyl/weights/best.pt \
    --out engine/obb.engine \
    --device 0
```

This exports PT → ONNX → TensorRT engine (FP16). The old engine is automatically backed up. Takes ~15 seconds.

### 5. Deploy to lime

Lime loads the engine from the path in its camera config JSON (`"yolo"` field). All configs currently point to:

```
/home/ratan/yolo_model/obb.engine
```

To deploy:
```bash
# Backup the current engine
cp /home/ratan/yolo_model/obb.engine /home/ratan/yolo_model/obb.engine.bak_$(date +%F)

# Copy new engine
cp engine/obb.engine /home/ratan/yolo_model/obb.engine
```

Lime will use the new model on next launch.

## Dataset history

| Prefix | Source | Train | Val |
|--------|--------|-------|-----|
| ds1 | obb_cyl | 29 | 8 |
| ds2 | obb_cyl_2 | 17 | 5 |
| ds3_* | rotation augmentation (90/180/270) of ds1+ds2 | 184 | 52 |
| ds4_* | rotation augmentation (more angles) of ds1+ds2 | 460 | 130 |
| ds5 | obb_cyl_4 (Apr 2026) | 24 | 7 |
| **Total** | | **714** | **202** |

## data.yaml format

```yaml
names:
- class_0
nc: 1
path: /home/ratan/src/TrainYOLO/data/combined_det_cyl
train: train/images
val: val/images
```

Single class (`class_0` = cylinder). Detection labels: `class_id cx cy w h` (normalized 0-1).
