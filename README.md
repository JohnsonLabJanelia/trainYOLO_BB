# TrainYOLO — YOLO OBB (oriented bounding box) training

Train YOLO models for **oriented bounding box (OBB)** detection (e.g. cylinders). This repo includes dataset merging, rotation augmentation, training, and evaluation scripts.

## Requirements

- Python 3.10+
- [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLO), PyTorch, OpenCV
- Conda: `conda create -n yolo python=3.10 && conda activate yolo && pip install ultralytics`

## Project layout

```
TrainYOLO/
├── README.md                 # This file
├── train.py                  # Legacy training entry (uses data/combined_obb_cyl)
├── data/
│   ├── README.md             # Data folder summary
│   ├── combined_obb_cyl/      # Merged dataset used for training (symlinks + data.yaml)
│   ├── combined_obb_cyl_rotated/    # Output of rotation augmentation (90/180/270)
│   ├── combined_obb_cyl_rotated_fine/ # Output of fine-angle augmentation (15°–165°)
│   └── obb_cyl_video/        # Optional: from video_obb_to_yolo_dataset.py
├── scripts/
│   ├── README_DATA_PIPELINE.md  # Full pipeline and script reference
│   ├── merge_obb_cyl_datasets.py # Merge datasets into combined_obb_cyl
│   ├── augment_obb_rotate.py    # Rotate images + labels (more angles)
│   ├── video_obb_to_yolo_dataset.py # Videos → YOLO OBB (bg-subtraction)
│   ├── train_obb_cyl.py      # Main training script
│   ├── test_obb_cyl.py       # Validate + sample prediction
│   ├── clean_broken_symlinks.py # Remove broken symlinks in combined dir
│   ├── matlab_obb_to_yolo.py # Bg-subtraction OBB → labels (used by video script)
│   └── ...                  # vis.py, visxywhr.py, eval.py, etc.
├── weights/                  # Pretrained .pt (yolo11n-obb.pt, etc.)
└── runs/obb/obb_cyl/         # Training outputs (best.pt, last.pt, results, etc.)
```

## Dataset sources

Place YOLO OBB datasets at the project root:

- **obb_cyl/yolo_obb_dataset/** — `train/images`, `train/labels`, `val/images`, `val/labels`, `data.yaml`
- **obb_cyl_2/yolo_obb_dataset/** — same structure

Labels are YOLO OBB format: one line per object, `class_id x1 y1 x2 y2 x3 y3 x4 y4` (normalized 0–1, four polygon corners).

## Quick start

From project root with conda env active:

```bash
# 1. Merge base datasets into data/combined_obb_cyl
python scripts/merge_obb_cyl_datasets.py

# 2. (Optional) Add rotation augmentation, then merge again with --clear
python scripts/augment_obb_rotate.py --dataset data/combined_obb_cyl --out data/combined_obb_cyl_rotated --angles 90,180,270 --split both
python scripts/merge_obb_cyl_datasets.py --clear --sources obb_cyl/yolo_obb_dataset obb_cyl_2/yolo_obb_dataset data/combined_obb_cyl_rotated --out data/combined_obb_cyl

# 3. Train (800 epochs by default)
python scripts/train_obb_cyl.py

# 4. Test
python scripts/test_obb_cyl.py --no-val
# Opens/saves test_pred_obb_cyl.jpg
```

## Training output

- **Weights:** `runs/obb/obb_cyl/weights/best.pt`, `last.pt`
- **Metrics/logs:** `runs/obb/obb_cyl/results.csv`, `results.png`, `args.yaml`
- Resume: `python scripts/train_obb_cyl.py --resume --epochs 800`

## Broken symlinks

If you delete or move source datasets (e.g. rotated folders) after merging, symlinks in `data/combined_obb_cyl` can break and training will report "No such file or directory". Fix:

```bash
python scripts/clean_broken_symlinks.py data/combined_obb_cyl
```

Then re-merge with `--clear` if you are repopulating from current sources.

## More detail

- **Data pipeline and all script options:** [scripts/README_DATA_PIPELINE.md](scripts/README_DATA_PIPELINE.md)
- **C++ OBB detector (real-time):** [OBB_DETECTOR_README.md](OBB_DETECTOR_README.md)
# TRAIN YOLO
