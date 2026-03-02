# Convert `best.pt` (Detect) to TensorRT Engine (GPU 0)

## Quick command (recommended)

```bash
python3 scripts/pt_to_engine.py --pt runs/detect/det_cyl/weights/best.pt --out engine/obb.engine --device 0
```

This uses the same validated flow:
1) `best.pt` -> `best.onnx`
2) `best.onnx` -> `engine/obb.engine` via TensorRT `trtexec`

The script also:
- Backs up existing output engine before overwrite
- Uses stable TensorRT flags (`builderOptimizationLevel=0`, `maxAuxStreams=0`, `skipInference`)

## Manual equivalent

```bash
# PT -> ONNX
python3 - <<'PY'
from ultralytics import YOLO
YOLO("runs/detect/det_cyl/weights/best.pt", task="detect").export(
    format="onnx", imgsz=640, simplify=True, opset=20, device=0
)
PY

# ONNX -> ENGINE
LD_LIBRARY_PATH=/home/ratan/nvidia/TensorRT/lib:$LD_LIBRARY_PATH \
/home/ratan/nvidia/TensorRT/bin/trtexec \
  --onnx=/home/ratan/src/TrainYOLO/runs/detect/det_cyl/weights/best.onnx \
  --saveEngine=/home/ratan/src/TrainYOLO/engine/obb.engine \
  --device=0 \
  --fp16 \
  --builderOptimizationLevel=0 \
  --maxAuxStreams=0 \
  --skipInference
```
