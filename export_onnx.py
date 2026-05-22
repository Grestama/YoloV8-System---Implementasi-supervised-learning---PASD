import os
from ultralytics import YOLO
import config

EXPORT_DIR = os.path.join(config.BASE_DIR, "onnx")

os.makedirs(EXPORT_DIR, exist_ok=True)

config.validate_paths(raise_error=True)

model = YOLO(config.MODEL_PATH)

print(f"Exporting {config.MODEL_PATH} → ONNX...")

exported = model.export(
    format="onnx",
    imgsz=config.IMG_SIZE,
    dynamic=True,
    opset=12,
    simplify=True,
    project=EXPORT_DIR,
    name="model",
)

print("✅ Export completed:", exported)