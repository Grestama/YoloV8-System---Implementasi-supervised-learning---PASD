"""
Utility functions untuk YOLOv8 Inference
"""
import os
import time
import random
from datetime import datetime
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO

import config


class DetectionResult:
    def __init__(self, class_id: int, class_name: str, confidence: float, bbox: List[float]):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox

    def to_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "bbox": [float(x) for x in self.bbox]
        }


# =========================
# HELPER
# =========================
def _get_class_name(class_id: int) -> str:
    if isinstance(config.CLASS_NAMES, dict):
        return config.CLASS_NAMES.get(class_id, "unknown")
    elif isinstance(config.CLASS_NAMES, list):
        return config.CLASS_NAMES[class_id] if class_id < len(config.CLASS_NAMES) else "unknown"
    return "unknown"


def load_model(model_path: str = None) -> YOLO:
    model_path = model_path or config.MODEL_PATH

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    return YOLO(model_path)


def parse_yolo_results(results) -> List[DetectionResult]:
    detections = []

    if not results or len(results) == 0:
        return detections

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return detections

    boxes = result.boxes

    if boxes.xyxy is None or len(boxes.xyxy) == 0:
        return detections

    xyxy = boxes.xyxy
    conf = boxes.conf
    cls = boxes.cls

    if hasattr(xyxy, "cpu"):
        xyxy = xyxy.cpu().numpy()
        conf = conf.cpu().numpy()
        cls = cls.cpu().numpy()

    for box, score, class_id in zip(xyxy, conf, cls):
        class_id = int(class_id)

        detections.append(
            DetectionResult(
                class_id=class_id,
                class_name=_get_class_name(class_id),
                confidence=float(score),
                bbox=[float(x) for x in box]
            )
        )

    return detections


def draw_detections_on_image(image: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
    annotated = image.copy()
    h, w = image.shape[:2]

    for det in detections:
        x1 = max(0, min(w, int(det.bbox[0])))
        y1 = max(0, min(h, int(det.bbox[1])))
        x2 = max(0, min(w, int(det.bbox[2])))
        y2 = max(0, min(h, int(det.bbox[3])))

        color = config.CLASS_COLORS.get(det.class_id, (0, 255, 0))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{det.class_name} {det.confidence*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        y_text = max(y1 - 10, th)

        cv2.rectangle(annotated, (x1, y_text - th), (x1 + tw, y_text), color, -1)
        cv2.putText(
            annotated, label, (x1, y_text - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

    return annotated


def load_image(image_path: str) -> np.ndarray:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    return image


def save_image(image: np.ndarray, output_path: str) -> None:
    dir_name = os.path.dirname(output_path)

    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    if not cv2.imwrite(output_path, image):
        raise IOError(f"Failed to save image: {output_path}")


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0, (x1_max - x1_min)) * max(0, (y1_max - y1_min))
    area2 = max(0, (x2_max - x2_min)) * max(0, (y2_max - y2_min))

    union = area1 + area2 - inter_area

    return inter_area / union if union > 0 else 0.0


def get_random_validation_images(num_images: int = 5) -> List[str]:
    data_yaml_path = os.path.abspath(config.DATA_YAML)
    base_dir = os.path.dirname(data_yaml_path)
    val_dir = os.path.join(base_dir, "images", "val")

    if not os.path.isdir(val_dir):
        return []

    images = [
        os.path.join(val_dir, f)
        for f in os.listdir(val_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)
    return images[:num_images]


def resize_image(image: np.ndarray, max_width=1280, max_height=720) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)

    if scale >= 1:
        return image

    return cv2.resize(image, (int(w * scale), int(h * scale)))


def get_timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_summary_text(detections: List[DetectionResult]) -> str:
    if not detections:
        return "Tidak ada penyakit terdeteksi"

    summary = {}

    for d in detections:
        summary[d.class_name] = summary.get(d.class_name, 0) + 1

    lines = ["Hasil Deteksi:"]
    for k, v in summary.items():
        confs = [d.confidence for d in detections if d.class_name == k]
        avg = sum(confs) / len(confs)
        lines.append(f"- {k}: {v}x ({avg*100:.1f}%)")

    return "\n".join(lines)


class PerformanceTimer:
    def __init__(self):
        self.times = {}

    def start(self, name: str):
        self.times[name] = {"start": time.time(), "end": None}

    def end(self, name: str):
        if name in self.times:
            self.times[name]["end"] = time.time()

    def get_elapsed(self, name: str) -> float:
        t = self.times.get(name)
        if not t or t["end"] is None:
            return 0.0
        return t["end"] - t["start"]