import argparse
import os
import random
import sys

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = os.path.join("runs", "train", "yolov8s_tuned", "weights", "best.pt")
SAMPLE_DIR = os.path.join("dataset", "images", "val")
CLASS_NAMES = {0: "Calculus", 1: "Caries", 2: "Gingivitis", 3: "Hypodontia"}


def load_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH}")
    print(f"Loading model dari {MODEL_PATH}...")
    return YOLO(MODEL_PATH)


def sample_images(sample_dir, limit=5):
    if not os.path.isdir(sample_dir):
        raise FileNotFoundError(f"Sample directory tidak ditemukan: {sample_dir}")

    image_files = [f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not image_files:
        raise FileNotFoundError("Tidak ada file gambar di sample directory")

    random.shuffle(image_files)
    return [os.path.join(sample_dir, f) for f in image_files[:limit]]


def run_local_inference(model, image_paths):
    print("\nRunning local inference:")
    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            print(f"  [SKIP] cannot load image {path}")
            continue

        results = model(image)
        if not results:
            print(f"  [OK] {os.path.basename(path)} -> no detections")
            continue

        detections = []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
            conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
            cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls
            for box, score, class_id in zip(xyxy, conf, cls):
                detections.append({
                    "class_name": CLASS_NAMES.get(int(class_id), "unknown"),
                    "confidence": float(score),
                    "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                })

        print(f"  {os.path.basename(path)} -> {len(detections)} detections")
        for det in detections:
            print(f"    - {det['class_name']} {det['confidence']:.3f} bbox={det['bbox']}")


def call_api(image_path: str, url: str):
    import requests

    with open(image_path, "rb") as f:
        files = {"image": (os.path.basename(image_path), f, "image/jpeg")}
        response = requests.post(url, files=files, timeout=30)
        response.raise_for_status()
        return response.json()


def main():
    parser = argparse.ArgumentParser(description="Internal test untuk YOLOv8 inference dan API")
    parser.add_argument("--api-url", type=str, help="URL backend FastAPI untuk uji API (opsional)")
    parser.add_argument("--limit", type=int, default=5, help="Jumlah sample gambar yang diuji")
    args = parser.parse_args()

    model = load_model()
    paths = sample_images(SAMPLE_DIR, limit=args.limit)
    run_local_inference(model, paths)

    if args.api_url:
        print(f"\nMengirim ke API {args.api_url}...")
        for path in paths:
            result = call_api(path, args.api_url)
            print(f"  {os.path.basename(path)} -> {len(result.get('detections', []))} detections")

    print("\nInternal test selesai.")


if __name__ == "__main__":
    main()
