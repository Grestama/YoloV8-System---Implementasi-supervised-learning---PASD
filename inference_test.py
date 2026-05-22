"""
Inference Testing - YOLOv8
"""
import os
import json
from pathlib import Path

from tqdm import tqdm

import config
import utils


# =====================================
# INIT
# =====================================
config.create_dirs()


# =====================================
# CORE
# =====================================
def load_model():
    model = utils.load_model()

    # warmup
    import numpy as np
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model(dummy, verbose=False)

    return model


def test_single_image(image_path: str, model) -> dict:
    print(f"\nTesting: {image_path}")

    image = utils.load_image(image_path)

    timer = utils.PerformanceTimer()
    timer.start("inf")

    results = model(image, conf=config.CONFIDENCE_THRESHOLD, verbose=False)

    timer.end("inf")

    detections = utils.parse_yolo_results(results)
    t = timer.get_elapsed("inf")

    annotated = utils.draw_detections_on_image(image, detections)

    out_path = os.path.join(
        config.TEST_OUTPUT_DIR,
        f"single_{utils.get_timestamp_string()}.jpg"
    )
    utils.save_image(annotated, out_path)

    return {
        "image": image_path,
        "detections": [d.to_dict() for d in detections],
        "time_ms": t * 1000,
        "output": out_path
    }


def test_batch(images, model):
    results = []
    total_time = 0

    for p in tqdm(images):
        try:
            img = utils.load_image(p)

            timer = utils.PerformanceTimer()
            timer.start("inf")

            r = model(img, conf=config.CONFIDENCE_THRESHOLD, verbose=False)

            timer.end("inf")

            det = utils.parse_yolo_results(r)
            t = timer.get_elapsed("inf")

            total_time += t

            results.append({
                "image": p,
                "n": len(det),
                "time_ms": t * 1000
            })

        except Exception as e:
            results.append({"image": p, "error": str(e)})

    return {
        "total": len(results),
        "avg_time_ms": (total_time / len(results) * 1000) if results else 0,
        "results": results
    }


# =====================================
# MAIN
# =====================================
def main():
    print("=== INFERENCE TEST ===")

    model = load_model()

    # ambil sample validation
    val_images = utils.get_random_validation_images(5)

    if not val_images:
        print("No validation images found")
        return

    # single test
    single = test_single_image(val_images[0], model)

    # batch test
    batch = test_batch(val_images, model)

    # save
    out = {
        "single": single,
        "batch": batch
    }

    path = os.path.join(config.TEST_OUTPUT_DIR, "test_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved: {path}")


if __name__ == "__main__":
    main()