import json
import os
from ultralytics import YOLO

# =========================
# CONFIGURATION
# =========================
DATA_YAML = "dataset/data.yaml"
OUTPUT_DIR = "runs/train"
BASE_MODEL = "yolov8s.pt"

EPOCHS = 10
IMGSZ = 640
BATCH = 16

TARGET_PRECISION = 0.82
TARGET_RECALL = 0.78
TARGET_F1 = 0.80


# =========================
# TRAINING
# =========================
def train_model(
    model_name: str,
    run_name: str,
    epochs: int,
    augment: bool = False,
    lr0: float = None
):
    print(f"\nTraining {run_name}")
    print(f"Model={model_name} | epochs={epochs} | augment={augment}")

    model = YOLO(model_name)

    train_args = {
        "data": DATA_YAML,
        "epochs": epochs,
        "imgsz": IMGSZ,
        "batch": BATCH,
        "project": OUTPUT_DIR,
        "name": run_name,
        "exist_ok": True,
    }

    if augment:
        train_args["augment"] = True

    if lr0 is not None:
        train_args["lr0"] = lr0

    results = model.train(**train_args)
    best_weight = os.path.join(results.save_dir, "weights", "best.pt")

    return best_weight


# =========================
# EVALUATION
# =========================
def evaluate_model(weights_path: str):
    print(f"\nEvaluating: {weights_path}")

    model = YOLO(weights_path)
    results = model.val(
        data=DATA_YAML,
        imgsz=IMGSZ,
        batch=BATCH
    )

    precision = float(results.box.mp)
    recall = float(results.box.mr)
    map50 = float(results.box.map50)
    map50_95 = float(results.box.map)

    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    # pseudo accuracy (for reporting only)
    accuracy = (precision + recall) / 2

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "f1": f1
    }

    return metrics


# =========================
# UTILITIES
# =========================
def print_metrics(metrics: dict, title: str):
    print(f"\n===== {title} =====")
    print(f"Accuracy   : {metrics['accuracy']*100:.2f}%")
    print(f"Precision  : {metrics['precision']*100:.2f}%")
    print(f"Recall     : {metrics['recall']*100:.2f}%")
    print(f"mAP50      : {metrics['mAP50']*100:.2f}%")
    print(f"mAP50-95   : {metrics['mAP50_95']*100:.2f}%")
    print(f"F1-score   : {metrics['f1']*100:.2f}%")


def save_metrics(metrics: dict, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)


def meets_target(metrics: dict):
    return (
        metrics["precision"] >= TARGET_PRECISION
        and metrics["recall"] >= TARGET_RECALL
        and metrics["f1"] >= TARGET_F1
    )


# =========================
# MAIN PIPELINE
# =========================
def main():
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"{DATA_YAML} not found")

    print("=== YOLOv8 Oral Disease Detection ===")

    # -------------------------
    # Baseline Training
    # -------------------------
    baseline_weight = train_model(
        model_name=BASE_MODEL,
        run_name="baseline",
        epochs=EPOCHS
    )

    baseline_metrics = evaluate_model(baseline_weight)
    print_metrics(baseline_metrics, "BASELINE RESULTS")

    save_metrics(
        baseline_metrics,
        os.path.join(OUTPUT_DIR, "baseline", "metrics.json")
    )

    if meets_target(baseline_metrics):
        print("\n✅ Target tercapai pada baseline.")
        return

    # -------------------------
    # Tuned Training
    # -------------------------
    print("\nTarget belum tercapai. Melakukan tuning...")

    tuned_weight = train_model(
        model_name=BASE_MODEL,
        run_name="tuned",
        epochs=min(EPOCHS + 20, 100),
        augment=True,
        lr0=0.01
    )

    tuned_metrics = evaluate_model(tuned_weight)
    print_metrics(tuned_metrics, "TUNED RESULTS")

    save_metrics(
        tuned_metrics,
        os.path.join(OUTPUT_DIR, "tuned", "metrics.json")
    )

    if meets_target(tuned_metrics):
        print("\n✅ Target tercapai setelah tuning.")
    else:
        print("\n❌ Target belum tercapai.")
        print("Saran:")
        print("- Tambah epoch hingga 100")
        print("- Gunakan yolov8m.pt")
        print("- Tambah augmentasi")
        print("- Perbaiki kualitas dataset")


if __name__ == "__main__":
    main()