"""
Model Evaluation Report Generator
"""
import os
import json
import csv
import traceback
from datetime import datetime
from typing import Dict, Any

from ultralytics import YOLO

import config
import utils


def evaluate_model(model_path: str = None, data_yaml: str = None) -> Dict[str, Any]:
    if model_path is None:
        model_path = config.MODEL_PATH
    if data_yaml is None:
        data_yaml = config.DATA_YAML

    print(f"\n{'='*60}")
    print("EVALUATING MODEL")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Dataset: {data_yaml}")

    # ✅ Validate paths
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not os.path.isfile(data_yaml):
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

    # Load model
    try:
        model = utils.load_model(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

    try:
        print("\nRunning validation...")

        # ✅ FIX: tambahkan conf & iou
        results = model.val(
            data=data_yaml,
            imgsz=config.IMG_SIZE_TRAIN,
            batch=config.BATCH_SIZE,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            verbose=False
        )

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "data_yaml": data_yaml,
            "metrics": {}
        }

        # Extract metrics
        if hasattr(results, 'box'):
            box = results.box
            metrics["metrics"]["precision"] = float(getattr(box, "mp", 0))
            metrics["metrics"]["recall"] = float(getattr(box, "mr", 0))
            metrics["metrics"]["mAP50"] = float(getattr(box, "map50", 0))
            metrics["metrics"]["mAP50_95"] = float(getattr(box, "map", 0))

        # F1 Score
        precision = metrics["metrics"].get("precision", 0)
        recall = metrics["metrics"].get("recall", 0)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        metrics["metrics"]["f1_score"] = f1

        # ✅ gunakan config (fallback kalau belum ada)
        precision_target = getattr(config, "PRECISION_TARGET", 0.82)
        recall_target = getattr(config, "RECALL_TARGET", 0.78)
        f1_target = getattr(config, "F1_TARGET", 0.80)

        metrics["target_achievement"] = {
            "precision_target": precision_target,
            "precision_achieved": precision,
            "precision_met": precision >= precision_target,
            "recall_target": recall_target,
            "recall_achieved": recall,
            "recall_met": recall >= recall_target,
            "f1_target": f1_target,
            "f1_achieved": f1,
            "f1_met": f1 >= f1_target
        }

        metrics["all_targets_met"] = (
            precision >= precision_target and
            recall >= recall_target and
            f1 >= f1_target
        )

        print("✅ Validation completed")
        return metrics

    except Exception as e:
        print(f"❌ Error during validation: {e}")
        traceback.print_exc()
        return None


def print_metrics_summary(metrics: Dict[str, Any]):
    if metrics is None:
        return

    print(f"\n{'='*60}")
    print("METRICS SUMMARY")
    print(f"{'='*60}")

    m = metrics["metrics"]
    t = metrics["target_achievement"]

    print(f"\nPerformance Metrics:")
    print(f"  Precision:  {m.get('precision', 0)*100:.2f}%")
    print(f"  Recall:     {m.get('recall', 0)*100:.2f}%")
    print(f"  F1-Score:   {m.get('f1_score', 0)*100:.2f}%")
    print(f"  mAP50:      {m.get('mAP50', 0)*100:.2f}%")
    print(f"  mAP50-95:   {m.get('mAP50_95', 0)*100:.2f}%")

    print(f"\nTarget Achievement:")
    print(f"  {'✅' if t.get('precision_met') else '❌'} Precision")
    print(f"  {'✅' if t.get('recall_met') else '❌'} Recall")
    print(f"  {'✅' if t.get('f1_met') else '❌'} F1-Score")

    print("\n🎯 ALL TARGETS MET!" if metrics.get("all_targets_met") else "\n⚠️ Some targets not met")
    print(f"{'='*60}\n")


def save_metrics_json(metrics: Dict[str, Any], output_path: str = None):
    if output_path is None:
        output_path = os.path.join(
            config.METRICS_DIR,
            f"evaluation_report_{utils.get_timestamp_string()}.json"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"💾 JSON saved: {output_path}")
    return output_path


def save_metrics_csv(metrics: Dict[str, Any], output_path: str = None):
    if output_path is None:
        output_path = os.path.join(
            config.METRICS_DIR,
            f"evaluation_report_{utils.get_timestamp_string()}.csv"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value", "Target", "Status"])

        m = metrics["metrics"]
        t = metrics["target_achievement"]

        rows = [
            ("Precision", f"{m.get('precision',0)*100:.2f}%", f"{t.get('precision_target',0)*100:.0f}%", "✅" if t.get("precision_met") else "❌"),
            ("Recall", f"{m.get('recall',0)*100:.2f}%", f"{t.get('recall_target',0)*100:.0f}%", "✅" if t.get("recall_met") else "❌"),
            ("F1-Score", f"{m.get('f1_score',0)*100:.2f}%", f"{t.get('f1_target',0)*100:.0f}%", "✅" if t.get("f1_met") else "❌"),
            ("mAP50", f"{m.get('mAP50',0)*100:.2f}%", "N/A", "✅"),
            ("mAP50-95", f"{m.get('mAP50_95',0)*100:.2f}%", "N/A", "✅"),
        ]

        writer.writerows(rows)

    print(f"💾 CSV saved: {output_path}")
    return output_path


def generate_html_report(metrics: Dict[str, Any], output_path: str = None):
    if output_path is None:
        output_path = os.path.join(
            config.METRICS_DIR,
            f"evaluation_report_{utils.get_timestamp_string()}.html"
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("<h1>YOLOv8 Evaluation Report</h1>")

    print(f"💾 HTML saved: {output_path}")
    return output_path


def main():
    metrics = evaluate_model()

    if metrics is None:
        print("❌ Evaluation failed")
        return

    print_metrics_summary(metrics)
    save_metrics_json(metrics)
    save_metrics_csv(metrics)
    generate_html_report(metrics)

if __name__ == "__main__":
    main()