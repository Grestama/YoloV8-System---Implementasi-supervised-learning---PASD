"""
Streamlit Web App - YOLOv8 Oral Disease Detection (API-based)
"""
import io
import os
import requests

import numpy as np
import streamlit as st
from PIL import Image

import config
import utils


# =====================================
# CONFIG (FIX: Docker-safe API URL)
# =====================================
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
HEALTH_URL = API_URL.replace("/predict", "/health")

st.set_page_config(
    page_title="YOLOv8 Oral Disease Detection",
    page_icon="🦷",
    layout="wide"
)


# =====================================
# CHECK API
# =====================================
def check_api():
    try:
        r = requests.get(HEALTH_URL, timeout=3)
        return r.status_code == 200
    except:
        return False


# =====================================
# MAIN
# =====================================
def main():
    st.title("🦷 Oral Disease Detection")

    # ✅ API status
    if not check_api():
        st.error("API not reachable")
        return
    else:
        st.success("API Connected")

    conf = st.slider("Confidence", 0.0, 1.0, config.CONFIDENCE_THRESHOLD, 0.05)

    file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if file:
        if file.size > config.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            st.error("File too large")
            return

        try:
            bytes_data = file.read()
            image = Image.open(io.BytesIO(bytes_data)).convert("RGB")
            image_np = np.array(image)
        except Exception:
            st.error("Invalid image")
            return

        image_np = utils.resize_image(image_np)
        st.image(image_np, caption="Original", use_column_width=True)

        if st.button("Detect"):
            with st.spinner("Processing..."):
                try:
                    files = {
                        "image": (file.name, bytes_data, file.type)
                    }

                    params = {"confidence_threshold": conf}

                    res = requests.post(API_URL, files=files, params=params, timeout=60)

                    if res.status_code != 200:
                        st.error(f"API Error: {res.text}")
                        return

                    data = res.json()

                    if not data.get("success"):
                        st.error(data.get("error"))
                        return

                    detections = data["detections"]

                    det_objs = [
                        utils.DetectionResult(
                            d["class_id"],
                            d["class_name"],
                            d["confidence"],
                            d["bbox"]
                        ) for d in detections
                    ]

                    annotated = utils.draw_detections_on_image(image_np, det_objs)

                    st.session_state["result_img"] = annotated

                    st.image(annotated, caption="Result", use_column_width=True)

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Detections", data["num_detections"])

                    with col2:
                        st.metric("Time (ms)", f"{data['inference_time_ms']:.1f}")

                    with col3:
                        fps = 1000 / data["inference_time_ms"] if data["inference_time_ms"] > 0 else 0
                        st.metric("FPS", f"{fps:.1f}")

                    st.text(utils.create_summary_text(det_objs))

                except Exception as e:
                    st.error(str(e))

        # ✅ Persist Save button
        if "result_img" in st.session_state:
            if st.button("Save Result"):
                path = os.path.join(
                    config.INFERENCE_OUTPUT_DIR,
                    f"det_{utils.get_timestamp_string()}.jpg"
                )
                utils.save_image(st.session_state["result_img"], path)
                st.success(f"Saved: {path}")

    # ✅ Dev utility
    if st.button("Clear Cache"):
        st.cache_resource.clear()
        st.success("Cache cleared")


if __name__ == "__main__":
    main()