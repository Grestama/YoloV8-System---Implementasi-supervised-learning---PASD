"""
FastAPI Backend - YOLOv8 Oral Disease Detection API
"""
import os
import logging
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ultralytics import YOLO  # ✅ FIX WAJIB

import config
import utils


# =====================================
# INIT
# =====================================
config.create_dirs()


# =====================================
# LOGGING
# =====================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.API_LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =====================================
# MODELS
# =====================================
class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]


class PredictResponse(BaseModel):
    success: bool
    message: str
    num_detections: int
    detections: List[Detection]
    inference_time_ms: float
    image_shape: Optional[List[int]] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    device: str


# =====================================
# APP
# =====================================
app = FastAPI(title="YOLOv8 Oral Disease Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


# =====================================
# STARTUP
# =====================================
@app.on_event("startup")
async def startup_event():
    global model
    try:
        logger.info(f"Loading model from: {config.MODEL_PATH}")

        model = utils.load_model(config.MODEL_PATH)

        # Warmup
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model(dummy, verbose=False)

        logger.info("✅ Model loaded & warmed up")

    except Exception as e:
        logger.error(traceback.format_exc())
        raise


# =====================================
# HEALTH
# =====================================
@app.get("/health", response_model=HealthResponse, operation_id="health_check_unique")
async def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        model_path=config.MODEL_PATH,
        device="GPU" if model else "CPU"
    )

# =====================================
# SINGLE PREDICT
# =====================================
@app.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    confidence_threshold: float = Query(config.CONFIDENCE_THRESHOLD, ge=0.0, le=1.0)
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if image.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(status_code=415, detail="Unsupported format")

    try:
        image_bytes = await image.read()

        if len(image_bytes) > config.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large")

        np_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Invalid image")

        timer = utils.PerformanceTimer()
        timer.start("inference")

        results = model(frame, conf=confidence_threshold, verbose=False)

        timer.end("inference")

        detections = utils.parse_yolo_results(results)
        inference_time = timer.get_elapsed("inference")

        detection_list = [
            Detection(**d.to_dict()) for d in detections
        ]

        return PredictResponse(
            success=True,
            message="OK",
            num_detections=len(detection_list),
            detections=detection_list,
            inference_time_ms=inference_time * 1000,
            image_shape=list(frame.shape)
        )

    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# =====================================
# BATCH PREDICT
# =====================================
@app.post("/predict-batch")
async def predict_batch(
    images: List[UploadFile] = File(...),
    confidence_threshold: float = Query(config.CONFIDENCE_THRESHOLD, ge=0.0, le=1.0)
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(images) > 20:
        raise HTTPException(status_code=413, detail="Too many files (max 20)")

    outputs = []

    for image in images:
        try:
            if image.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
                raise ValueError("Unsupported format")

            image_bytes = await image.read()

            if len(image_bytes) > config.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
                raise ValueError("File too large")

            np_array = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

            if frame is None:
                raise ValueError("Invalid image")

            results = model(frame, conf=confidence_threshold, verbose=False)
            detections = utils.parse_yolo_results(results)

            outputs.append({
                "filename": image.filename,
                "success": True,
                "detections": [d.to_dict() for d in detections]
            })

        except Exception as e:
            outputs.append({
                "filename": image.filename,
                "success": False,
                "error": str(e)
            })

    return {"success": True, "results": outputs}


# =====================================
# INFO
# =====================================
@app.get("/info")
async def get_info():
    return {
        "model_path": config.MODEL_PATH,
        "classes": config.CLASS_NAMES
    }


# =====================================
# ERROR HANDLER
# =====================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail}
    )


# =====================================
# MAIN
# =====================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT
    )

# ADDITIONS ONLY (patch-style, integrate ke file kamu)

import time
import torch

# =====================================
# DEVICE DETECTION
# =====================================
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# =====================================
# STARTUP (REPLACE EXISTING)
# =====================================
@app.on_event("startup")
async def startup_event():
    global model

    try:
        config.validate_paths(raise_error=True)

        device = get_device()
        logger.info(f"Loading model on: {device}")

        model = YOLO(config.MODEL_PATH)

        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        model(dummy, verbose=False)

        logger.info("✅ Model loaded & warmed up")

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


# =====================================
# MIDDLEWARE (NEW)
# =====================================
@app.middleware("http")
async def add_process_time_header(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    response.headers["X-Process-Time-ms"] = str(round(duration, 2))
    return response


# =====================================
# HEALTH (REPLACE)
# =====================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    device = get_device()

    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        model_path=config.MODEL_PATH,
        device=device
    )