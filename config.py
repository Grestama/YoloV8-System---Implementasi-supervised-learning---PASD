import os

# =====================================
# BASE PATH
# =====================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =====================================
# MODEL PATHS
# =====================================
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "model", "best.pt")
)

BASE_MODEL = os.getenv(
    "BASE_MODEL",
    os.path.join(BASE_DIR, "model", "yolov8s.pt")
)


# =====================================
# DATASET
# =====================================
DATA_YAML = os.getenv(
    "DATA_YAML",
    os.path.join(BASE_DIR, "dataset", "data.yaml")
)


# =====================================
# CLASS CONFIG
# =====================================
CLASS_NAMES = {
    0: "Calculus",
    1: "Caries",
    2: "Gingivitis",
    3: "Hypodontia"
}

CLASS_COLORS = {
    0: (0, 255, 255),
    1: (0, 165, 255),
    2: (0, 0, 255),
    3: (255, 0, 0)
}


# =====================================
# OUTPUT DIRECTORIES
# =====================================
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TEST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "test_results")
INFERENCE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "inference")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
API_LOG_DIR = os.path.join(OUTPUT_DIR, "api_logs")


def create_dirs():
    for d in [
        OUTPUT_DIR,
        TEST_OUTPUT_DIR,
        INFERENCE_OUTPUT_DIR,
        METRICS_DIR,
        API_LOG_DIR
    ]:
        os.makedirs(d, exist_ok=True)


create_dirs()

API_LOG_FILE = os.path.join(API_LOG_DIR, "api.log")


# =====================================
# PATH VALIDATION (NEW)
# =====================================
def validate_paths(raise_error: bool = False) -> dict:
    status = {
        "model_exists": os.path.isfile(MODEL_PATH),
        "data_yaml_exists": os.path.isfile(DATA_YAML)
    }

    if raise_error:
        if not status["model_exists"]:
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        if not status["data_yaml_exists"]:
            raise FileNotFoundError(f"Data YAML not found: {DATA_YAML}")

    return status


# =====================================
# INFERENCE SETTINGS
# =====================================
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.5))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", 0.45))
IMG_SIZE = int(os.getenv("IMG_SIZE", 640))


# =====================================
# API SETTINGS
# =====================================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))


# =====================================
# STREAMLIT SETTINGS
# =====================================
STREAMLIT_THEME = os.getenv("STREAMLIT_THEME", "light")
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", 10))


# =====================================
# TARGETS
# =====================================
TARGET_PRECISION = 0.82
TARGET_RECALL = 0.78
TARGET_F1 = 0.80


# =====================================
# TRAINING
# =====================================
EPOCHS = int(os.getenv("EPOCHS", 10))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))
IMG_SIZE_TRAIN = int(os.getenv("IMG_SIZE_TRAIN", 640))


# =====================================
# DEVICE
# =====================================
DEVICE = os.getenv("DEVICE", "cpu")  # "cpu" / "cuda"