# Paths used by RoME.
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("ROME_DATA_ROOT", os.path.join(PROJECT_ROOT, "dataset"))
OUTPUT_ROOT = os.environ.get("ROME_OUTPUT_ROOT", os.path.join(PROJECT_ROOT, "saved"))

DATA_DIR = {
    "CMUMOSI": os.path.join(DATA_ROOT, "CMUMOSI"),
    "CMUMOSEI": os.path.join(DATA_ROOT, "CMUMOSEI"),
}
PATH_TO_FEATURES = {name: os.path.join(path, "features") for name, path in DATA_DIR.items()}
PATH_TO_LABEL = {
    "CMUMOSI": os.path.join(DATA_DIR["CMUMOSI"], "CMUMOSI_features_raw_2way.pkl"),
    "CMUMOSEI": os.path.join(DATA_DIR["CMUMOSEI"], "CMUMOSEI_features_raw_2way.pkl"),
}

SAVED_ROOT = OUTPUT_ROOT
MODEL_DIR = os.path.join(SAVED_ROOT, "model")
LOG_DIR = os.path.join(SAVED_ROOT, "log")
NPZ_DIR = os.path.join(SAVED_ROOT, "npz")
PRE_TRAINED_DIR = os.path.join(SAVED_ROOT, "pretrained")
