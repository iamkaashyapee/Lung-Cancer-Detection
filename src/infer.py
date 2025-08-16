from pathlib import Path
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
MODELS = BASE / "models"

def load_model():
    return joblib.load(MODELS / "lung_cancer_survival_model.joblib")

def predict_from_dict(features: dict):
    pipe = load_model()
    X = pd.DataFrame([features])
    return pipe.predict_proba(X)[0,1]
