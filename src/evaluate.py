from pathlib import Path
import joblib
from sklearn.metrics import classification_report, roc_auc_score
from .data_prep import load_data

BASE = Path(__file__).resolve().parents[1]
MODELS = BASE / "models"

def main():
    model_path = MODELS / "lung_cancer_survival_model.joblib"
    pipe = joblib.load(model_path)
    df = load_data()
    target = "survived"
    drop_cols = ["id", "diagnosis_date", "end_treatment_date"]
    X = df.drop(columns=[target] + drop_cols, errors="ignore")
    y = df[target].astype(int)

    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)[:,1] if hasattr(pipe, "predict_proba") else None
    print(classification_report(y, pred))
    if proba is not None:
        print("AUC:", roc_auc_score(y, proba))

if __name__ == "__main__":
    main()
