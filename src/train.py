from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from .data_prep import load_data

BASE = Path(__file__).resolve().parents[1]
MODELS = BASE / "models"
REPORTS = BASE / "reports"
MODELS.mkdir(exist_ok=True, parents=True)
REPORTS.mkdir(exist_ok=True, parents=True)

def main():
    df = load_data()
    target = "survived"
    drop_cols = ["id", "diagnosis_date", "end_treatment_date"]
    X = df.drop(columns=[target] + drop_cols, errors="ignore")
    y = df[target].astype(int)

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ])

    model = HistGradientBoostingClassifier(random_state=42)
    pipe = Pipeline([("prep", preprocess), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:,1] if hasattr(pipe, "predict_proba") else None

    report = classification_report(y_test, pred)
    auc = roc_auc_score(y_test, proba) if proba is not None else None

    (REPORTS / "evaluation.md").write_text(
        f"# Evaluation\n\n{report}\n\nAUC: {auc if auc is not None else 'N/A'}\n"
    )
    joblib.dump(pipe, MODELS / "lung_cancer_survival_model.joblib")
    print("Model trained and saved.")

if __name__ == "__main__":
    main()
