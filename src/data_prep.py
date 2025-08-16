import pandas as pd
from pathlib import Path

RAW_CSV = Path(__file__).resolve().parents[1] / "dataset_med.csv"

def load_data():
    df = pd.read_csv(RAW_CSV)
    for col in ["diagnosis_date", "end_treatment_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["diagnosis_year"] = df["diagnosis_date"].dt.year
    df["diagnosis_month"] = df["diagnosis_date"].dt.month
    df["treatment_year"] = df["end_treatment_date"].dt.year
    df["treatment_month"] = df["end_treatment_date"].dt.month
    df["treatment_duration_days"] = (df["end_treatment_date"] - df["diagnosis_date"]).dt.days
    return df
