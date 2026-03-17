import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "ml" / "model.pkl")
scaler = joblib.load(BASE_DIR / "ml" / "scaler.pkl")
columns = joblib.load(BASE_DIR / "ml" / "columns.pkl")