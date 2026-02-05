from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

from data_utils import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "greece_listings.csv"
ENRICHED_PATH = DATA_DIR / "greece_listings_enriched.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "rf_pipeline.joblib"

def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def main():
    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    feature_cols = bundle["feature_cols"]

    data_path = ENRICHED_PATH if ENRICHED_PATH.exists() else DATA_PATH
    df = load_data(data_path)

    # Same split as training scripts for apples-to-apples comparisons
    X = df[feature_cols].copy()
    y = df["res_price_sqr"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_pred = pipeline.predict(X_test)

    overall = mae(y_test.to_numpy(), y_pred)

    # Top 20% by TRUE €/m² (in the test set)
    threshold = np.quantile(y_test.to_numpy(), 0.8)
    mask_top20 = y_test.to_numpy() >= threshold
    top20 = mae(y_test.to_numpy()[mask_top20], y_pred[mask_top20])



    print(f"Overall MAE (€/m²): {overall:.2f}")
    print(f"Top 20% MAE (€/m²): {top20:.2f}  (threshold >= {threshold:.2f} €/m²)")



if __name__ == "__main__":
    main()
