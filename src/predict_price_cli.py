from pathlib import Path
import joblib
import pandas as pd

from data_utils import LOCATION_COL

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "rf_pipeline.joblib"

def yes_no(prompt: str) -> int:
    val = input(prompt + " (y/n): ").strip().lower()
    return 1 if val in ("y", "yes") else 0

def main():
    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["pipeline"]
    feature_cols = bundle["feature_cols"]

    print("Enter property details (Greece listing predictor)\n")

    res_sqr = float(input("Square meters (res_sqr): ").strip())
    bedrooms = float(input("Bedrooms: ").strip())
    bathrooms = float(input("Bathrooms: ").strip())
    construction_year = float(input("Construction year (e.g., 2005): ").strip())
    location_name = input("Location name (exact-ish, e.g. 'kolonaki'): ").strip().lower()
    energyclass = input("Energy class (e.g. 'a', 'b', 'c' or leave blank): ").strip().lower() or "unknown"

    # amenities
    parking = yes_no("Parking")
    auto_heating = yes_no("Auto heating")
    solar = yes_no("Solar")
    cooling = yes_no("Cooling")
    safe_door = yes_no("Safe door")
    gas = yes_no("Gas")
    fireplace = yes_no("Fireplace")
    furniture = yes_no("Furniture")
    student = yes_no("Student-friendly")

    # engineered
    CURRENT_YEAR = 2024
    property_age = max(0, CURRENT_YEAR - construction_year)
    import math
    log_res_sqr = math.log(res_sqr) if res_sqr > 0 else 0.0

    row = {
        "res_sqr": res_sqr,
        "log_res_sqr": log_res_sqr,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "construction_year": construction_year,
        "property_age": property_age,
        LOCATION_COL: location_name,
        "energyclass": energyclass,
        "parking": parking,
        "auto_heating": auto_heating,
        "solar": solar,
        "cooling": cooling,
        "safe_door": safe_door,
        "gas": gas,
        "fireplace": fireplace,
        "furniture": furniture,
        "student": student,
    }

    X_new = pd.DataFrame([row])

    # Ensure the same columns used during training exist (missing -> NaN handled by imputers)
    for c in feature_cols:
        if c not in X_new.columns:
            X_new[c] = pd.NA
    X_new = X_new[feature_cols]

    pred_eur_per_sqm = float(pipeline.predict(X_new)[0])
    pred_total = pred_eur_per_sqm * res_sqr

    print("\nPrediction:")
    print(f"Estimated €/m²: {pred_eur_per_sqm:,.2f}")
    print(f"Estimated total price (€): {pred_total:,.2f}")

if __name__ == "__main__":
    main()
