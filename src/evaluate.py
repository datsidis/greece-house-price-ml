import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from data_utils import load_data, LOCATION_COL

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "greece_listings.csv"
ENRICHED_PATH = DATA_DIR / "greece_listings_enriched.csv"

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def main():
    data_path = ENRICHED_PATH if ENRICHED_PATH.exists() else DATA_PATH
    df = load_data(data_path)

    feature_cols = [
        "res_sqr", "log_res_sqr", "bedrooms", "bathrooms", "property_age",
        "dist_station_km", "dist_beach_km", "dist_acropolis_km",
        LOCATION_COL, "energyclass",
        "parking", "auto_heating", "solar", "cooling", "safe_door", "gas",
        "fireplace", "furniture", "student",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df["res_price_sqr"]

    numeric_features = [
        "res_sqr", "log_res_sqr", "bedrooms", "bathrooms", "property_age",
        "dist_station_km", "dist_beach_km", "dist_acropolis_km",
    ]
    numeric_features = [c for c in numeric_features if c in X.columns]
    categorical_features = [c for c in [LOCATION_COL, "energyclass"] if c in X.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]), numeric_features),

            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_features),
        ],
        remainder="drop"
    )

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([("prep", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    error = mae(y_test.to_numpy(), y_pred)
    print(f"MAE (€/m²): {error:.2f}")

    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.35)
    plt.xlabel("True €/m²")
    plt.ylabel("Predicted €/m²")
    plt.title("True vs Predicted (Random Forest)")
    plt.show()

if __name__ == "__main__":
    main()
