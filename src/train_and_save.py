from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from data_utils import load_data, LOCATION_COL

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "greece_listings.csv"
ENRICHED_PATH = DATA_DIR / "greece_listings_enriched.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "rf_pipeline.joblib"

def build_pipeline(X):
    numeric_features = [
        "res_sqr", "log_res_sqr", "bedrooms", "bathrooms",
        "property_age",
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
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(steps=[("prep", preprocessor), ("model", model)])

def main():
    # Use enriched data (with geo features) when available
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_cols": feature_cols,
            "target": "res_price_sqr",
        },
        MODEL_PATH
    )

    print("Saved model to:", MODEL_PATH)
    print("Features used:", feature_cols)
    print("Rows used:", len(df))

if __name__ == "__main__":
    main()
