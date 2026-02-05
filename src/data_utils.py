import pandas as pd
import numpy as np

LOCATION_COL = "location_name"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    needed = [
        "res_price_sqr",
        "res_sqr",
        "bedrooms",
        "bathrooms",
        "construction_year",
        LOCATION_COL,
        "energyclass",
        "parking",
        "auto_heating",
        "solar",
        "cooling",
        "safe_door",
        "gas",
        "fireplace",
        "furniture",
        "student",
    ]

    needed = [c for c in needed if c in df.columns]
    df = df[needed].copy()

    # Numeric columns
    numeric_cols = [
        "res_price_sqr",
        "res_sqr",
        "bedrooms",
        "bathrooms",
        "construction_year",
    ]


    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Amenity columns: treat missing as 0 (not present / not specified)
    amenities = [
        "parking", "auto_heating", "solar", "cooling", "safe_door", "gas",
        "fireplace", "furniture", "student"
    ]
    for c in amenities:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # Clean text
    df[LOCATION_COL] = df[LOCATION_COL].astype(str).str.strip().str.lower()
    if "energyclass" in df.columns:
        df["energyclass"] = df["energyclass"].astype(str).str.strip().str.lower()

    # Drop invalid rows
    df = df.dropna(subset=["res_price_sqr", "res_sqr", "construction_year", LOCATION_COL])
    df = df[df["res_sqr"] > 0]
    df = df[df["res_price_sqr"] > 0]

    # ---------- FEATURE ENGINEERING ----------
    CURRENT_YEAR = 2026
    df["property_age"] = CURRENT_YEAR - df["construction_year"]
    df["property_age"] = df["property_age"].clip(lower=0)

    df["log_res_sqr"] = np.log(df["res_sqr"])

    # ---------- OUTLIER REMOVAL ----------
    upper = df["res_price_sqr"].quantile(0.99)
    df = df[df["res_price_sqr"] <= upper]

    return df
