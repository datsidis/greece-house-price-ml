# Greece House Price Prediction (€/m²) — Machine Learning

This project predicts **price per square meter (€/m²)** for residential properties in Greece using a complete end-to-end machine learning pipeline.

The project is intended **strictly for educational purposes**.  
It is **not** a real-estate valuation tool and **must not** be used for financial, legal, or investment decisions.

---

## Problem Definition

Given a property’s characteristics (size, rooms, age, location, amenities), estimate:

**Expected market-level price per square meter (€/m²)**

The objective is to produce a reasonable approximation based on historical listing data, not an exact valuation.

---

## Dataset

- Approximately 19,000 residential listings in Greece
- Features include:
    - Property size (m²)
    - Bedrooms and bathrooms
    - Construction year
    - Area-level location
    - Energy class
    - Binary amenities (parking, heating, solar, etc.)
    - Floor information provided as Greek text

### Target Variable
- `res_price_sqr`: listed price per square meter

---

## Data Cleaning and Preparation

- Numeric coercion with invalid values removed
- Outlier handling using the 99th percentile of €/m²
- Missing values handled inside the ML pipeline to avoid data leakage

---

## Feature Engineering

### Size and Non-Linearity
- `res_sqr`
- `log_res_sqr` to model diminishing returns of larger properties

### Property Age
- `property_age = current_year − construction_year`

**Renovation handling**:
- When renovation year is known, construction year is approximated as:
  renovation_year − 5

- This approximates partial depreciation rather than treating renovations as brand-new builds

### Location and Amenities
- Area-level location (`location_name`)
- Energy class
- Binary amenities (parking, heating, solar, cooling, etc.)

---

## Model

- **RandomForestRegressor**
- Implemented using a scikit-learn `Pipeline`
- Includes:
- Imputation for missing numeric and categorical values
- One-hot encoding for categorical features

The Random Forest model was chosen for:
- Robustness on tabular data
- Ability to capture non-linear relationships
- Strong baseline performance without heavy tuning

---

## Evaluation

### Metric
- Mean Absolute Error (MAE) 580.15 in €/m² 

### Segment-Based Evaluation
In addition to overall MAE, performance is evaluated on:
- Top 20% of listings by 1002.39 €/m²


This provides insight into performance across different market segments.

---

## Qualitative Validation (No Screenshots)

To complement quantitative metrics, predictions were compared against **publicly available real listings**.

For typical, non-luxury apartments without special views:
- Predicted €/m² values are often close to listed prices
- Differences generally fall within normal market and negotiation variance

These comparisons are used as **sanity checks**, not as proof of pricing accuracy.

No listing screenshots or proprietary content are included.

For example, a typical 96 m² apartment in a common area with the price of 270,000€ would yield a predicted price of approximately 266,849.03
and in the same area a 42 m² apartment with the price of 130,000€ would yield a predicted price of approximately 135,041.26.

---

## Limitations

This model cannot account for factors not present in the dataset, including:
- Actual view (sea, landmark, obstruction)
- Interior renovation quality
- Building aesthetics
- Street-level noise or orientation
- Seller urgency or negotiation dynamics

Predictions should be interpreted as **approximate market-level estimates only**.

---

## Educational Disclaimer

This project is provided **for educational and learning purposes only**.

- It is **not** a real-estate valuation system
- It should **not** be used to determine property prices
- It should **not** be used for financial, legal, or investment decisions

The author assumes no responsibility for any use of this project outside of its educational scope.

---

## How to Run

Install dependencies:
```bash
pip install -r requirements.txt
python src/train_and_save.py
python src/evaluate_segments.py
python src/feature_importance.py
