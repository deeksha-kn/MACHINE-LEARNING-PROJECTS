"""
generate_dataset.py
--------------------
Creates a synthetic (but realistic) house price dataset and saves it as
'house_prices.csv'. Run this once before running house_price_prediction.py
if you don't already have your own dataset.

If you have a real dataset (e.g. from Kaggle), just skip this file and
point house_price_prediction.py to your own CSV instead.
"""

import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

np.random.seed(42)

n_samples = 1000

locations = ["Downtown", "Suburb", "Rural", "Uptown", "Waterfront"]
location_price_factor = {
    "Downtown": 1.4,
    "Uptown": 1.25,
    "Waterfront": 1.5,
    "Suburb": 1.0,
    "Rural": 0.7,
}

area = np.random.normal(1800, 600, n_samples).clip(400, 5000)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
age = np.random.randint(0, 50, n_samples)
location = np.random.choice(locations, n_samples)

# Introduce some missing values on purpose (to demonstrate preprocessing)
area_with_nan = area.copy()
missing_idx = np.random.choice(n_samples, size=30, replace=False)
area_with_nan[missing_idx] = np.nan

# Base price formula + noise
base_price = (
    area * 120
    + bedrooms * 8000
    + bathrooms * 5000
    - age * 400
)
loc_factor = np.array([location_price_factor[loc] for loc in location])
price = base_price * loc_factor + np.random.normal(0, 15000, n_samples)
price = price.clip(20000, None)

df = pd.DataFrame({
    "area": area_with_nan,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "age": age,
    "location": location,
    "price": price.round(2),
})

df.to_csv(os.path.join(BASE_DIR, "house_prices.csv"), index=False)
print("Dataset created: house_prices.csv")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")
