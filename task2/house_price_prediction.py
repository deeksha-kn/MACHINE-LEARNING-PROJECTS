"""
house_price_prediction.py
--------------------------
Task 02: Linear Regression Model
Predicts house prices using features like area, bedrooms, bathrooms,
age, and location.

Pipeline:
1. Load & preprocess data (handle missing values, encode categoricals)
2. Correlation analysis
3. Train/test split
4. Feature scaling
5. Train Linear Regression model (scikit-learn)
6. Evaluate with MSE, RMSE, R2
7. Visualize actual vs predicted values
"""

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
# Resolve paths relative to this script's folder, so the pipeline works
# no matter what directory it's run from (e.g. cloned repo root vs task2/).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "house_prices.csv")
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("STEP 1: Data Overview")
print("=" * 60)
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nMissing values before cleaning:\n{df.isnull().sum()}")

# ------------------------------------------------------------------
# 2. PREPROCESS: handle missing values
# ------------------------------------------------------------------
# Fill numeric missing values with the column median (robust to outliers)
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

print(f"\nMissing values after cleaning:\n{df.isnull().sum()}")

# One-hot encode the categorical 'location' column
df_encoded = pd.get_dummies(df, columns=["location"], drop_first=True)

# ------------------------------------------------------------------
# 3. CORRELATION ANALYSIS
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Correlation with price")
print("=" * 60)
correlations = df_encoded.corr(numeric_only=True)["price"].sort_values(ascending=False)
print(correlations)

plt.figure(figsize=(8, 6))
sns.heatmap(df_encoded.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "correlation_heatmap.png"), dpi=150)
print("\nSaved: correlation_heatmap.png")
plt.close()

# ------------------------------------------------------------------
# 4. SPLIT FEATURES / TARGET, TRAIN/TEST SPLIT
# ------------------------------------------------------------------
X = df_encoded.drop(columns=["price"])
y = df_encoded["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------------
# 5. FEATURE SCALING
# ------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------
# 6. TRAIN MODEL (multiple linear regression)
# ------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ------------------------------------------------------------------
# 7. EVALUATE
# ------------------------------------------------------------------
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("STEP 3: Model Evaluation")
print("=" * 60)
print(f"MSE  : {mse:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R^2  : {r2:.4f}")

# Show feature importance (via coefficients)
coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_
}).sort_values(by="coefficient", key=abs, ascending=False)
print("\nFeature coefficients (scaled):")
print(coef_df)

# ------------------------------------------------------------------
# 8. VISUALIZE: Actual vs Predicted
# ------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor="k")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2, label="Perfect prediction")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "actual_vs_predicted.png"), dpi=150)
print("\nSaved: actual_vs_predicted.png")
plt.close()

# Residual plot (extra, useful for diagnostics)
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True)
plt.title("Distribution of Residuals")
plt.xlabel("Residual (Actual - Predicted)")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "residuals_distribution.png"), dpi=150)
print("Saved: residuals_distribution.png")
plt.close()

print("\nDone! Check the generated PNG files for visualizations.")
