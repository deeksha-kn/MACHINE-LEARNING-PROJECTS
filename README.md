# Task 02 — Linear Regression: House Price Prediction

Predicts house prices from features like area, bedrooms, bathrooms, age, and
location. Covers preprocessing, correlation analysis, feature scaling,
multiple linear regression, evaluation (MSE, RMSE, R²), and visualization.

## Files
- `generate_dataset.py` — creates a synthetic `house_prices.csv` (skip this if you have your own dataset; just make sure it has the same column names, or edit the script's column references).
- `house_price_prediction.py` — the actual ML pipeline (main script to run).
- `requirements.txt` — Python packages needed.

## How to run in VS Code

1. **Install Python** (3.9+) if you don't have it: https://www.python.org/downloads/
   During install, check "Add Python to PATH".

2. **Install the VS Code Python extension**
   Open VS Code → Extensions (Ctrl+Shift+X) → search "Python" (by Microsoft) → Install.

3. **Open the project folder**
   VS Code → File → Open Folder → select this `house_price_project` folder.

4. **Open a terminal in VS Code**
   Terminal → New Terminal (or `` Ctrl+` ``).

5. **(Recommended) Create a virtual environment**
   ```bash
   python -m venv venv
   ```
   Activate it:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

6. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

7. **Generate the dataset** (skip if using your own CSV)
   ```bash
   python generate_dataset.py
   ```

8. **Run the model**
   ```bash
   python house_price_prediction.py
   ```

You'll see console output with data stats, correlations, and evaluation
metrics (MSE, RMSE, R²), and three PNG files will be saved in the folder:
- `correlation_heatmap.png`
- `actual_vs_predicted.png`
- `residuals_distribution.png`

## Using your own dataset instead

If you have a real dataset (e.g. from Kaggle), just place your CSV in this
folder, rename it `house_prices.csv` (or change `DATA_PATH` in
`house_price_prediction.py`), and make sure it has columns similar to:
`area, bedrooms, bathrooms, age, location, price`. If your column names
differ, update the column references near the top of the script accordingly.

## What the script does (matches the task requirements)

| Requirement | Where it's handled |
|---|---|
| Load & preprocess with pandas/NumPy, handle missing values | Step 1–2: `fillna` with median |
| Train/test split | `train_test_split` |
| Train with scikit-learn `LinearRegression` | Step 6 |
| Evaluate with MSE and R² | Step 7 (also RMSE for interpretability) |
| Visualize actual vs predicted | `actual_vs_predicted.png` |
| Feature scaling | `StandardScaler` |
| Correlation analysis | `correlation_heatmap.png` + printed correlations |
| Multiple regression | Uses all features (area, bedrooms, bathrooms, age, location) |
