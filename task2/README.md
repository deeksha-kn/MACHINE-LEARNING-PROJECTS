# Task 02 — Linear Regression: House Price Prediction

Predicts house prices from features like area, bedrooms, bathrooms, age, and
location. Covers preprocessing, correlation analysis, feature scaling,
multiple linear regression, evaluation (MSE, RMSE, R²), and visualization.

## Files
- `generate_dataset.py` — creates a synthetic `house_prices.csv` (skip this if you have your own dataset; just make sure it has the same column names, or edit the script's column references).
- `house_price_prediction.py` — the actual ML pipeline (main script to run).
- `requirements.txt` — Python packages needed.
- `house_prices.csv` — the dataset used to produce the results below.
- `correlation_heatmap.png`, `actual_vs_predicted.png`, `residuals_distribution.png` — generated output plots.

## How to run

1. Clone the repo and move into this folder:
   ```bash
   git clone https://github.com/deeksha-kn/MACHINE-LEARNING-PROJECTS.git
   cd MACHINE-LEARNING-PROJECTS/task2
   ```

2. (Recommended) create a virtual environment:
   ```bash
   python -m venv venv
   ```
   Activate it:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) regenerate the dataset — skip this if you want to use the `house_prices.csv` already in this folder:
   ```bash
   python generate_dataset.py
   ```

5. Run the model:
   ```bash
   python house_price_prediction.py
   ```

You'll see console output with data stats, correlations, and evaluation
metrics (MSE, RMSE, R²), and the three PNG files will be (re)saved into this
same folder — the scripts resolve all paths relative to their own location,
so this works whether you run them from `task2/` or from the repo root.

## Using your own dataset instead

If you have a real dataset (e.g. from Kaggle), place your CSV in this
folder, rename it `house_prices.csv` (or change `DATA_PATH` in
`house_price_prediction.py`), and make sure it has columns similar to:
`area, bedrooms, bathrooms, age, location, price`. If your column names
differ, update the column references near the top of the script accordingly.

## What the script does

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

## Results (on the included dataset)

- **R²**: ~0.93
- **RMSE**: ~27,100

Feature coefficients show `area` as the strongest positive driver of price,
with `location` (Waterfront/Downtown vs Rural) and `age` also having a
meaningful effect.
