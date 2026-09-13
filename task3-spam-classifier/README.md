# Task 03 — Spam Email/SMS Classifier

Classifies messages as **spam** or **ham** (not spam) using TF-IDF text
vectorization and scikit-learn classifiers (Naive Bayes, Logistic
Regression, SVM). Includes an interactive command-line tool to test your
own messages.

## Dataset

Uses the classic **SMS Spam Collection** dataset (5,572 real labeled SMS
messages: 4,825 ham / 747 spam), saved locally as `spam_dataset.csv` with
two columns: `label` (`ham`/`spam`) and `message`.

If you'd rather use your own dataset (e.g. an email dataset from Kaggle),
just replace `spam_dataset.csv` with your own file using the same two
column names, or edit `DATA_PATH`/column names at the top of
`spam_classifier.py`.

## Files
- `spam_dataset.csv` — the labeled dataset
- `spam_classifier.py` — main script: preprocesses text, trains 3 models, evaluates, saves the best one
- `predict.py` — interactive CLI: type a message, get a spam/ham prediction
- `requirements.txt` — Python packages needed

## How to run in VS Code

1. **Install Python** (3.9+) if you don't have it: https://www.python.org/downloads/
   Check "Add Python to PATH" during install.

2. **Install the VS Code Python extension**
   Extensions (Ctrl+Shift+X) → search "Python" (by Microsoft) → Install.

3. **Open this project folder in VS Code**
   File → Open Folder → select `spam_classifier_project`.

4. **Open a terminal**
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

7. **Train the models**
   ```bash
   python spam_classifier.py
   ```
   This preprocesses the text, trains Naive Bayes / Logistic Regression /
   SVM, prints accuracy/precision/recall/F1 for each, saves confusion
   matrix plots + a model comparison chart, and saves the best model as
   `best_model.joblib` + `vectorizer.joblib`.

8. **Try it out interactively**
   ```bash
   python predict.py
   ```
   Type any message and press Enter to see if it's classified as spam.
   Type `quit` to exit.

## What you'll get after running

- Console output: dataset overview, cleaned text samples, per-model
  accuracy/precision/recall/F1, classification reports
- `confusion_matrix_naive_bayes.png`, `confusion_matrix_logistic_regression.png`, `confusion_matrix_svm.png`
- `model_comparison.png` — bar chart comparing all 3 models
- `best_model.joblib` + `vectorizer.joblib` — the saved best-performing model, used by `predict.py`

Typical results on this dataset: **SVM performs best** (~98% accuracy,
~94% F1), since spam detection needs high precision (don't flag real
messages as spam) alongside strong recall.

## What the script does (matches the task requirements)

| Requirement | Where it's handled |
|---|---|
| Labeled dataset (spam vs non-spam) | `spam_dataset.csv` |
| Preprocess text: tokenization, stopword removal | `clean_text()` function |
| TF-IDF vectorization | `TfidfVectorizer` |
| Train Naive Bayes, Logistic Regression, SVM | Step 5 — all three trained & compared |
| Evaluate with accuracy, precision, recall, F1 | Step 5 — printed per model |
| Simple interface to check a message | `predict.py` |
