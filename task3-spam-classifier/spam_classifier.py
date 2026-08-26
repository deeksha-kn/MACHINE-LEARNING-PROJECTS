"""
spam_classifier.py
--------------------
Task 03: Spam Email Classifier

Pipeline:
1. Load labeled dataset (spam vs ham messages)
2. Preprocess text: lowercase, remove punctuation/numbers, remove stopwords
3. TF-IDF vectorization
4. Train 3 models: Naive Bayes, Logistic Regression, SVM
5. Evaluate each with accuracy, precision, recall, F1-score
6. Save confusion matrix + model comparison plots
7. Save the best model + vectorizer to disk (used later by predict.py)
"""

import re
import string
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import nltk
from nltk.corpus import stopwords

# ------------------------------------------------------------------
# 0. SETUP: download NLTK stopwords (only happens once)
# ------------------------------------------------------------------
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))

# ------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------
DATA_PATH = "spam_dataset.csv"
df = pd.read_csv(DATA_PATH)
df.columns = ["label", "message"]
df = df.dropna(subset=["label", "message"])

print("=" * 60)
print("STEP 1: Data Overview")
print("=" * 60)
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\nClass distribution:\n{df['label'].value_counts()}")

# Convert labels to binary: spam=1, ham=0
df["target"] = df["label"].map({"ham": 0, "spam": 1})

# ------------------------------------------------------------------
# 2. TEXT PREPROCESSING: clean, tokenize (implicitly), remove stopwords
# ------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)       # remove URLs
    text = re.sub(r"\d+", " ", text)                   # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    tokens = text.split()                               # simple tokenization
    tokens = [w for w in tokens if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)

print("\n" + "=" * 60)
print("STEP 2: Text Preprocessing")
print("=" * 60)
df["clean_message"] = df["message"].apply(clean_text)
print(df[["message", "clean_message"]].head())

# ------------------------------------------------------------------
# 3. TRAIN/TEST SPLIT
# ------------------------------------------------------------------
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["clean_message"], df["target"],
    test_size=0.2, random_state=42, stratify=df["target"]
)

# ------------------------------------------------------------------
# 4. TF-IDF VECTORIZATION
# ------------------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print(f"\nTF-IDF matrix shape (train): {X_train.shape}")

# ------------------------------------------------------------------
# 5. TRAIN & EVALUATE MULTIPLE MODELS
# ------------------------------------------------------------------
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel="linear", probability=True),
}

results = []
trained_models = {}

print("\n" + "=" * 60)
print("STEP 3: Model Training & Evaluation")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
    trained_models[name] = model

    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    fname = f"confusion_matrix_{name.replace(' ', '_').lower()}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")

# ------------------------------------------------------------------
# 6. MODEL COMPARISON CHART
# ------------------------------------------------------------------
results_df = pd.DataFrame(results)
print("\n" + "=" * 60)
print("STEP 4: Model Comparison")
print("=" * 60)
print(results_df)

results_df.set_index("model")[["accuracy", "precision", "recall", "f1"]].plot(
    kind="bar", figsize=(9, 6)
)
plt.title("Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()
print("\nSaved: model_comparison.png")

# ------------------------------------------------------------------
# 7. SAVE BEST MODEL (highest F1) + vectorizer
# ------------------------------------------------------------------
best_row = results_df.loc[results_df["f1"].idxmax()]
best_model_name = best_row["model"]
best_model = trained_models[best_model_name]

joblib.dump(best_model, "best_model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")

print(f"\nBest model: {best_model_name} (F1 = {best_row['f1']:.4f})")
print("Saved: best_model.joblib, vectorizer.joblib")
print("\nDone! Run 'python predict.py' to test messages interactively.")
