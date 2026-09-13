"""
predict.py
-----------
Simple command-line interface: type a message, get spam/ham prediction.
Run spam_classifier.py first to generate best_model.joblib and vectorizer.joblib.
"""

import re
import string
import joblib
import nltk
from nltk.corpus import stopwords

try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)


def main():
    try:
        model = joblib.load("best_model.joblib")
        vectorizer = joblib.load("vectorizer.joblib")
    except FileNotFoundError:
        print("Model files not found. Run 'python spam_classifier.py' first to train and save the model.")
        return

    print("=" * 50)
    print("Spam Email/SMS Classifier")
    print("=" * 50)
    print("Type a message to check if it's spam. Type 'quit' to exit.\n")

    while True:
        text = input("Message: ").strip()
        if text.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if not text:
            continue

        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]

        label = "SPAM" if pred == 1 else "HAM (not spam)"

        # Show confidence if the model supports predict_proba
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vec)[0]
            confidence = proba[1] if pred == 1 else proba[0]
            print(f"--> {label}  (confidence: {confidence:.2%})\n")
        else:
            print(f"--> {label}\n")


if __name__ == "__main__":
    main()
