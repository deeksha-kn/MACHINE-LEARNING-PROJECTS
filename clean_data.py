"""
clean_data.py
--------------
Cleans discord_ecosystem_2026.csv and saves a cleaned version.

Run:
    python clean_data.py
"""

import pandas as pd
import numpy as np
import os

# ---------- CONFIG ----------
INPUT_FILE = "discord_ecosystem_2026.csv"
OUTPUT_FILE = "discord_ecosystem_2026_cleaned.csv"


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find '{path}'. Place the CSV in the same folder as this script.")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Drop exact duplicate rows (if any)
    df = df.drop_duplicates()

    # 2. Drop duplicate repos by id (keep first occurrence)
    df = df.drop_duplicates(subset="id", keep="first")

    # 3. Fill missing text fields with sensible placeholders
    df["description"] = df["description"].fillna("No description provided")
    df["language"] = df["language"].fillna("Unknown")
    df["license"] = df["license"].fillna("No License")
    df["topics"] = df["topics"].fillna("")
    df["homepage"] = df["homepage"].fillna("")

    # 4. Convert date columns to proper datetime
    for col in ["created_at", "updated_at", "pushed_at"]:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # 5. Ensure boolean columns are proper bool dtype
    for col in ["is_official_discord", "is_fork", "archived"]:
        df[col] = df[col].astype(bool)

    # 6. Ensure numeric columns are numeric (coerce bad values to NaN, then fill)
    numeric_cols = ["stars", "forks", "watchers", "open_issues", "stars_per_day",
                     "repo_age_days", "days_since_last_push", "size_kb"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # 7. Remove impossible/negative values in count columns
    for col in ["stars", "forks", "watchers", "open_issues", "size_kb", "repo_age_days"]:
        df = df[df[col] >= 0]

    # 8. Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    # 9. Feature engineering: helpful derived columns for analysis
    df["primary_topic"] = df["topics"].apply(lambda x: x.split(";")[0] if x else "none")
    df["fork_to_star_ratio"] = np.where(df["stars"] > 0, df["forks"] / df["stars"], 0)
    df["is_active"] = df["days_since_last_push"] <= 90  # pushed in last 90 days
    df["created_year"] = df["created_at"].dt.year

    # 10. Reset index
    df = df.reset_index(drop=True)

    return df


def main():
    print("Loading data...")
    df = load_data(INPUT_FILE)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    print("Cleaning data...")
    df_clean = clean_data(df)
    print(f"After cleaning: {len(df_clean)} rows.")

    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved cleaned data to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()
