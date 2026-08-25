"""
visualize_data.py
------------------
Generates charts from the cleaned discord_ecosystem_2026_cleaned.csv.
Saves each chart as a PNG in the 'charts' folder AND shows them on screen.

Run (after clean_data.py):
    python visualize_data.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

INPUT_FILE = "discord_ecosystem_2026_cleaned.csv"
CHARTS_DIR = "charts"

sns.set_theme(style="whitegrid")
plt.rcParams["figure.autolayout"] = True


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find '{path}'. Run clean_data.py first.")
    return pd.read_csv(path)


def save_and_show(fig, name):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    path = os.path.join(CHARTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")


def top_repos_by_stars(df):
    top = df.nlargest(15, "stars")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=top, x="stars", y="name", hue="name", legend=False, palette="viridis", ax=ax)
    ax.set_title("Top 15 Discord-related Repos by Stars")
    ax.set_xlabel("Stars")
    ax.set_ylabel("Repository")
    save_and_show(fig, "01_top_repos_by_stars.png")


def language_distribution(df):
    top_langs = df["language"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=top_langs.values, y=top_langs.index, hue=top_langs.index, legend=False, palette="mako", ax=ax)
    ax.set_title("Top 10 Programming Languages Used")
    ax.set_xlabel("Number of Repositories")
    ax.set_ylabel("Language")
    save_and_show(fig, "02_language_distribution.png")


def owner_type_pie(df):
    counts = df["owner_type"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
           colors=sns.color_palette("pastel"), startangle=90)
    ax.set_title("Repository Owner Type (User vs Organization)")
    save_and_show(fig, "03_owner_type_pie.png")


def stars_vs_forks(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="stars", y="forks", hue="owner_type", alpha=0.6, ax=ax)
    ax.set_title("Stars vs Forks")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Stars (log scale)")
    ax.set_ylabel("Forks (log scale)")
    save_and_show(fig, "04_stars_vs_forks.png")


def repos_created_per_year(df):
    counts = df["created_year"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=counts.index, y=counts.values, marker="o", ax=ax)
    ax.set_title("Number of Repositories Created per Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Repositories Created")
    save_and_show(fig, "05_repos_created_per_year.png")


def license_distribution(df):
    top_licenses = df["license"].value_counts().head(8)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=top_licenses.values, y=top_licenses.index, hue=top_licenses.index, legend=False, palette="crest", ax=ax)
    ax.set_title("Top 8 Licenses Used")
    ax.set_xlabel("Number of Repositories")
    ax.set_ylabel("License")
    save_and_show(fig, "06_license_distribution.png")


def correlation_heatmap(df):
    numeric_cols = ["stars", "forks", "watchers", "open_issues", "stars_per_day",
                     "repo_age_days", "days_since_last_push", "size_kb"]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Between Numeric Metrics")
    save_and_show(fig, "07_correlation_heatmap.png")


def active_vs_inactive(df):
    counts = df["is_active"].value_counts()
    labels = ["Active (pushed in last 90 days)" if v else "Inactive" for v in counts.index]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts.values, labels=labels, autopct="%1.1f%%", colors=sns.color_palette("Set2"), startangle=90)
    ax.set_title("Active vs Inactive Repositories")
    save_and_show(fig, "08_active_vs_inactive.png")


def main():
    df = load_data(INPUT_FILE)
    print(f"Loaded {len(df)} cleaned rows. Generating charts...")

    top_repos_by_stars(df)
    language_distribution(df)
    owner_type_pie(df)
    stars_vs_forks(df)
    repos_created_per_year(df)
    license_distribution(df)
    correlation_heatmap(df)
    active_vs_inactive(df)

    print(f"\nAll charts saved in the '{CHARTS_DIR}' folder.")
    plt.show()


if __name__ == "__main__":
    main()
