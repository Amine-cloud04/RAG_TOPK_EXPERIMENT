from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")
INPUT_CSV = RESULTS_DIR / "human_evaluation_template.csv"
OUTPUT_CSV = RESULTS_DIR / "human_evaluation_summary.csv"


SCORE_COLUMNS = [
    "Human Relevance Score",
    "Human Faithfulness Score",
    "Human Clarity Score",
]


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError("Run python scripts\\generate_human_eval_template.py first.")

    df = pd.read_csv(INPUT_CSV)
    for column in SCORE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    missing = df[SCORE_COLUMNS].isna().any(axis=1).sum()
    if missing:
        print(f"Warning: {missing} rows still have missing human scores.")

    df["Human Mean Score"] = df[SCORE_COLUMNS].mean(axis=1)
    summary = (
        df.groupby("K")
        .agg(
            Human_Evaluated_Answers=("Human Mean Score", "count"),
            Avg_Human_Relevance=("Human Relevance Score", "mean"),
            Avg_Human_Faithfulness=("Human Faithfulness Score", "mean"),
            Avg_Human_Clarity=("Human Clarity Score", "mean"),
            Avg_Human_Mean=("Human Mean Score", "mean"),
            Std_Human_Mean=("Human Mean Score", "std"),
        )
        .reset_index()
        .sort_values("K")
    )

    summary.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
