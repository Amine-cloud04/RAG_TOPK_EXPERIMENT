from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")
INPUT_CSV = RESULTS_DIR / "rag_results_evaluated.csv"
SUMMARY_CSV = RESULTS_DIR / "topk_summary.csv"
OUTPUT_CSV = RESULTS_DIR / "human_evaluation_template.csv"
QUESTION_LIMIT = 20


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError("Run python analyze_topk_results.py first.")

    df = pd.read_csv(INPUT_CSV)
    summary = pd.read_csv(SUMMARY_CSV)
    best_k = int(summary.sort_values("Final_Score", ascending=False).iloc[0]["K"])

    selected_questions = df["Question"].drop_duplicates().head(QUESTION_LIMIT).tolist()
    selected_k = [1, best_k, 20]
    template = df[
        df["Question"].isin(selected_questions)
        & df["K"].astype(int).isin(selected_k)
    ].copy()

    template = template[
        [
            "Question",
            "K",
            "Sources",
            "Answer",
            "Reference Answer",
            "ROUGE-L",
            "BLEU",
            "Retrieval Recall",
            "Latency",
        ]
    ].sort_values(["Question", "K"])

    template["Human Relevance Score"] = ""
    template["Human Faithfulness Score"] = ""
    template["Human Clarity Score"] = ""
    template["Human Comment"] = ""

    template.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV}")
    print("Fill the three human score columns from 1 to 5, then run:")
    print("python scripts\\summarize_human_eval.py")


if __name__ == "__main__":
    main()
