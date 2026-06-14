from pathlib import Path
import ast
import json

import pandas as pd


RESULTS_DIR = Path("results")
INPUT_CSV = RESULTS_DIR / "rag_results_evaluated.csv"
OUTPUT_CSV = RESULTS_DIR / "ragas_scores.csv"


def parse_contexts(value):
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, list) else []


def run_ragas(input_csv=INPUT_CSV, output_csv=OUTPUT_CSV):
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS is not installed. Install optional dependencies with:\n"
            "pip install ragas datasets\n"
        ) from exc

    df = pd.read_csv(input_csv)
    required = {"Question", "Answer", "Retrieved Contexts", "Reference Answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for RAGAS evaluation: {', '.join(sorted(missing))}")

    dataset = Dataset.from_dict(
        {
            "question": df["Question"].fillna("").astype(str).tolist(),
            "answer": df["Answer"].fillna("").astype(str).tolist(),
            "contexts": df["Retrieved Contexts"].fillna("[]").apply(parse_contexts).tolist(),
            "ground_truth": df["Reference Answer"].fillna("").astype(str).tolist(),
        }
    )

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    scores = result.to_pandas()
    scores.to_csv(output_csv, index=False)
    return scores


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_CSV}. Run python analyze_topk_results.py first."
        )

    scores = run_ragas()
    print(f"Saved {OUTPUT_CSV}")
    print(scores.head().to_string(index=False))


if __name__ == "__main__":
    main()
