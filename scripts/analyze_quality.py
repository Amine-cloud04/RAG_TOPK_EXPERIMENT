"""
Quality Analysis for RAG Top-K Experiment

This script:
1. Reads results/rag_results.csv
2. Scores answer quality (1-5 scale)
3. Computes trade-off metrics (quality vs latency)
4. Identifies the best Top-K value
5. Saves analysis to results/best_k_summary.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path


RESULTS_DIR = Path("results")
INPUT_CSV = RESULTS_DIR / "rag_results.csv"
OUTPUT_SUMMARY = RESULTS_DIR / "best_k_summary.csv"


def score_answer_quality(answer):
    """
    Auto-score answer quality (1-5):
    - 5: Complete answer with details
    - 4: Good answer with some detail
    - 3: Partial answer / some useful info
    - 2: Minimal answer / mostly insufficient
    - 1: "Insufficient information" only
    """
    answer_str = str(answer).lower()
    answer_len = len(answer_str)
    
    # Rule 1: Explicit "insufficient information"
    if "insufficient information" in answer_str:
        # If there's extra text, it's a 2 (partial attempt)
        return 2 if answer_len > 100 else 1
    
    # Rule 2: Length-based scoring (more detailed = higher quality)
    if answer_len > 400:
        return 5
    elif answer_len > 250:
        return 4
    elif answer_len > 150:
        return 3
    else:
        return 2


def analyze_results():
    """Main analysis function"""
    
    if not INPUT_CSV.exists():
        print(f"Error: {INPUT_CSV} not found. Run main.py first.")
        return
    
    # Load results
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} results from {INPUT_CSV}")
    
    # Score each answer
    df["QualityScore"] = df["Answer"].apply(score_answer_quality)
    
    # Group by K and compute aggregate metrics
    k_analysis = df.groupby("K").agg({
        "QualityScore": ["mean", "std"],
        "Latency": ["mean", "std", "max"],
        "Question": "count"
    }).round(3)
    
    k_analysis.columns = ["AvgQuality", "StdQuality", "AvgLatency", "StdLatency", "MaxLatency", "SampleCount"]
    
    # Compute trade-off score (maximize quality, minimize latency)
    # Normalize to 0-1 scale
    min_latency = k_analysis["AvgLatency"].min()
    max_latency = k_analysis["AvgLatency"].max()
    
    k_analysis["NormLatency"] = (k_analysis["AvgLatency"] - min_latency) / (max_latency - min_latency)
    k_analysis["TradeoffScore"] = (k_analysis["AvgQuality"] / 5.0) - (0.3 * k_analysis["NormLatency"])
    
    # Identify best K
    best_k = k_analysis["TradeoffScore"].idxmax()
    best_score = k_analysis.loc[best_k, "TradeoffScore"]
    best_quality = k_analysis.loc[best_k, "AvgQuality"]
    best_latency = k_analysis.loc[best_k, "AvgLatency"]
    
    # Save summary
    summary_df = k_analysis.reset_index()
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)
    print(f"\nSaved analysis to {OUTPUT_SUMMARY}")
    
    # Print results
    print("\n" + "="*70)
    print("TOP-K ANALYSIS SUMMARY")
    print("="*70)
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print(f"✓ Best Top-K value: K = {best_k}")
    print(f"  - Average Quality Score: {best_quality:.2f} / 5.0")
    print(f"  - Average Latency: {best_latency:.2f} seconds")
    print(f"  - Trade-off Score: {best_score:.3f}")
    
    print("\nInterpretation:")
    if best_k <= 1:
        print("  → Low K (speed optimized): Fast responses, may lack context.")
    elif best_k <= 3:
        print("  → Moderate K (balanced): Good quality/speed trade-off.")
    else:
        print("  → High K (quality optimized): Rich context, slower responses.")
    
    print("\n" + "="*70)
    
    # Save detailed per-question analysis
    per_question = df.groupby("K").apply(
        lambda x: x[["Question", "QualityScore", "Latency", "Answer"]].to_dict("records")
    )
    
    return {
        "summary": k_analysis,
        "best_k": best_k,
        "best_quality": best_quality,
        "best_latency": best_latency
    }


if __name__ == "__main__":
    analyze_results()
