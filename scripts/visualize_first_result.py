import textwrap
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results")
INPUT_CSV = RESULTS_DIR / "rag_results.csv"
BASELINE_K = 1


def wrap_labels(labels, width=28):
    return ["\n".join(textwrap.wrap(label, width=width)) for label in labels]


def is_insufficient(answer):
    return "insufficient information" in str(answer).lower()


def save_latency_chart(baseline):
    plot_data = baseline.sort_values("Latency", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#3f7f93" if not value else "#b44b4b" for value in plot_data["Insufficient"]]

    ax.barh(wrap_labels(plot_data["Question"]), plot_data["Latency"], color=colors)
    ax.set_title("Baseline K=1: Latency by Question")
    ax.set_xlabel("Latency (seconds)")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)

    answered_patch = plt.Rectangle((0, 0), 1, 1, color="#3f7f93")
    insufficient_patch = plt.Rectangle((0, 0), 1, 1, color="#b44b4b")
    ax.legend(
        [answered_patch, insufficient_patch],
        ["Answered", "Insufficient information"],
        loc="lower right",
    )

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "baseline_k1_latency_by_question.png", dpi=160)
    plt.close(fig)


def save_coverage_chart(baseline):
    counts = baseline["Answer Status"].value_counts().reindex(
        ["Answered", "Insufficient information"],
        fill_value=0,
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#3f7f93", "#b44b4b"]
    wedges, _, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        colors=colors,
        autopct=lambda pct: f"{pct:.0f}%" if pct > 0 else "",
        startangle=90,
        counterclock=False,
    )
    for text in autotexts:
        text.set_color("white")
        text.set_weight("bold")

    ax.set_title("Baseline K=1: Answer Coverage")
    ax.legend(wedges, [f"{label}: {value}" for label, value in counts.items()], loc="lower center")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "baseline_k1_answer_coverage.png", dpi=160)
    plt.close(fig)


def save_source_chart(baseline):
    sources = []
    for value in baseline["Sources"].dropna():
        sources.extend(source.strip() for source in value.split(",") if source.strip())

    counts = Counter(sources)
    source_names = list(counts.keys())
    source_values = list(counts.values())

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(wrap_labels(source_names, width=18), source_values, color="#4f6f9f")
    ax.set_title("Baseline K=1: Retrieved Source Frequency")
    ax.set_xlabel("")
    ax.set_ylabel("Retrieval count")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "baseline_k1_source_frequency.png", dpi=160)
    plt.close(fig)


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_CSV}")

    RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    baseline = df[df["K"] == BASELINE_K].copy()

    if baseline.empty:
        raise ValueError(f"No rows found for K={BASELINE_K} in {INPUT_CSV}")

    baseline["Insufficient"] = baseline["Answer"].apply(is_insufficient)
    baseline["Answer Status"] = baseline["Insufficient"].map(
        {True: "Insufficient information", False: "Answered"}
    )
    baseline["Answer Length"] = baseline["Answer"].fillna("").str.len()

    summary_path = RESULTS_DIR / "baseline_k1_summary.csv"
    baseline[
        ["Question", "Latency", "Sources", "Answer Status", "Answer Length", "Answer"]
    ].to_csv(summary_path, index=False)

    save_latency_chart(baseline)
    save_coverage_chart(baseline)
    save_source_chart(baseline)

    print(f"Saved {summary_path}")
    print("Saved results/baseline_k1_latency_by_question.png")
    print("Saved results/baseline_k1_answer_coverage.png")
    print("Saved results/baseline_k1_source_frequency.png")


if __name__ == "__main__":
    main()
