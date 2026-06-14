from pathlib import Path
import ast
import json
import math
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None
    util = None

from evaluation_dataset import get_reference_map


RESULTS_DIR = Path("results")
INPUT_CSV = RESULTS_DIR / "rag_results.csv"
SUMMARY_CSV = RESULTS_DIR / "topk_summary.csv"
REPORT_TXT = RESULTS_DIR / "rapport_experimentation_topk.txt"
CORPUS_SUMMARY_CSV = RESULTS_DIR / "corpus_summary.csv"
QUALITY_PLOT = RESULTS_DIR / "quality_vs_top_k.png"
TRADEOFF_PLOT = RESULTS_DIR / "quality_latency_tradeoff.png"
RETRIEVAL_PLOT = RESULTS_DIR / "retrieval_precision_recall_vs_top_k.png"
MEMORY_PLOT = RESULTS_DIR / "memory_vs_top_k.png"
DETAILED_CSV = RESULTS_DIR / "rag_results_evaluated.csv"
TTEST_CSV = RESULTS_DIR / "student_t_tests.csv"


def is_insufficient(answer):
    return "insufficient information" in str(answer).lower()


def safe_json_list(value):
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError):
            return [item.strip() for item in text.split(",") if item.strip()]


def tokenize(text):
    return [token for token in str(text).lower().replace("-", " ").split() if token]


def ngrams(tokens, n):
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def bleu_score(reference, answer):
    reference_tokens = tokenize(reference)
    answer_tokens = tokenize(answer)

    if not reference_tokens or not answer_tokens or is_insufficient(answer):
        return 0.0

    precisions = []
    for n in range(1, 5):
        answer_ngrams = Counter(ngrams(answer_tokens, n))
        reference_ngrams = Counter(ngrams(reference_tokens, n))
        total = sum(answer_ngrams.values())
        if total == 0:
            precisions.append(0.0)
            continue
        overlap = sum(min(count, reference_ngrams[gram]) for gram, count in answer_ngrams.items())
        precisions.append((overlap + 1) / (total + 1))

    geo_mean = math.exp(sum(math.log(max(p, 1e-9)) for p in precisions) / 4)
    brevity_penalty = min(1.0, math.exp(1 - len(reference_tokens) / len(answer_tokens)))
    return round(brevity_penalty * geo_mean, 4)


def rouge_l_score(reference, answer):
    reference_tokens = tokenize(reference)
    answer_tokens = tokenize(answer)

    if not reference_tokens or not answer_tokens or is_insufficient(answer):
        return 0.0

    previous = [0] * (len(answer_tokens) + 1)
    for ref_token in reference_tokens:
        current = [0]
        for j, answer_token in enumerate(answer_tokens, start=1):
            if ref_token == answer_token:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current

    lcs = previous[-1]
    precision = lcs / len(answer_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def lexical_similarity(left, right):
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def answer_quality_score(answer):
    text = str(answer).strip()
    lower = text.lower()

    if not text or "insufficient information" in lower:
        return 1

    length = len(text)
    score = 2

    if length >= 80:
        score += 1
    if length >= 180:
        score += 1
    if any(marker in text for marker in [":", ";", "1.", "2.", "-", "\n"]):
        score += 1

    return min(score, 5)


def source_count(sources):
    return len([source.strip() for source in str(sources).split(",") if source.strip()])


def source_diversity(sources):
    values = [source.strip() for source in str(sources).split(",") if source.strip()]
    if not values:
        return 0
    return len(set(values)) / len(values)


def expected_source_metrics(row):
    expected_sources = safe_json_list(row.get("Expected Sources"))
    if not expected_sources:
        reference = get_reference_map().get(row["Question"], {})
        expected_sources = reference.get("expected_sources", [])

    retrieved_sources = [source.strip() for source in str(row["Sources"]).split(",") if source.strip()]
    if not retrieved_sources:
        return pd.Series(
            {
                "Expected Sources Parsed": json.dumps(expected_sources),
                "Expected Source Found": False,
                "Retrieval Precision": 0.0,
                "Retrieval Recall": 0.0,
            }
        )

    expected_set = set(expected_sources)
    retrieved_set = set(retrieved_sources)
    relevant_retrieved = len([source for source in retrieved_sources if source in expected_set])

    return pd.Series(
        {
            "Expected Sources Parsed": json.dumps(expected_sources),
            "Expected Source Found": bool(expected_set & retrieved_set),
            "Retrieval Precision": relevant_retrieved / len(retrieved_sources),
            "Retrieval Recall": len(expected_set & retrieved_set) / len(expected_set) if expected_set else 0.0,
        }
    )


def add_semantic_metrics(df):
    if SentenceTransformer is None:
        enriched = df.copy()
        enriched["Answer Relevance"] = enriched.apply(
            lambda row: round(lexical_similarity(row["Answer"], row["Question"]), 4),
            axis=1,
        )
        enriched["Semantic Similarity"] = enriched.apply(
            lambda row: round(lexical_similarity(row["Answer"], row["Reference Answer"]), 4),
            axis=1,
        )
        enriched["Faithfulness Proxy"] = enriched.apply(
            lambda row: round(
                max(
                    [lexical_similarity(row["Answer"], context) for context in safe_json_list(row.get("Retrieved Contexts"))]
                    or [0.0]
                ),
                4,
            ),
            axis=1,
        )
        enriched["Semantic Metric Backend"] = "lexical_fallback"
        return enriched

    model = SentenceTransformer("all-MiniLM-L6-v2")
    enriched = df.copy()
    answers = enriched["Answer"].fillna("").astype(str).tolist()
    questions = enriched["Question"].fillna("").astype(str).tolist()
    references = enriched["Reference Answer"].fillna("").astype(str).tolist()

    answer_embeddings = model.encode(answers, convert_to_tensor=True, show_progress_bar=False)
    question_embeddings = model.encode(questions, convert_to_tensor=True, show_progress_bar=False)
    reference_embeddings = model.encode(references, convert_to_tensor=True, show_progress_bar=False)

    relevance = util.cos_sim(answer_embeddings, question_embeddings).diagonal().cpu().tolist()
    semantic_similarity = util.cos_sim(answer_embeddings, reference_embeddings).diagonal().cpu().tolist()

    enriched["Answer Relevance"] = [round(max(0.0, value), 4) for value in relevance]
    enriched["Semantic Similarity"] = [round(max(0.0, value), 4) for value in semantic_similarity]

    faithfulness_values = []
    for idx, row in enriched.iterrows():
        contexts = safe_json_list(row.get("Retrieved Contexts"))
        answer = str(row["Answer"])
        if not contexts or is_insufficient(answer):
            faithfulness_values.append(0.0)
            continue

        context_embeddings = model.encode(contexts, convert_to_tensor=True, show_progress_bar=False)
        answer_embedding = answer_embeddings[idx]
        similarities = util.cos_sim(answer_embedding, context_embeddings)[0].cpu().tolist()
        faithfulness_values.append(round(max(0.0, max(similarities)), 4))

    enriched["Faithfulness Proxy"] = faithfulness_values
    enriched["Semantic Metric Backend"] = "sentence_transformers"
    return enriched


def prepare_dataframe(df):
    required_columns = {"K", "Question", "Latency", "Sources", "Answer"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {INPUT_CSV}: {', '.join(sorted(missing))}")

    enriched = df.copy()
    references = get_reference_map()
    if "Reference Answer" not in enriched.columns:
        enriched["Reference Answer"] = enriched["Question"].map(
            lambda question: references.get(question, {}).get("reference_answer", "")
        )
    if "Expected Sources" not in enriched.columns:
        enriched["Expected Sources"] = enriched["Question"].map(
            lambda question: json.dumps(references.get(question, {}).get("expected_sources", []))
        )

    enriched["Insufficient"] = enriched["Answer"].apply(is_insufficient)
    enriched["Answered"] = ~enriched["Insufficient"]
    enriched["Quality Score"] = enriched["Answer"].apply(answer_quality_score)
    enriched["BLEU"] = enriched.apply(lambda row: bleu_score(row["Reference Answer"], row["Answer"]), axis=1)
    enriched["ROUGE-L"] = enriched.apply(lambda row: rouge_l_score(row["Reference Answer"], row["Answer"]), axis=1)
    enriched["Answer Length"] = enriched["Answer"].fillna("").astype(str).str.len()
    enriched["Retrieved Sources"] = enriched["Sources"].apply(source_count)
    enriched["Source Diversity"] = enriched["Sources"].apply(source_diversity)
    retrieval_metrics = enriched.apply(expected_source_metrics, axis=1)
    for column in retrieval_metrics.columns:
        enriched[column] = retrieval_metrics[column]

    if "Memory Delta MB" not in enriched.columns:
        enriched["Memory Delta MB"] = 0.0
    if "Peak Traced Memory MB" not in enriched.columns:
        enriched["Peak Traced Memory MB"] = 0.0
    if "Context Characters" not in enriched.columns:
        enriched["Context Characters"] = 0.0

    for column in ["Memory Delta MB", "Peak Traced Memory MB", "Context Characters"]:
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(0.0)

    return add_semantic_metrics(enriched)


def add_row_scores(df):
    enriched = df.copy()

    if enriched["Latency"].max() == enriched["Latency"].min():
        latency_normalized = pd.Series(0, index=enriched.index)
    else:
        latency_normalized = (
            enriched["Latency"] - enriched["Latency"].min()
        ) / (enriched["Latency"].max() - enriched["Latency"].min())

    enriched["Row_Quality_Normalized"] = (
        0.25 * ((enriched["Quality Score"] - 1) / 4)
        + 0.20 * enriched["ROUGE-L"]
        + 0.15 * enriched["BLEU"]
        + 0.20 * enriched["Answer Relevance"]
        + 0.20 * enriched["Faithfulness Proxy"]
    )
    enriched["Row_Final_Score"] = (
        0.45 * enriched["Row_Quality_Normalized"]
        + 0.20 * enriched["Answered"].astype(float)
        + 0.20 * enriched["Retrieval Recall"]
        + 0.20 * (1 - latency_normalized)
    )
    return enriched


def build_summary(df):
    summary = (
        df.groupby("K")
        .agg(
            Questions=("Question", "count"),
            Avg_Latency=("Latency", "mean"),
            Std_Latency=("Latency", "std"),
            Min_Latency=("Latency", "min"),
            Max_Latency=("Latency", "max"),
            Answered=("Answered", "sum"),
            Insufficient=("Insufficient", "sum"),
            Avg_Quality=("Quality Score", "mean"),
            Std_Quality=("Quality Score", "std"),
            Avg_BLEU=("BLEU", "mean"),
            Avg_ROUGE_L=("ROUGE-L", "mean"),
            Std_ROUGE_L=("ROUGE-L", "std"),
            Avg_Answer_Relevance=("Answer Relevance", "mean"),
            Avg_Faithfulness_Proxy=("Faithfulness Proxy", "mean"),
            Avg_Semantic_Similarity=("Semantic Similarity", "mean"),
            Expected_Source_Found_Rate=("Expected Source Found", "mean"),
            Avg_Retrieval_Precision=("Retrieval Precision", "mean"),
            Avg_Retrieval_Recall=("Retrieval Recall", "mean"),
            Avg_Answer_Length=("Answer Length", "mean"),
            Avg_Source_Diversity=("Source Diversity", "mean"),
            Avg_Memory_Delta_MB=("Memory Delta MB", "mean"),
            Avg_Peak_Traced_Memory_MB=("Peak Traced Memory MB", "mean"),
            Avg_Context_Characters=("Context Characters", "mean"),
        )
        .reset_index()
    )

    summary["Answer_Coverage"] = summary["Answered"] / summary["Questions"]
    summary["CI95_Latency"] = 1.96 * summary["Std_Latency"].fillna(0) / summary["Questions"].pow(0.5)
    summary["CI95_Quality"] = 1.96 * summary["Std_Quality"].fillna(0) / summary["Questions"].pow(0.5)
    summary["CI95_ROUGE_L"] = 1.96 * summary["Std_ROUGE_L"].fillna(0) / summary["Questions"].pow(0.5)
    summary["Latency_Normalized"] = (
        summary["Avg_Latency"] - summary["Avg_Latency"].min()
    ) / (summary["Avg_Latency"].max() - summary["Avg_Latency"].min())

    if summary["Avg_Latency"].max() == summary["Avg_Latency"].min():
        summary["Latency_Normalized"] = 0

    summary["Quality_Normalized"] = (
        0.25 * ((summary["Avg_Quality"] - 1) / 4)
        + 0.20 * summary["Avg_ROUGE_L"]
        + 0.15 * summary["Avg_BLEU"]
        + 0.20 * summary["Avg_Answer_Relevance"]
        + 0.20 * summary["Avg_Faithfulness_Proxy"]
    )
    summary["Final_Score"] = (
        0.45 * summary["Quality_Normalized"]
        + 0.20 * summary["Answer_Coverage"]
        + 0.20 * summary["Avg_Retrieval_Recall"]
        + 0.20 * (1 - summary["Latency_Normalized"])
    )

    return summary.sort_values("K")


def select_best_k(summary):
    ranked = summary.sort_values(
        ["Final_Score", "Answer_Coverage", "Avg_Quality", "Avg_Latency"],
        ascending=[False, False, False, True],
    )
    return ranked.iloc[0]


def normal_approx_p_value(t_stat):
    return math.erfc(abs(t_stat) / math.sqrt(2))


def paired_student_t_tests(enriched, best_k):
    rows = []
    best_scores = enriched[enriched["K"] == best_k][["Question", "Row_Final_Score"]]
    best_scores = best_scores.rename(columns={"Row_Final_Score": "Best_Row_Final_Score"})

    for k in sorted(enriched["K"].unique()):
        if int(k) == int(best_k):
            continue

        candidate = enriched[enriched["K"] == k][["Question", "Row_Final_Score"]]
        candidate = candidate.rename(columns={"Row_Final_Score": "Candidate_Row_Final_Score"})
        paired = best_scores.merge(candidate, on="Question", how="inner")
        differences = paired["Best_Row_Final_Score"] - paired["Candidate_Row_Final_Score"]
        n = len(differences)
        mean_difference = differences.mean() if n else 0
        std_difference = differences.std(ddof=1) if n > 1 else 0

        if n > 1 and std_difference > 0:
            t_stat = mean_difference / (std_difference / math.sqrt(n))
            try:
                from scipy import stats

                p_value = float(stats.t.sf(abs(t_stat), df=n - 1) * 2)
                p_method = "scipy_t_distribution"
            except ImportError:
                p_value = normal_approx_p_value(t_stat)
                p_method = "normal_approximation"
        else:
            t_stat = 0
            p_value = 1
            p_method = "not_enough_variance"

        rows.append(
            {
                "Best K": int(best_k),
                "Compared K": int(k),
                "Paired Questions": n,
                "Mean Difference": mean_difference,
                "Std Difference": std_difference,
                "T Statistic": t_stat,
                "P Value": p_value,
                "P Value Method": p_method,
                "Significant 0.05": p_value < 0.05,
            }
        )

    return pd.DataFrame(rows)


def save_plots(summary):
    plt.figure(figsize=(8, 5))
    plt.plot(summary["K"], summary["Avg_Quality"], marker="o", color="#2f6f5e", label="Quality")
    plt.plot(summary["K"], summary["Answer_Coverage"] * 5, marker="s", color="#4b5f9f", label="Coverage x5")
    plt.title("Quality and Answer Coverage vs Top-K")
    plt.xlabel("Top-K")
    plt.ylabel("Score")
    plt.xticks(summary["K"])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(QUALITY_PLOT, dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    sizes = 120 + summary["Answer_Coverage"] * 260
    plt.scatter(summary["Avg_Latency"], summary["Avg_Quality"], s=sizes, color="#8b5a2b", alpha=0.85)

    for _, row in summary.iterrows():
        plt.annotate(f"K={int(row['K'])}", (row["Avg_Latency"], row["Avg_Quality"]), xytext=(7, 5), textcoords="offset points")

    plt.title("Quality / Performance Tradeoff")
    plt.xlabel("Average latency (seconds)")
    plt.ylabel("Average quality score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(TRADEOFF_PLOT, dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(summary["K"], summary["Avg_Retrieval_Precision"], marker="o", label="Precision")
    plt.plot(summary["K"], summary["Avg_Retrieval_Recall"], marker="s", label="Recall")
    plt.plot(summary["K"], summary["Expected_Source_Found_Rate"], marker="^", label="Expected source found")
    plt.title("Retrieval Precision and Recall vs Top-K")
    plt.xlabel("Top-K")
    plt.ylabel("Score")
    plt.xticks(summary["K"])
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RETRIEVAL_PLOT, dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(summary["K"], summary["Avg_Memory_Delta_MB"], marker="o", label="RSS delta")
    plt.plot(summary["K"], summary["Avg_Peak_Traced_Memory_MB"], marker="s", label="Peak traced memory")
    plt.title("Memory Usage vs Top-K")
    plt.xlabel("Top-K")
    plt.ylabel("Memory (MB)")
    plt.xticks(summary["K"])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(MEMORY_PLOT, dpi=160)
    plt.close()


def format_percent(value):
    return f"{value * 100:.0f}%"


def truncate_text(text, max_chars=420):
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def find_qualitative_examples(enriched, best_k):
    examples = []
    preferred_k = [1, int(best_k), 20]

    for question, group in enriched.groupby("Question"):
        available = set(group["K"].astype(int).tolist())
        if not set(preferred_k).issubset(available):
            continue

        k1 = group[group["K"] == 1].iloc[0]
        best = group[group["K"] == int(best_k)].iloc[0]
        k20 = group[group["K"] == 20].iloc[0]

        if (not bool(k1["Answered"]) and bool(best["Answered"])) or (
            best["Quality Score"] > k1["Quality Score"]
            and best["Latency"] < k20["Latency"]
        ):
            examples.append((question, k1, best, k20))

        if len(examples) == 2:
            break

    return examples


def build_report(summary, best, enriched, t_tests):
    fastest = summary.sort_values("Avg_Latency").iloc[0]
    highest_quality = summary.sort_values(["Avg_Quality", "Answer_Coverage"], ascending=False).iloc[0]
    has_context_metrics = summary["Avg_Context_Characters"].sum() > 0
    has_memory_metrics = (
        summary["Avg_Memory_Delta_MB"].abs().sum()
        + summary["Avg_Peak_Traced_Memory_MB"].abs().sum()
    ) > 0
    max_recall = summary["Avg_Retrieval_Recall"].max()
    saturation_threshold = max_recall * 0.95
    saturation = summary[summary["Avg_Retrieval_Recall"] >= saturation_threshold].sort_values("K").iloc[0]
    rouge_peak = summary.sort_values("Avg_ROUGE_L", ascending=False).iloc[0]
    examples = find_qualitative_examples(enriched, int(best["K"]))
    corpus_lines = [
        "Le resume du corpus n'est pas disponible. Relancer main.py pour generer results/corpus_summary.csv."
    ]
    if CORPUS_SUMMARY_CSV.exists():
        corpus = pd.read_csv(CORPUS_SUMMARY_CSV).iloc[0]
        corpus_lines = [
            f"Nombre de documents: {int(corpus['Documents'])} "
            f"(objectif academique: {corpus['Target Documents']}).",
            f"Nombre de chunks: {int(corpus['Chunks'])} "
            f"(objectif academique: {corpus['Target Chunks']}).",
            f"Taille des chunks: {int(corpus['Chunk Size'])} caracteres, "
            f"overlap: {int(corpus['Chunk Overlap'])} caracteres.",
            f"Objectif corpus respecte: {bool(corpus['Academic Target Met'])}.",
        ]

    lines = [
        "Rapport d'experimentation RAG: analyse du parametre Top-K",
        "=" * 62,
        "",
        "Objectif du projet",
        "-----------------",
        "Ce projet etudie l'impact du parametre Top-K dans une solution RAG.",
        "L'experience compare plusieurs valeurs de K afin de mesurer:",
        "- l'impact d'un K faible par rapport a un K eleve;",
        "- le compromis entre qualite des reponses et performance;",
        "- la valeur de Top-K la plus pertinente pour ce corpus.",
        "",
        "Methodologie",
        "-----------",
        "Le pipeline charge les documents du dossier data/, les decoupe en chunks,",
        "calcule des embeddings avec sentence-transformers, indexe les chunks dans",
        "ChromaDB, puis interroge un modele LLM avec les contextes recuperes.",
        "Les valeurs de K comparees sont: "
        + ", ".join(str(int(k)) for k in summary["K"].tolist())
        + ".",
        f"Chaque valeur de K est evaluee sur {int(summary['Questions'].iloc[0])} questions.",
        (
            "Les contextes recuperes et les mesures memoire sont disponibles pour cette experience."
            if has_context_metrics and has_memory_metrics
            else "Remarque: si le rapport est regenere depuis un ancien CSV, les colonnes contexts/memoire peuvent etre indisponibles."
        ),
        "",
        "Dimensionnement du corpus",
        "------------------------",
        *corpus_lines,
        "",
        "Indicateurs utilises",
        "-------------------",
        "- Latence moyenne: temps moyen necessaire pour recuperer le contexte et generer la reponse.",
        "- Taux de couverture: proportion de questions ayant recu une reponse exploitable.",
        "- Qualite: score heuristique, BLEU, ROUGE-L, similarite semantique,",
        "  answer relevance et faithfulness proxy.",
        "- Performance: latence moyenne et memoire utilisee.",
        "- Retrieval: precision, rappel et verification de la presence de la bonne source.",
        "- Score final: combinaison ponderee de la qualite, du retrieval, de la couverture et de la latence.",
        "",
        "Resultats synthetiques",
        "---------------------",
        "K | Latence | Memoire pic MB | Couverture | ROUGE-L | BLEU | Relevance | Faithfulness | Recall retrieval | Score final",
        "--|---------|----------------|------------|---------|------|-----------|--------------|------------------|------------",
    ]

    for _, row in summary.iterrows():
        lines.append(
            f"{int(row['K'])} | {row['Avg_Latency']:.3f}s | "
            f"{row['Avg_Peak_Traced_Memory_MB']:.3f} | "
            f"{format_percent(row['Answer_Coverage'])} | "
            f"{row['Avg_ROUGE_L']:.3f} | {row['Avg_BLEU']:.3f} | "
            f"{row['Avg_Answer_Relevance']:.3f} | {row['Avg_Faithfulness_Proxy']:.3f} | "
            f"{row['Avg_Retrieval_Recall']:.3f} | {row['Final_Score']:.3f}"
        )

    lines.extend(
        [
            "",
            "Robustesse statistique",
            "----------------------",
            "Les valeurs du tableau sont des moyennes calculees pour chaque valeur de K.",
            "Comme l'experience utilise un nombre limite de questions, il faut interpreter",
            "les resultats comme une comparaison experimentale sur ce corpus, et non comme",
            "une preuve universelle. Pour renforcer l'analyse, le tableau ci-dessous ajoute",
            "l'ecart-type et un intervalle de confiance approximatif a 95%.",
            "",
            "K | Latence moyenne +/- IC95 | Qualite moyenne +/- IC95 | ROUGE-L moyen +/- IC95",
            "--|--------------------------|--------------------------|----------------------",
        ]
    )

    for _, row in summary.iterrows():
        lines.append(
            f"{int(row['K'])} | {row['Avg_Latency']:.3f}s +/- {row['CI95_Latency']:.3f} | "
            f"{row['Avg_Quality']:.2f}/5 +/- {row['CI95_Quality']:.2f} | "
            f"{row['Avg_ROUGE_L']:.3f} +/- {row['CI95_ROUGE_L']:.3f}"
        )

    lines.extend(
        [
            "",
            "Test t de Student",
            "-----------------",
            f"Un test t apparie compare le score final par question de K={int(best['K'])}",
            "avec chaque autre valeur de K. Le test est apparie car les memes questions",
            "sont evaluees pour toutes les valeurs de K.",
            "",
            "K compare | Difference moyenne | t | p-value | Significatif 0.05",
            "----------|--------------------|---|---------|------------------",
        ]
    )

    for _, row in t_tests.iterrows():
        lines.append(
            f"{int(row['Compared K'])} | {row['Mean Difference']:.4f} | "
            f"{row['T Statistic']:.3f} | {row['P Value']:.4f} | "
            f"{bool(row['Significant 0.05'])}"
        )

    lines.extend(
        [
            "",
            "Analyse de l'impact de K",
            "-----------------------",
            f"Le K le plus rapide est K={int(fastest['K'])}, avec une latence moyenne de "
            f"{fastest['Avg_Latency']:.3f}s. Cependant, sa couverture est de "
            f"{format_percent(fastest['Answer_Coverage'])}, ce qui montre qu'un K trop faible "
            "peut manquer de contexte et produire davantage de reponses insuffisantes.",
            "",
            f"Le meilleur niveau de qualite brute est observe avec K={int(highest_quality['K'])}, "
            f"avec une qualite moyenne de {highest_quality['Avg_Quality']:.2f}/5 et une couverture "
            f"de {format_percent(highest_quality['Answer_Coverage'])}. Cette valeur recupere plus "
            "de contexte, mais elle peut augmenter fortement la latence.",
            "",
            "Pollution du contexte",
            "---------------------",
            f"Le score ROUGE-L atteint son maximum a K={int(rouge_peak['K'])} "
            f"({rouge_peak['Avg_ROUGE_L']:.3f}), puis diminue pour plusieurs valeurs plus elevees.",
            "Ce comportement illustre le phenomene de pollution du contexte: quand K augmente,",
            "le systeme recupere davantage de chunks, mais tous ne sont pas egalement utiles.",
            "Des passages secondaires ou redondants peuvent ajouter du bruit dans le prompt,",
            "ce qui peut rendre la reponse moins proche de la reponse de reference malgre un",
            "meilleur rappel du retrieval.",
            "",
            "Courbe de saturation",
            "--------------------",
            f"Le rappel du retrieval atteint un palier important a partir de K={int(saturation['K'])} "
            f"avec un rappel moyen de {saturation['Avg_Retrieval_Recall']:.3f}.",
            f"Au-dela, le gain reste limite jusqu'au maximum observe de {max_recall:.3f},",
            "tandis que la latence et la taille du contexte continuent d'augmenter.",
            f"K={int(saturation['K'])} est donc le point de saturation du retrieval, mais",
            f"la recommandation finale reste K={int(best['K'])}, car le score global prend",
            "aussi en compte la qualite des reponses, la couverture, la latence et le",
            "rappel du retrieval.",
            "",
            "Compromis qualite / performance",
            "------------------------------",
            "L'augmentation de K ameliore generalement la couverture des reponses, car le modele",
            "dispose de plus d'informations. En revanche, un K eleve agrandit le prompt, ajoute",
            "potentiellement du bruit documentaire et augmente le temps de reponse.",
            "Les metriques BLEU et ROUGE-L comparent la reponse generee a une reponse de reference.",
            "Les scores answer relevance et faithfulness proxy mesurent respectivement la proximite",
            "semantique avec la question et la coherence avec les contextes recuperes.",
            (
                "Les contextes recuperes sont disponibles dans ce CSV, donc le faithfulness proxy est calcule."
                if has_context_metrics
                else "Les contextes recuperes ne sont pas presents dans l'ancien CSV; le faithfulness proxy sera complet apres un nouveau lancement de main.py."
            ),
            (
                "Les mesures memoire sont disponibles pour cette experience."
                if has_memory_metrics
                else "Les mesures memoire ne sont pas presentes dans l'ancien CSV; elles seront completes apres un nouveau lancement de main.py."
            ),
            "",
            "Precision du retrieval",
            "----------------------",
            "La precision du retrieval mesure la proportion de passages recuperes provenant des sources",
            "attendues. Le rappel mesure la proportion des sources attendues retrouvees par le systeme.",
            "Quand K augmente, le rappel tend a progresser, mais la precision peut diminuer car le systeme",
            "ajoute aussi des passages moins directement utiles.",
            "",
            "Analyse qualitative",
            "-------------------",
        ]
    )

    if examples:
        for idx, (question, k1, best_row, k20) in enumerate(examples, start=1):
            lines.extend(
                [
                    f"Exemple {idx}: {question}",
                    f"- K=1: latence {k1['Latency']:.3f}s, recall retrieval {k1['Retrieval Recall']:.3f}, reponse: {truncate_text(k1['Answer'])}",
                    f"- K={int(best['K'])}: latence {best_row['Latency']:.3f}s, recall retrieval {best_row['Retrieval Recall']:.3f}, reponse: {truncate_text(best_row['Answer'])}",
                    f"- K=20: latence {k20['Latency']:.3f}s, recall retrieval {k20['Retrieval Recall']:.3f}, reponse: {truncate_text(k20['Answer'])}",
                    "Cet exemple montre que le K recommande apporte plus de contexte utile que K=1,",
                    "tout en evitant le cout important observe avec K=20.",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "Aucun exemple automatique K=1 / K recommande / K=20 n'a pu etre extrait.",
                "Les reponses detaillees restent disponibles dans results/rag_results_evaluated.csv.",
                "",
            ]
        )

    lines.extend(
        [
            "Valeur Top-K recommandee",
            "-----------------------",
            f"La valeur la plus pertinente selon le score final est K={int(best['K'])}.",
            f"Elle obtient une latence moyenne de {best['Avg_Latency']:.3f}s, une couverture de "
            f"{format_percent(best['Answer_Coverage'])}, une qualite moyenne de "
            f"{best['Avg_Quality']:.2f}/5 et un score final de {best['Final_Score']:.3f}.",
            "",
            "Conclusion",
            "----------",
            f"Pour ce corpus et cette serie de questions, K={int(best['K'])} represente le meilleur "
            "compromis entre qualite et performance. Un K faible favorise la rapidite, mais limite",
            "la quantite de contexte disponible. Un K eleve ameliore la couverture, mais augmente",
            "nettement la latence. Le Top-K recommande est donc celui qui maximise la qualite utile",
            "sans penaliser excessivement le temps de reponse.",
            "",
            "Limites du projet",
            "-----------------",
            "- Le corpus reste limite a un ensemble de documents sur l'IA et le RAG.",
            "- Les questions sont controlees et synthetiques; des questions d'utilisateurs reels",
            "  pourraient produire des comportements differents.",
            "- Les resultats dependent du modele LLM utilise, ici configure via Groq.",
            "- Les latences peuvent varier selon l'API, le reseau et la charge du service.",
            "- Les metriques automatiques comme BLEU et ROUGE-L ne remplacent pas totalement",
            "  une evaluation humaine de la pertinence et de la clarte des reponses.",
            "- Les tests statistiques indiquent si les ecarts observes sont importants sur",
            "  cet echantillon, mais ils restent dependants du nombre de questions et du",
            "  choix du score final.",
            "",
            "Fichiers generes",
            "---------------",
            f"- {SUMMARY_CSV.as_posix()}",
            f"- {CORPUS_SUMMARY_CSV.as_posix()}",
            f"- {QUALITY_PLOT.as_posix()}",
            f"- {TRADEOFF_PLOT.as_posix()}",
            f"- {RETRIEVAL_PLOT.as_posix()}",
            f"- {MEMORY_PLOT.as_posix()}",
            f"- {DETAILED_CSV.as_posix()}",
            f"- {TTEST_CSV.as_posix()}",
            f"- {REPORT_TXT.as_posix()}",
        ]
    )

    return "\n".join(lines) + "\n"


def analyze_results(input_csv=INPUT_CSV):
    RESULTS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(input_csv)
    enriched = prepare_dataframe(df)
    enriched = add_row_scores(enriched)
    summary = build_summary(enriched)
    best = select_best_k(summary)
    t_tests = paired_student_t_tests(enriched, int(best["K"]))

    enriched.to_csv(DETAILED_CSV, index=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    t_tests.to_csv(TTEST_CSV, index=False)
    save_plots(summary)
    REPORT_TXT.write_text(build_report(summary, best, enriched, t_tests), encoding="utf-8")

    return summary, best


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_CSV}")

    summary, best = analyze_results(INPUT_CSV)
    print(f"Saved {SUMMARY_CSV}")
    print(f"Saved {QUALITY_PLOT}")
    print(f"Saved {TRADEOFF_PLOT}")
    print(f"Saved {REPORT_TXT}")
    print(f"Recommended Top-K: {int(best['K'])}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
