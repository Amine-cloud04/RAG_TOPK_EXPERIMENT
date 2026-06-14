# ============================================================
# RAG TOP-K PROJECT (USING YOUR DATA FOLDER)
# Full Version adapted to your 15 txt files
# ============================================================

# INSTALL:
# pip install chromadb sentence-transformers pandas matplotlib requests

import os
import re
import json
import time
import tracemalloc
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import chromadb

from analyze_topk_results import analyze_results
from evaluation_dataset import EVALUATION_QUESTIONS

try:
    import psutil
except ImportError:
    psutil = None

# ============================================================
# ENV
# ============================================================

def load_env_file(path=".env"):
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as env_file:
        for line in env_file:
            line = line.strip()

            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

load_env_file()

# ============================================================
# CONFIG
# ============================================================

DATA_FOLDER = "data"          # folder containing your txt files
DATA_FOLDER = os.getenv("DATA_FOLDER", DATA_FOLDER)
RESULTS_FOLDER = os.getenv("RESULTS_FOLDER", "results")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

TOP_K_VALUES = [1, 2, 3, 5, 7, 10, 15, 20]
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
EVALUATION_QUESTION_LIMIT = int(os.getenv("EVALUATION_QUESTION_LIMIT", "30"))
GROQ_CALL_DELAY_SECONDS = float(os.getenv("GROQ_CALL_DELAY_SECONDS", "2"))
MAX_RATE_LIMIT_WAIT_SECONDS = float(os.getenv("MAX_RATE_LIMIT_WAIT_SECONDS", "600"))
ACADEMIC_MIN_DOCS = 15
ACADEMIC_MAX_DOCS = 30
ACADEMIC_MIN_CHUNKS = 500
ACADEMIC_MAX_CHUNKS = 1500

os.makedirs(RESULTS_FOLDER, exist_ok=True)

if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
    raise RuntimeError(
        "Missing GROQ_API_KEY. Set it in your terminal before running, for example:\n"
        '$env:GROQ_API_KEY="your_groq_api_key"'
    )

if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
    raise RuntimeError(
        "Missing GEMINI_API_KEY. Set it in your terminal before running, for example:\n"
        '$env:GEMINI_API_KEY="your_gemini_api_key"'
    )

if LLM_PROVIDER not in {"groq", "gemini"}:
    raise RuntimeError("LLM_PROVIDER must be either 'groq' or 'gemini'.")

# ============================================================
# LOAD FILES
# ============================================================

def load_documents(folder):

    docs = []

    files = os.listdir(folder)

    for file in files:

        if file.endswith(".txt"):

            path = os.path.join(folder, file)

            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            docs.append({
                "filename": file,
                "content": text
            })

    return docs

documents = load_documents(DATA_FOLDER)

print("Loaded files:", len(documents))

# ============================================================
# CHUNKING
# ============================================================

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

all_chunks = []

for doc in documents:

    chunks = split_text(doc["content"])

    for c in chunks:

        all_chunks.append({
            "source": doc["filename"],
            "text": c
        })

print("Total chunks:", len(all_chunks))

corpus_status = (
    ACADEMIC_MIN_DOCS <= len(documents) <= ACADEMIC_MAX_DOCS
    and ACADEMIC_MIN_CHUNKS <= len(all_chunks) <= ACADEMIC_MAX_CHUNKS
)

corpus_summary = pd.DataFrame(
    [
        {
            "Documents": len(documents),
            "Chunks": len(all_chunks),
            "Chunk Size": CHUNK_SIZE,
            "Chunk Overlap": CHUNK_OVERLAP,
            "Target Documents": f"{ACADEMIC_MIN_DOCS}-{ACADEMIC_MAX_DOCS}",
            "Target Chunks": f"{ACADEMIC_MIN_CHUNKS}-{ACADEMIC_MAX_CHUNKS}",
            "Academic Target Met": corpus_status,
        }
    ]
)
corpus_summary_path = os.path.join(RESULTS_FOLDER, "corpus_summary.csv")
corpus_summary.to_csv(corpus_summary_path, index=False)

if not corpus_status:
    raise RuntimeError(
        "Corpus does not match the academic target. "
        f"Documents: {len(documents)} target {ACADEMIC_MIN_DOCS}-{ACADEMIC_MAX_DOCS}; "
        f"chunks: {len(all_chunks)} target {ACADEMIC_MIN_CHUNKS}-{ACADEMIC_MAX_CHUNKS}."
    )

print(
    "Academic corpus target met:",
    f"{len(documents)} docs and {len(all_chunks)} chunks",
    f"(target: {ACADEMIC_MIN_DOCS}-{ACADEMIC_MAX_DOCS} docs,",
    f"{ACADEMIC_MIN_CHUNKS}-{ACADEMIC_MAX_CHUNKS} chunks).",
)
print(f"Saved {corpus_summary_path}")

# ============================================================
# EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ============================================================
# VECTOR DATABASE
# ============================================================

client = chromadb.Client()

collection = client.create_collection(name="rag_project")

for i, chunk in enumerate(all_chunks):

    emb = embedding_model.encode(chunk["text"]).tolist()

    collection.add(
        ids=[str(i)],
        documents=[chunk["text"]],
        metadatas=[{"source": chunk["source"]}],
        embeddings=[emb]
    )

print("Vector DB ready.")

# ============================================================
# TEST QUESTIONS
# ============================================================

questions = EVALUATION_QUESTIONS[:EVALUATION_QUESTION_LIMIT]

print(
    f"Evaluation questions: {len(questions)} "
    f"of {len(EVALUATION_QUESTIONS)} available "
    f"(set EVALUATION_QUESTION_LIMIT to change this)."
)
print(f"Groq delay between calls: {GROQ_CALL_DELAY_SECONDS}s")

# ============================================================
# LLM CALL
# ============================================================

def parse_retry_after_seconds(message):
    text = str(message).lower()
    match = re.search(
        r"try again in (?:(?P<minutes>[\d.]+)m)?(?:(?P<seconds>[\d.]+)s)?",
        text,
    )
    if not match:
        return None

    minutes = float(match.group("minutes") or 0)
    seconds = float(match.group("seconds") or 0)
    return minutes * 60 + seconds


def ask_groq(prompt, max_retries=20):

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
    }

    for attempt in range(max_retries + 1):
        r = requests.post(url, headers=headers, json=payload, timeout=60)

        try:
            result = r.json()
        except ValueError:
            result = {"error": {"message": r.text}}

        if r.status_code == 200 and "choices" in result:
            return result["choices"][0]["message"]["content"]

        message = result.get("error", {}).get("message", result)

        if r.status_code == 429 and attempt < max_retries:
            retry_after = parse_retry_after_seconds(message)
            wait_seconds = retry_after + 2 if retry_after is not None else 10 + attempt * 5
            if wait_seconds > MAX_RATE_LIMIT_WAIT_SECONDS:
                raise RuntimeError(
                    "Groq rate limit wait is too long "
                    f"({round(wait_seconds, 2)} seconds). "
                    "Checkpoint saved; rerun main.py later to resume."
                )
            print(f"Rate limit reached. Retrying in {round(wait_seconds, 2)} seconds...")
            time.sleep(wait_seconds)
            continue

        raise RuntimeError(f"Groq API request failed ({r.status_code}): {message}")


def ask_gemini(prompt, max_retries=10):
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0
        },
    }

    for attempt in range(max_retries + 1):
        r = requests.post(url, headers=headers, json=payload, timeout=60)

        try:
            result = r.json()
        except ValueError:
            result = {"error": {"message": r.text}}

        if r.status_code == 200 and result.get("candidates"):
            parts = result["candidates"][0].get("content", {}).get("parts", [])
            return "\n".join(part.get("text", "") for part in parts).strip()

        message = result.get("error", {}).get("message", result)

        if r.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
            wait_seconds = min(10 + attempt * 5, MAX_RATE_LIMIT_WAIT_SECONDS)
            print(f"Gemini/API limit reached. Retrying in {round(wait_seconds, 2)} seconds...")
            time.sleep(wait_seconds)
            continue

        raise RuntimeError(f"Gemini API request failed ({r.status_code}): {message}")


def ask_llm(prompt):
    if LLM_PROVIDER == "gemini":
        return ask_gemini(prompt)
    return ask_groq(prompt)

# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(query, k):

    q_emb = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k
    )

    docs = results["documents"][0]
    meta = results["metadatas"][0]
    distances = results["distances"][0] if "distances" in results else []

    return docs, meta, distances

# ============================================================
# PROMPT
# ============================================================

def build_prompt(question, docs):

    context = "\n\n".join(docs)

    prompt = f"""
You are a precise assistant.

Use ONLY the context below.

If answer not found, say insufficient information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    return prompt

# ============================================================
# EXPERIMENT LOOP
# ============================================================

csv_path = os.path.join(RESULTS_FOLDER, "rag_results.csv")

if os.path.exists(csv_path):
    existing_df = pd.read_csv(csv_path)
    results = existing_df.to_dict("records")
    completed = {
        (int(row["K"]), row["Question"])
        for _, row in existing_df.iterrows()
        if "K" in row and "Question" in row
    }
    print(f"Resume mode: loaded {len(results)} existing rows from {csv_path}")
else:
    results = []
    completed = set()


def save_results_checkpoint():
    checkpoint_df = pd.DataFrame(results)
    if not checkpoint_df.empty:
        checkpoint_df = checkpoint_df.drop_duplicates(["K", "Question"], keep="last")
    checkpoint_df.to_csv(csv_path, index=False)
    return checkpoint_df


process = psutil.Process(os.getpid()) if psutil else None

for k in TOP_K_VALUES:

    print("\n==========================")
    print("Testing K =", k)
    print("==========================")

    for item in questions:
        q = item["question"]
        expected_sources = item["expected_sources"]
        reference_answer = item["reference_answer"]

        if (k, q) in completed:
            print(f"Skipping completed result: K={k} | {q}")
            continue

        memory_before_mb = process.memory_info().rss / (1024 * 1024) if process else None
        tracemalloc.start()

        start = time.time()

        docs, meta, distances = retrieve(q, k)

        retrieval_done = time.time()

        prompt = build_prompt(q, docs)

        try:
            answer = ask_llm(prompt)
        except Exception:
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            save_results_checkpoint()
            print(f"Saved checkpoint before stopping: {csv_path}")
            raise

        generation_done = time.time()
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_after_mb = process.memory_info().rss / (1024 * 1024) if process else None

        retrieval_latency = round(retrieval_done - start, 3)
        generation_latency = round(generation_done - retrieval_done, 3)
        latency = round(generation_done - start, 3)

        sources = ", ".join([m["source"] for m in meta])
        retrieved_sources = [m["source"] for m in meta]
        expected_set = set(expected_sources)
        retrieved_set = set(retrieved_sources)
        relevant_retrieved = len([source for source in retrieved_sources if source in expected_set])
        expected_found = bool(expected_set & retrieved_set)
        retrieval_precision = relevant_retrieved / len(retrieved_sources) if retrieved_sources else 0
        retrieval_recall = len(expected_set & retrieved_set) / len(expected_set) if expected_set else 0

        results.append({
            "K": k,
            "Question": q,
            "LLM Provider": LLM_PROVIDER,
            "Model": GEMINI_MODEL if LLM_PROVIDER == "gemini" else MODEL_NAME,
            "Data Folder": DATA_FOLDER,
            "Chunk Size": CHUNK_SIZE,
            "Chunk Overlap": CHUNK_OVERLAP,
            "Reference Answer": reference_answer,
            "Expected Sources": json.dumps(expected_sources),
            "Latency": latency,
            "Retrieval Latency": retrieval_latency,
            "Generation Latency": generation_latency,
            "Context Characters": len("\n\n".join(docs)),
            "Retrieved Chunks": len(docs),
            "Memory Before MB": round(memory_before_mb, 3) if memory_before_mb is not None else "",
            "Memory After MB": round(memory_after_mb, 3) if memory_after_mb is not None else "",
            "Memory Delta MB": round(memory_after_mb - memory_before_mb, 3) if memory_before_mb is not None and memory_after_mb is not None else "",
            "Peak Traced Memory MB": round(peak_memory / (1024 * 1024), 3),
            "Expected Source Found": expected_found,
            "Retrieval Precision": round(retrieval_precision, 4),
            "Retrieval Recall": round(retrieval_recall, 4),
            "Retrieval Distances": json.dumps(distances),
            "Retrieved Contexts": json.dumps(docs, ensure_ascii=False),
            "Sources": sources,
            "Answer": answer
        })
        completed.add((k, q))
        save_results_checkpoint()

        print("\nQ:", q)
        print("Latency:", latency)
        print("Memory delta MB:", round(memory_after_mb - memory_before_mb, 3) if memory_before_mb is not None and memory_after_mb is not None else "psutil not installed")
        print("Expected source found:", expected_found)
        print("Sources:", sources)
        print("Answer:", answer[:150], "...")

        if GROQ_CALL_DELAY_SECONDS > 0:
            time.sleep(GROQ_CALL_DELAY_SECONDS)

# ============================================================
# SAVE CSV
# ============================================================

df = pd.DataFrame(results)
df = save_results_checkpoint()

print(f"\nSaved {csv_path}")

# ============================================================
# LATENCY GRAPH
# ============================================================

avg = df.groupby("K")["Latency"].mean()

plt.figure(figsize=(8,5))
plt.plot(avg.index, avg.values, marker="o")
plt.title("Average Latency vs Top-K")
plt.xlabel("Top-K")
plt.ylabel("Seconds")
plt.grid(True)
latency_plot_path = os.path.join(RESULTS_FOLDER, "latency_vs_top_k.png")
plt.savefig(latency_plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {latency_plot_path}")

# ============================================================
# SOURCE FILE USAGE GRAPH
# ============================================================

source_counts = {}

for s in df["Sources"]:

    for x in s.split(","):

        x = x.strip()

        source_counts[x] = source_counts.get(x, 0) + 1

plt.figure(figsize=(10,5))
plt.bar(source_counts.keys(), source_counts.values())
plt.xticks(rotation=90)
plt.title("Retrieved Source Frequency")
plt.tight_layout()
source_plot_path = os.path.join(RESULTS_FOLDER, "source_frequency.png")
plt.savefig(source_plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {source_plot_path}")

# ============================================================
# TOP-K ANALYSIS AND FINAL REPORT
# ============================================================

summary, best = analyze_results(csv_path)

print("\nTop-K summary:")
print(summary.to_string(index=False))
print(f"\nRecommended Top-K: K={int(best['K'])}")
print("Saved results/topk_summary.csv")
print("Saved results/quality_vs_top_k.png")
print("Saved results/quality_latency_tradeoff.png")
print("Saved results/rapport_experimentation_topk.txt")
