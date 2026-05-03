# ============================================================
# RAG TOP-K PROJECT (USING YOUR DATA FOLDER)
# Full Version adapted to your 15 txt files
# ============================================================

# INSTALL:
# pip install chromadb sentence-transformers pandas matplotlib requests

import os
import re
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import chromadb

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
RESULTS_FOLDER = "results"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

TOP_K_VALUES = [1, 3, 5, 10]

os.makedirs(RESULTS_FOLDER, exist_ok=True)

if not GROQ_API_KEY:
    raise RuntimeError(
        "Missing GROQ_API_KEY. Set it in your terminal before running, for example:\n"
        '$env:GROQ_API_KEY="your_groq_api_key"'
    )

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

def split_text(text, chunk_size=600, overlap=100):

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

questions = [
    "What is retrieval augmented generation?",
    "What is a vector database?",
    "What is prompt engineering?",
    "How do transformers use attention?",
    "What is a large language model?",
    "What are the benefits of RAG?",
    "What are open source RAG frameworks?",
    "What is deep learning?",
    "What is artificial intelligence?",
    "How does Top K retrieval work?"
]

# ============================================================
# GROQ CALL
# ============================================================

def ask_llm(prompt, max_retries=5):

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
        "temperature": 0
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
            match = re.search(r"try again in ([\d.]+)s", str(message), re.IGNORECASE)
            wait_seconds = float(match.group(1)) + 1 if match else 5 + attempt * 2
            print(f"Rate limit reached. Retrying in {round(wait_seconds, 2)} seconds...")
            time.sleep(wait_seconds)
            continue

        raise RuntimeError(f"Groq API request failed ({r.status_code}): {message}")

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

    return docs, meta

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

results = []

for k in TOP_K_VALUES:

    print("\n==========================")
    print("Testing K =", k)
    print("==========================")

    for q in questions:

        start = time.time()

        docs, meta = retrieve(q, k)

        prompt = build_prompt(q, docs)

        answer = ask_llm(prompt)

        latency = round(time.time() - start, 2)

        sources = ", ".join([m["source"] for m in meta])

        results.append({
            "K": k,
            "Question": q,
            "Latency": latency,
            "Sources": sources,
            "Answer": answer
        })

        print("\nQ:", q)
        print("Latency:", latency)
        print("Sources:", sources)
        print("Answer:", answer[:150], "...")

# ============================================================
# SAVE CSV
# ============================================================

df = pd.DataFrame(results)

csv_path = os.path.join(RESULTS_FOLDER, "rag_results.csv")
df.to_csv(csv_path, index=False)

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
# NEXT STEP
# ============================================================

print("""
Next recommended step:

1. Open results/rag_results.csv
2. Add manual score column (1 to 5)
3. Compare quality vs K
4. Final report conclusion:
   moderate K often best tradeoff
""")
