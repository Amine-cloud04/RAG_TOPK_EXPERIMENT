import os
import requests
from bs4 import BeautifulSoup
import wikipedia
from urllib.parse import urlparse

# 1. Setup the data directory
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# 2. List of non-Wikipedia URLs (AI Articles)
article_urls = [
    "https://squirro.com/squirro-blog/state-of-rag-genai",
    "https://cloud.google.com/use-cases/retrieval-augmented-generation",
    "https://www.telerik.com/blogs/building-rag-aspnet-core",
    "https://www.techment.com/blogs/rag-architectures-enterprise-use-cases-2026/",
    "https://www.firecrawl.dev/blog/best-open-source-rag-frameworks"
]

# 3. List of Wikipedia Topics
wiki_topics = [
    "Retrieval-augmented generation",
    "Large language model",
    "Prompt engineering",
    "Attention (machine learning)",
    "Knowledge graph",
    "Vector database",
    "Transformer (deep learning architecture)",
    "Natural language processing",
    "Artificial intelligence",
    "Deep learning"
]

def save_to_file(filename, content):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Saved: {filename}")

def article_filename(url):
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    slug = path.split("/")[-1] if path else parsed.netloc
    return f"article_{slug}.txt"

# --- Step 1: Scrape Wikipedia ---
print("Fetching Wikipedia pages...")
for topic in wiki_topics:
    try:
        page = wikipedia.page(topic, auto_suggest=False)
        filename = f"wiki_{topic.replace(' ', '_').lower()}.txt"
        save_to_file(filename, page.content)
    except Exception as e:
        print(f"Error fetching wiki '{topic}': {e}")

# --- Step 2: Scrape AI Articles ---
print("\nFetching AI articles...")
headers = {"User-Agent": "Mozilla/5.0"}

for url in article_urls:
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract title and clean text (removing scripts/styles)
        title = soup.title.string if soup.title else url.split("/")[-1]
        for script in soup(["script", "style"]):
            script.extract()
        
        text = soup.get_text(separator='\n')
        # Basic cleanup: remove extra whitespace
        lines = (line.strip() for line in text.splitlines())
        clean_text = '\n'.join(chunk for chunk in lines if chunk)

        filename = article_filename(url)
        save_to_file(filename, f"Source: {url}\n\n{clean_text}")
    except Exception as e:
        print(f"Error scraping {url}: {e}")

print("\nDone! Check your /data folder for 15 documents.")
