import arxiv
import json
import os
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------
# Setup
# -----------------------------
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

client = arxiv.Client()

search = arxiv.Search(
    query="cat:cs.LG OR cat:cs.AI OR cat:cs.CL",
    max_results=200,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

papers = []

# -----------------------------
# Step 1: Download papers
# -----------------------------
for paper in tqdm(client.results(search), total=200):
    papers.append({
        "title": paper.title,
        "authors": [author.name for author in paper.authors],
        "summary": paper.summary,
        "published": str(paper.published),
        "pdf_url": paper.pdf_url,
        "entry_id": paper.entry_id
    })

# Save raw data
with open("data/raw/arxiv_papers.json", "w", encoding="utf-8") as f:
    json.dump(papers, f, indent=2, ensure_ascii=False)

print(f"Saved {len(papers)} raw papers.")

# -----------------------------
# Step 2: Chunking (IMPORTANT)
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,        # 🔥 increased from 500 → 800
    chunk_overlap=100      # 🔥 increased from 50 → 100
)

chunks = []

for paper in papers:
    split_texts = text_splitter.split_text(paper["summary"])

    for chunk in split_texts:
        chunks.append({
            "text": chunk,
            "metadata": {
                "title": paper["title"],
                "pdf_url": paper["pdf_url"],
                "published": paper["published"]
            }
        })

# Save processed chunks
with open("data/processed/chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

print(f"Created {len(chunks)} chunks.")