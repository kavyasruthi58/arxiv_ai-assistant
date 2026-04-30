import json
import os
import re
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "arxiv_papers.json")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "chunks.json")

os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[[0-9]+\]", "", text)
    text = re.sub(r"http\S+", "", text)
    return text.strip()


def split_into_sentences(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def chunk_text(text, sentences_per_chunk=10, overlap=3):
    """
    Create sentence-based chunks with overlap.

    Fix 1:
    - sentences_per_chunk = 10
    - overlap = 3

    This gives richer context and improves RAG recall.
    """
    sentences = split_into_sentences(text)
    chunks = []

    if not sentences:
        return []

    step = sentences_per_chunk - overlap

    for i in range(0, len(sentences), step):
        chunk_sentences = sentences[i:i + sentences_per_chunk]
        chunk = " ".join(chunk_sentences)

        if len(chunk.split()) > 40:
            chunks.append(chunk)

    return chunks


with open(RAW_PATH, "r", encoding="utf-8") as f:
    papers = json.load(f)

all_chunks = []

for paper in tqdm(papers):
    title = paper.get("title", "")
    summary = paper.get("summary", "")
    entry_id = paper.get("entry_id", "")
    pdf_url = paper.get("pdf_url", "")
    published = paper.get("published", "")

    full_text = f"""
    Title: {title}.

    Abstract:
    {summary}

    Detailed Explanation:
    {summary}

    Key Ideas:
    {summary}
    """

    cleaned_text = clean_text(full_text)

    chunks = chunk_text(
        cleaned_text,
        sentences_per_chunk=10,
        overlap=3
    )

    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "chunk_id": f"{entry_id}_{i}",
            "text": chunk,
            "title": title,
            "pdf_url": pdf_url,
            "published": published
        })

with open(PROCESSED_PATH, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print(f"Created {len(all_chunks)} sentence-based chunks.")