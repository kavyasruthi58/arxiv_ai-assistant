import json
import os
import shutil
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHUNKS_PATH = os.path.join(BASE_DIR, "data", "processed", "chunks.json")
CHROMA_PATH = os.path.join(BASE_DIR, "vector_db", "chroma")
COLLECTION_NAME = "arxiv_papers"

# Delete old vector database completely
if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

os.makedirs(CHROMA_PATH, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

for chunk in tqdm(chunks):
    embedding = model.encode(chunk["text"]).tolist()

    collection.add(
        ids=[chunk["chunk_id"]],
        documents=[chunk["text"]],
        embeddings=[embedding],
        metadatas=[{
            "title": chunk["title"],
            "pdf_url": chunk["pdf_url"],
            "published": chunk["published"]
        }]
    )

print(f"✅ Stored {len(chunks)} fresh sentence-based chunks in ChromaDB.")