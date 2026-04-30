import os
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHROMA_PATH = os.path.join(BASE_DIR, "vector_db", "chroma")
COLLECTION_NAME = "arxiv_papers"

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


client = chromadb.PersistentClient(path=CHROMA_PATH)

try:
    collection = client.get_collection(name=COLLECTION_NAME)
except Exception:
    raise RuntimeError(
        "\nVector store not found!\n"
        "Please run embedder.py first to build the ChromaDB index.\n"
        "Command: python src/embedder.py"
    )


def search_papers(query, top_k=10, fetch_k=50):
    """
    Search ChromaDB and return the top unique papers.

    top_k:
        Final number of unique papers returned.

    fetch_k:
        Number of raw chunks retrieved before removing duplicate papers.
        Higher fetch_k improves diversity because many chunks may come
        from the same paper.
    """
    if not query or not query.strip():
        return [], [], []

    query_embedding = get_model().encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"]
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    seen_titles = set()
    unique_docs = []
    unique_metas = []
    unique_distances = []

    for doc, meta, distance in zip(docs, metas, distances):
        title = meta.get("title", "Untitled Paper")

        if title in seen_titles:
            continue

        seen_titles.add(title)
        unique_docs.append(doc)
        unique_metas.append(meta)
        unique_distances.append(distance)

        if len(unique_docs) >= top_k:
            break

    return unique_docs, unique_metas, unique_distances


if __name__ == "__main__":
    query = input("Ask a research question: ")

    docs, metas, distances = search_papers(query, top_k=10, fetch_k=30)

    if not docs:
        print("No relevant results found. Try a different query.")
    else:
        for i, (doc, meta, distance) in enumerate(zip(docs, metas, distances), start=1):
            print("\n" + "=" * 70)
            print(f"Result   : {i}")
            print(f"Distance : {round(distance, 4)}")
            print(f"Title    : {meta.get('title', 'Untitled Paper')}")
            print(f"PDF      : {meta.get('pdf_url', '')}")
            print(f"Text     : {doc[:500]}...")