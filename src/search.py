import os
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHROMA_PATH = os.path.join(BASE_DIR, "vector_db", "chroma")
COLLECTION_NAME = "arxiv_papers"

_model = None
_client = None
_collection = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_collection():
    global _client, _collection

    if _collection is not None:
        return _collection

    if not os.path.exists(CHROMA_PATH):
        raise RuntimeError(
            f"\nVector DB folder not found at: {CHROMA_PATH}\n"
            "Make sure vector_db/chroma is pushed to GitHub."
        )

    _client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        _collection = _client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(
            f"\nChromaDB collection '{COLLECTION_NAME}' not found.\n"
            f"Chroma path used: {CHROMA_PATH}\n"
            "Please rebuild the vector database locally and push vector_db to GitHub.\n"
            "Command: python src/embedder.py"
        ) from e

    return _collection


def search_papers(query, top_k=10, fetch_k=50):
    """
    Search ChromaDB and return the top unique papers.
    """

    if not query or not query.strip():
        return [], [], []

    collection = get_collection()

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

    docs, metas, distances = search_papers(query, top_k=10, fetch_k=50)

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