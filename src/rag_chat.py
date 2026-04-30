import os
from dotenv import load_dotenv
from groq import Groq
try:
    from src.search import search_papers
except ModuleNotFoundError:
    from search import search_papers

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

groq_client = Groq(api_key=GROQ_API_KEY)

MODEL = "llama-3.3-70b-versatile"


def distance_to_relevance(distance):
    if distance is None:
        return 0.0

    score = 1.0 - (distance / 2.0)
    return round(max(0.0, min(1.0, score)), 3)


def filter_context(documents, metadatas):
    clean_items = []

    for doc, meta in zip(documents, metadatas):
        if not doc or len(doc.strip()) < 50:
            continue

        clean_items.append({
            "text": doc.strip(),
            "title": meta.get("title", "Untitled Paper"),
            "pdf_url": meta.get("pdf_url", "")
        })

    return clean_items


def calculate_confidence_score(distances):
    if not distances:
        return 0.0

    avg_distance = sum(distances) / len(distances)
    confidence = 1.0 - (avg_distance / 2.0)

    return round(max(0.0, min(1.0, confidence)), 3)


def build_sources_with_scores(metadatas, distances):
    source_map = {}

    for i, meta in enumerate(metadatas):
        distance = distances[i] if i < len(distances) else 1.0

        title = meta.get("title", "Untitled Paper")
        pdf_url = meta.get("pdf_url", "")
        published = meta.get("published", "")
        relevance_score = distance_to_relevance(distance)

        if title not in source_map:
            source_map[title] = {
                "title": title,
                "pdf_url": pdf_url,
                "published": published,
                "relevance_score": relevance_score
            }
        else:
            if relevance_score > source_map[title]["relevance_score"]:
                source_map[title]["relevance_score"] = relevance_score

    sources = list(source_map.values())
    sources.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

    return sources


def ask_question(query):
    documents, metadatas, distances = search_papers(query)

    context_items = filter_context(documents, metadatas)

    if not context_items:
        context_text = "No relevant context retrieved."
    else:
        context_text = "\n\n---\n\n".join(
            [item["text"] for item in context_items[:6]]
        )

    score = calculate_confidence_score(distances)
    sources = build_sources_with_scores(metadatas, distances)

    prompt = f"""
You are an expert AI research assistant.

Answer the question using the provided context as the primary source.

Guidelines:
- Use the retrieved context to answer the question.
- If the context is incomplete, you may use general knowledge to provide a complete explanation.
- Ensure the answer is accurate and relevant to the question.
- Do NOT hallucinate incorrect facts.
- Keep the answer clear, structured, and easy to understand.
- Do NOT mention the context explicitly.

Context:
{context_text}

Question:
{query}

Answer format:

1. Definition
2. Explanation
3. Key Points
4. Example
"""

    try:
        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.05,
            max_tokens=1024
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"

    return answer, sources, score, context_items


def generate_related_questions(query, answer):
    prompt = f"""
You are an AI research assistant.

Based on the user's question and the generated answer, suggest exactly 3 useful follow-up research questions.

User Question:
{query}

Answer:
{answer}

Rules:
- Return only the 3 questions.
- Do not include numbering explanations.
- Keep each question short and clear.
"""

    try:
        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )

        text = response.choices[0].message.content.strip()

        questions = []
        for line in text.split("\n"):
            line = line.strip()
            line = line.lstrip("1234567890.-) ").strip()

            if line:
                questions.append(line)

        return questions[:3]

    except Exception:
        return []


def summarize_paper(title, pdf_url=None):
    prompt = f"""
You are an AI research assistant.

Summarize this research paper in simple terms.

Paper Title:
{title}

PDF URL:
{pdf_url if pdf_url else "Not provided"}

Provide the summary in this format:

1. Problem
2. Approach
3. Key Contribution
4. Why It Matters

Keep it clear, concise, and beginner-friendly.
"""

    try:
        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error summarizing paper: {str(e)}"


def compare_papers(paper1, paper2):
    title1 = paper1.get("title", "Paper 1")
    url1 = paper1.get("pdf_url", "")

    title2 = paper2.get("title", "Paper 2")
    url2 = paper2.get("pdf_url", "")

    prompt = f"""
You are an AI research assistant.

Compare the following two research papers in simple terms.

Paper 1:
Title: {title1}
PDF URL: {url1}

Paper 2:
Title: {title2}
PDF URL: {url2}

Provide comparison in this format:

1. Main Topic of Each Paper
2. Key Differences
3. Similarities
4. Strengths of Paper 1
5. Strengths of Paper 2
6. When Each Paper Is Useful

Keep it clear and beginner-friendly.
"""

    try:
        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1024
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error comparing papers: {str(e)}"


if __name__ == "__main__":
    query = input("Ask a research question: ")

    answer, sources, score, context_docs = ask_question(query)

    print("\nAnswer:\n")
    print(answer)

    print("\nConfidence:")
    print(score)

    print("\nSources:")
    for i, meta in enumerate(sources, 1):
        print(f"{i}. {meta['title']}")
        print(f"   Relevance: {meta.get('relevance_score', 0)}")
        print(f"   {meta['pdf_url']}")

    related = generate_related_questions(query, answer)
    print("\nRelated Questions:")
    for q in related:
        print("-", q)