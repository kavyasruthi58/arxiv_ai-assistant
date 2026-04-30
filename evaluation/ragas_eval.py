import os
import sys
from dotenv import load_dotenv
from datasets import Dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_PATH)

from rag_chat import ask_question

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# Groq supports only n=1, so force RAGAS answer relevancy to 1 generation
try:
    answer_relevancy.strictness = 1
except Exception:
    pass

# Lighter evaluator model to reduce Groq rate-limit issues
evaluator_llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    temperature=0,
    max_tokens=2048,   # 🔥 increased
    n=1,
)

evaluator_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Better detailed ground-truth answers for RAGAS
TEST_DATA = [
    {
        "question": "What are transformers?",
        "ground_truth": (
            "Transformers are deep learning architectures designed for sequence modeling. "
            "They use self-attention mechanisms to process input tokens in parallel and "
            "capture relationships between tokens. Unlike RNNs, transformers do not process "
            "tokens step by step, which makes them faster and more scalable for NLP tasks "
            "such as translation, summarization, question answering, and text generation."
        ),
    },
    {
        "question": "How does attention mechanism work?",
        "ground_truth": (
            "The attention mechanism helps a model decide which parts of the input are most "
            "important for producing an output. It compares tokens or features, assigns "
            "attention weights based on relevance, and combines information using those "
            "weights. This allows the model to capture context, relationships, and long-range "
            "dependencies more effectively."
        ),
    },
    {
        "question": "What are hallucinations in LLMs?",
        "ground_truth": (
            "Hallucinations in large language models are generated statements that sound "
            "plausible but are incorrect, fabricated, or unsupported by the available evidence. "
            "They can happen when a model relies on learned patterns instead of verified "
            "context. RAG can help reduce hallucinations by grounding answers in retrieved "
            "documents or source material."
        ),
    },
    {
        "question": "How does RAG improve LLM responses?",
        "ground_truth": (
            "Retrieval-Augmented Generation improves LLM responses by retrieving relevant "
            "external information before generating an answer. The retrieved context gives "
            "the model factual support, improves grounding, reduces hallucinations, and helps "
            "produce answers that are more accurate and source-supported."
        ),
    },
    {
        "question": "Compare transformers and RNNs.",
        "ground_truth": (
            "Transformers and RNNs are both used for sequence modeling, but they work differently. "
            "RNNs process tokens sequentially and maintain hidden states, which can make them "
            "slower and weaker for long-range dependencies. Transformers use self-attention "
            "and process tokens in parallel, making them faster, more scalable, and better "
            "suited for modern NLP and large language model tasks."
        ),
    },
]


def build_eval_dataset():
    records = []

    for item in TEST_DATA:
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\nRunning RAG for: {question}")

        answer, sources, score, context_docs = ask_question(question)

        contexts = []
        for c in context_docs[:3]:
            text = c.get("text", "")
            if text and text.strip():
                contexts.append(text.strip())

        if not contexts:
            contexts = ["No retrieved context available."]

        records.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

    return Dataset.from_list(records)


def run_ragas_evaluation():
    dataset = build_eval_dataset()

    print("\nStarting RAGAS evaluation...")

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False,
        batch_size=1,
    )

    print("\nRAGAS Results:")
    print(result)

    df = result.to_pandas()

    output_path = os.path.join(PROJECT_ROOT, "evaluation", "ragas_results.csv")
    df.to_csv(output_path, index=False)

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    run_ragas_evaluation()