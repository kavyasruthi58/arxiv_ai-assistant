from fastapi import FastAPI
from pydantic import BaseModel
from src.rag_chat import ask_question

app = FastAPI(
    title="AI Research Assistant API",
    description="RAG API for asking questions over ArXiv AI/ML papers",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def home():
    return {
        "message": "AI Research Assistant API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/ask")
def ask(request: QuestionRequest):
    answer, sources, score, context_docs = ask_question(request.question)

    return {
        "question": request.question,
        "answer": answer,
        "confidence": score,
        "sources": sources[:5],
        "context_chunks": context_docs[:3]
    }