from fastapi import FastAPI
from pydantic import BaseModel
from chatbot import create_qa_chain
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

import os

app = FastAPI()
QDRANT_URL = os.getenv("QDRANT_URL")
OLLAMA_URL = os.getenv("OLLAMA_URL")

# Init Qdrant collection
client = QdrantClient(url=QDRANT_URL)
if "finance_knowledge" not in [c.name for c in client.get_collections().collections]:
    client.create_collection(
        collection_name="finance_knowledge",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

qa_chain = create_qa_chain(QDRANT_URL, OLLAMA_URL)

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(req: QueryRequest):
    res = qa_chain(req.question)
    return {
        "answer": res["result"],
        "sources": [doc.metadata.get("source", "unknown") for doc in res["source_documents"]]
    }
