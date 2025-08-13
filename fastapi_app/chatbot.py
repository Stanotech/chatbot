import os
import requests
from typing import Any, List, Optional

from langchain.llms.base import LLM
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from pydantic import Field


# --- Embedding wrapper for SentenceTransformer ---
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()


# --- LLM for Ollama ---
class OllamaLLM(LLM):
    ollama_url: str = Field(...)

    def _call(self, prompt: str, stop=None) -> str:
        print("in call 1")
        resp = requests.post(f"{self.ollama_url}/api/generate", json={"model": "mistral-finance-ft", "prompt": prompt, "stream": False})
        print("in call 2")
        print("in call 3")
        print(resp.text)
        print("after resp")
        return resp.json().get("response", "")

    @property
    def _identifying_params(self):
        return {"ollama_url": self.ollama_url}

    @property
    def _llm_type(self):
        return "ollama"

  
# --- Creating QA chain---
def create_qa_chain(qdrant_url, ollama_url):
    print("2")
    llm = OllamaLLM(ollama_url=ollama_url)
    print("3")
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(texts):
        return emb_model.encode(texts).tolist()

    from urllib.parse import urlparse
    parsed_url = urlparse(qdrant_url)
    host = parsed_url.hostname or "localhost"
    port = parsed_url.port or 6333

    client = QdrantClient(host=host, port=port)

    vectordb = Qdrant(client=client, collection_name="finance_knowledge", embeddings=embed)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    print("4")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)


if __name__ == "__main__":
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

    qa_chain = create_qa_chain(QDRANT_URL, OLLAMA_URL)
    query = "Opowiedz mi o giełdzie w Polsce"
    result = qa_chain.run(query)
    print("Odpowiedź:", result)
