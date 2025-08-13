import os
import requests
from typing import Any, List, Optional

from langchain.llms.base import LLM
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
from pydantic import Field


# --- LLM for Ollama ---
class OllamaLLM(LLM):
    ollama_url: str = Field(...)

    def _call(self, prompt: str, stop=None) -> str:
        resp = requests.post(f"{self.ollama_url}/api/generate", json={"model": "mistral-finance-ft", "prompt": prompt, "stream": False})
        return resp.json().get("response", "")

    @property
    def _identifying_params(self):
        return {"ollama_url": self.ollama_url}

    @property
    def _llm_type(self):
        return "ollama"

  
# --- Creating QA chain---
def create_qa_chain(qdrant_url, ollama_url):
    llm = OllamaLLM(ollama_url=ollama_url)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    from urllib.parse import urlparse
    parsed_url = urlparse(qdrant_url) 
    host = parsed_url.hostname or "localhost"
    port = parsed_url.port or 6333

    client = QdrantClient(host=host, port=port)

    vectordb = Qdrant(client=client, collection_name="finance_knowledge", embeddings=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
