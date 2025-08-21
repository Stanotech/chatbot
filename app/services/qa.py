from urllib.parse import urlparse

from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient

from services.ollama_llm import OllamaLLM


def init_qa_chain(qdrant_url: str, ollama_url: str) -> RetrievalQA:
    llm = OllamaLLM(ollama_url=ollama_url)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    parsed_url = urlparse(qdrant_url)
    host = parsed_url.hostname or "localhost"
    port = parsed_url.port or 6333

    client = QdrantClient(host=host, port=port)

    vectordb = Qdrant(client=client, collection_name="finance_docs", embeddings=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
