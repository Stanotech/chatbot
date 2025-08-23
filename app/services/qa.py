from urllib.parse import urlparse

from langchain.chains import RetrievalQA
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.rerank.base import CrossEncoderReranker

from services.ollama_llm import OllamaLLM


def init_qa_chain(qdrant_url: str, ollama_url: str) -> RetrievalQA:
    llm = OllamaLLM(ollama_url=ollama_url)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2", batch_size = 32)

    parsed_url = urlparse(qdrant_url)
    host = parsed_url.hostname or "localhost"
    port = parsed_url.port or 6333

    client = QdrantClient(host=host, port=port)

    vectordb = Qdrant(client=client, collection_name="finance_docs", embeddings=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=retriever,
    )
    return RetrievalQA.from_chain_type(llm=llm, retriever=compression_retriever , return_source_documents=True)
