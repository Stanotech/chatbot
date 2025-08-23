from urllib.parse import urlparse

from app.services.ollama_llm import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


def init_qa_chain(qdrant_url: str, ollama_url: str) -> RetrievalQA:
    llm = OllamaLLM(ollama_url=ollama_url)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    parsed_url = urlparse(qdrant_url)
    host = parsed_url.hostname or "localhost"
    port = parsed_url.port or 6333

    client = QdrantClient(host=host, port=port)

    vectordb = QdrantVectorStore(
        client=client, collection_name="finance_docs", embedding=embeddings
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    cross_encoder_model = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    reranker = CrossEncoderReranker(model=cross_encoder_model)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=retriever,
    )
    return RetrievalQA.from_chain_type(
        llm=llm, retriever=compression_retriever, return_source_documents=True
    )
