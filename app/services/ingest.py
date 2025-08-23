from logging import getLogger

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Qdrant

from core.config import settings

logger = getLogger(__name__)


def load_documents() -> list:
    docs = []
    if not settings.DATA_DIR.exists():
        raise Exception(f"📂 Data directory not found: {settings.DATA_DIR}")

    file_paths = settings.DATA_DIR.glob("*")

    if not file_paths:
        raise Exception(f"📂 Data directory is empty: {settings.DATA_DIR}")
    for file_path in file_paths:
        if file_path.suffix == ".txt":
            loader = TextLoader(str(file_path))
        elif file_path.suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        else:
            logger.warning(f"⚠ Pomijam format: {file_path.name}")
            continue
        docs.extend(loader.load())
    return docs


def main() -> None:
    logger.info("📂 Loading documents...")
    documents = load_documents()

    logger.info(f"📄 {len(documents)} dokuments found. Chunking...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(documents)

    logger.info("🔢 Embedding...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    logger.info(
        f"💾 Saving into Qdrant ({settings.QDRANT_URL}), collection: {settings.QDRANT_COLLECTION}"
    )
    Qdrant.from_documents(
        split_docs,
        embeddings,
        url=settings.QDRANT_URL,
        collection_name=settings.QDRANT_COLLECTION,
    )

    logger.info("✅ Ingest finished!")


if __name__ == "__main__":
    main()
