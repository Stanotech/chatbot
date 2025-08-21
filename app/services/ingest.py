from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Qdrant

from core.config import settings


def load_documents() -> list:
    docs = []
    for file_path in settings.DATA_DIR.glob("*"):
        if file_path.suffix == ".txt":
            loader = TextLoader(str(file_path))
        elif file_path.suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        else:
            print(f"⚠ Pomijam format: {file_path.name}")
            continue
        docs.extend(loader.load())
    return docs


def main() -> None:
    print("📂 Loading documents...")
    documents = load_documents()

    print(f"📄 {len(documents)} dokuments found. Chunking...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(documents)

    print("🔢 Embedding...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    print(
        f"💾 Saving into Qdrant ({settings.QDRANT_URL}), collection: {settings.QDRANT_COLLECTION}"
    )
    Qdrant.from_documents(
        split_docs,
        embeddings,
        url=settings.QDRANT_URL,
        collection_name=settings.QDRANT_COLLECTION,
    )

    print("✅ Ingest finished!")


if __name__ == "__main__":
    main()
