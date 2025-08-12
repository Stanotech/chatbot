import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "finance_docs")
DATA_DIR = os.getenv("DATA_DIR", "./data")

def load_documents():
    docs = []
    for file_name in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file_name)
        if file_name.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file_name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            print(f"⚠ Skip other format: {file_name}")
            continue
        docs.extend(loader.load())
    return docs

def main():
    print("📂 Loading documents...")
    documents = load_documents()

    print(f"📄 {len(documents)} dokuments found. Chunking...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(documents)

    print("🔢 Embedding...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"💾 Saving into Qdrant ({QDRANT_URL}), collection: {COLLECTION_NAME}")
    Qdrant.from_documents(
        split_docs,
        embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )

    print("✅ Ingest finished!")

if __name__ == "__main__":
    main()
