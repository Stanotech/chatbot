import os
from pathlib import Path


class Settings:
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "finance_docs")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "mistral-7b-finance")
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "../data"))


settings = Settings()
