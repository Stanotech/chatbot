from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from core.config import settings


def init_qdrant() -> QdrantClient:
    try:
        client = QdrantClient(url=settings.QDRANT_URL)
        collections = [c.name for c in client.get_collections().collections]

        if "finance_docs" not in collections:
            client.create_collection(
                collection_name="finance_docs",
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
        return client

    except Exception as e:
        raise Exception(f"Qdrant initialization error: {e}")
