from fastapi import FastAPI

from app.api.routes_ask import router as ask_router
from app.api.routes_health import router as health_router
from app.core.config import settings
from app.core.db import init_qdrant
from app.services.qa import init_qa_chain

app = FastAPI(title="Finance QA Chatbot")

init_qdrant()

qa_chain = init_qa_chain(qdrant_url=settings.QDRANT_URL, ollama_url=settings.OLLAMA_URL)

app.state.qa_chain = qa_chain

app.include_router(ask_router, prefix="/ask", tags=["chatbot"])
app.include_router(health_router, prefix="/system", tags=["Health"])
