from fastapi import APIRouter, Request

from app.models.schemas import QueryRequest, QueryResponse

router = APIRouter()


@router.post("/")
async def ask(req: QueryRequest, request: Request) -> QueryResponse:
    res = request.app.state.qa_chain(req.question)
    return QueryResponse(
        answer=res["result"],
        sources=[doc.metadata.get("source", "unknown") for doc in res["source_documents"]],
    )
