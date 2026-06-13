from fastapi import APIRouter, HTTPException
from app.models import QueryRequest, QueryResponse, HealthResponse
from app.rag import get_rag_service
from app.config import settings
import logging
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

router = APIRouter()

def check_db_connection():
    try:
        engine = create_engine(settings.database_url)
        conn = engine.connect()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"DB Connection failed: {e}")
        return False

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API, RAG dependencies, and Database are healthy."""
    try:
        rag = get_rag_service()
        return {
            "status": "healthy",
            "model_loaded": rag.llm is not None,
            "vectorstore_loaded": rag.retriever_service.vectorstore is not None,
            "database_connected": check_db_connection()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "vectorstore_loaded": False,
            "database_connected": False
        }

@router.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Ask a Python programming question."""
    try:
        rag = get_rag_service()
        result = rag.query(request.question)
        
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))