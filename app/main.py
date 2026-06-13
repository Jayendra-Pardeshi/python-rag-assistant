from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Python RAG Assistant (Groq + MySQL)",
    description="AI-native Python Q&A system using Stack Overflow data, MySQL, and Groq.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Python RAG Assistant...")
    try:
        from app.rag import get_rag_service
        get_rag_service()
        logger.info("RAG Service initialized successfully.")
    except Exception as e:
        logger.warning(f"RAG Service initialization warning: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)