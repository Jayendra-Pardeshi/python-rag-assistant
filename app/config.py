import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    # API
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    
    # Models
    llm_model: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Paths
    data_dir: str = os.getenv("DATA_DIR", "./data")
    vectorstore_dir: str = os.getenv("VECTORSTORE_DIR", "./vectorstore/faiss_index")
    
    # Ingestion Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    sample_size: int = 5000 # Limit data for faster assessment indexing

    class Config:
        env_file = ".env"

settings = Settings()