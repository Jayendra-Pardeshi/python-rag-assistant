import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    # Groq API
    groq_api_key: str = os.getenv("GROQ_API_KEY")
    
    # MySQL Database
    mysql_host: str = os.getenv("MYSQL_HOST", "localhost")
    mysql_port: int = int(os.getenv("MYSQL_PORT", "3306"))
    mysql_user: str = os.getenv("MYSQL_USER", "root")
    mysql_password: str = os.getenv("MYSQL_PASSWORD", "")
    mysql_db_name: str = os.getenv("MYSQL_DB_NAME", "python_rag_db")

    @property
    def database_url(self):
        return f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}@{self.mysql_host}:{self.mysql_port}/{self.mysql_db_name}"
    
    # Models
    llm_model: str = os.getenv("LLM_MODEL", "llama3-8b-8192")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    
    # Paths
    data_dir: str = os.getenv("DATA_DIR", "./data")
    vectorstore_dir: str = os.getenv("VECTORSTORE_DIR", "./vectorstore/faiss_index")
    
    # Ingestion Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    sample_size: int = 5000

    class Config:
        env_file = ".env"

settings = Settings()