from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import settings
import os

class RetrieverService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
        self.vectorstore = None
        self._load_vectorstore()

    def _load_vectorstore(self):
        if not os.path.exists(settings.vectorstore_dir):
            raise FileNotFoundError(
                f"Vector store not found at {settings.vectorstore_dir}. "
                "Please run the ingestion pipeline first."
            )
        print(f"Loading FAISS index from {settings.vectorstore_dir}...")
        self.vectorstore = FAISS.load_local(
            settings.vectorstore_dir, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        print("Vector store loaded successfully.")

    def get_retriever(self, k=3):
        if not self.vectorstore:
            raise RuntimeError("Vector store is not initialized.")
        return self.vectorstore.as_retriever(search_kwargs={"k": k})