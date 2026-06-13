import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config import settings
from ingestion.load_dataset import load_stackoverflow_data
from ingestion.preprocess import merge_and_clean

def run_ingestion():
    print("--- Starting Ingestion Pipeline ---")
    
    # 1. Load
    df_q, df_a = load_stackoverflow_data()
    
    # 2. Preprocess
    texts, metadatas = merge_and_clean(df_q, df_a, sample_size=settings.sample_size)
    
    # 3. Chunk
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )
    
    # Note: We split the combined Q&A text
    documents = text_splitter.create_documents(texts, metadatas=metadatas)
    
    # 4. Embed & Index
    print(f"Creating FAISS index with {len(documents)} chunks...")
    print("This may take a few minutes depending on CPU...")
    
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # 5. Save
    os.makedirs(settings.vectorstore_dir, exist_ok=True)
    vectorstore.save_local(settings.vectorstore_dir)
    
    print(f"Index saved successfully to {settings.vectorstore_dir}")

if __name__ == "__main__":
    run_ingestion()