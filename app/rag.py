from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from app.config import settings
from app.prompts import get_qa_prompt
from app.retriever import RetrieverService

class RAGService:
    def __init__(self):
        self.retriever_service = RetrieverService()
        # Initialize Groq LLM
        self.llm = ChatGroq(
            model=settings.llm_model, 
            temperature=0,
            api_key=settings.groq_api_key
        )
        self.chain = self._create_chain()

    def _create_chain(self):
        prompt = get_qa_prompt()
        retriever = self.retriever_service.get_retriever()
        
        # Create the question-answering chain
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        
        # Create the retrieval chain
        chain = create_retrieval_chain(retriever, question_answer_chain)
        return chain

    def query(self, question: str):
        """
        Executes the RAG pipeline using Groq.
        """
        response = self.chain.invoke({"input": question})
        
        # Format sources for the API response
        sources = [
            {"content": doc.page_content, "metadata": doc.metadata}
            for doc in response["context"]
        ]
        
        return {
            "answer": response["answer"],
            "sources": sources
        }

# Global instance
rag_instance = None

def get_rag_service():
    global rag_instance
    if rag_instance is None:
        rag_instance = RAGService()
    return rag_instance