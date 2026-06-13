from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str = Field(..., description="The Python programming question to ask.")

class SourceDocument(BaseModel):
    content: str
    metadata: dict

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vectorstore_loaded: bool