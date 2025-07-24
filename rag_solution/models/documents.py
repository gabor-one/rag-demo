from pydantic import BaseModel, Field
from typing import List, Optional


class DocumentIngest(BaseModel):
    text: str = Field(..., description="Raw content of the document")
    metadata: Optional[dict[str, str]] = Field(None, description="Optional metadata for the document")

class DocumentsIngestRequest(BaseModel):
    documents: list[DocumentIngest] = Field(..., description="List of documents to ingest")


class QueryRequest(BaseModel):
    query: str = Field(..., description="Query string for semantic search")
    top_k: Optional[int] = Field(5, description="Number of top results to return")
    similarity_threshold: Optional[float] = Field(0.75, description="Similarity threshold for results")

class Document(BaseModel):
    id: int = Field(..., description="Unique identifier for the document")
    text: str = Field(..., description="Text content of the document")

class QueryResponse(BaseModel):
    results: List[Document] = Field(..., description="List of document results from the query")

class DocumentListResponse(BaseModel):
    documents: List[Document]

class DeleteDocumentResponse(BaseModel):
    id: int = Field(..., description="Unique identifier for the document")

