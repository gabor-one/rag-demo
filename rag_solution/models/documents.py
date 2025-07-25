from pydantic import BaseModel, Field
from typing import List, Optional


class DocumentIngest(BaseModel):
    text: str = Field(..., description="Raw content of the document")
    metadata: Optional[dict[str, str]] = Field(
        {}, description="Optional metadata for the document"
    )


class DocumentsIngestRequest(BaseModel):
    documents: list[DocumentIngest] = Field(
        ..., description="List of documents to ingest"
    )


class QueryRequest(BaseModel):
    query: str = Field(..., description="Query string for semantic search")
    limit: Optional[int] = Field(5, description="Number of top results to return")
    similarity_threshold: Optional[float] = Field(
        None, description="Similarity threshold for results", examples=[None]
    )
    # Putting this on API is generally not a good approach, we are exposing ourself to vulnerabilities in Milvus
    # In case of production grade system filtering would be more structured. 
    # e.g.: Expecting an array of (filter key, operator: enum, value) tuples.
    filter_phrase: Optional[str] = Field(
        None,
        description="Filter expression. Use it with 'metadata' field. More on it: https://milvus.io/docs/boolean.md",
        examples=[None, 'metadata["group"] == "value"'],
    )


class Document(DocumentIngest):
    id: int = Field(..., description="Unique identifier for the document")


class QueryResponse(BaseModel):
    results: List[Document] = Field(
        ..., description="List of document results from the query"
    )


class DocumentListResponse(BaseModel):
    documents: List[Document]


class DeleteDocumentResponse(BaseModel):
    id: int = Field(..., description="Unique identifier for the document")
