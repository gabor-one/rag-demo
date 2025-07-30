from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic import field_validator


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    SLIDING_WINDOW = "sliding_window"


class DocumentIngest(BaseModel):
    """Schema for individual document to ingest."""

    text: str = Field(..., description="Raw content of the document")
    metadata: Optional[Dict[str, str | int | float]] = Field(
        default_factory=dict, description="Optional metadata for the document"
    )

    @field_validator("text")
    @classmethod
    def validate_text_not_empty(cls, v):
        """Ensure text is not empty or just whitespace."""
        if not v or not v.strip():
            raise ValueError("Text content cannot be empty")
        return v.strip()


class DocumentsIngestRequest(BaseModel):
    """Schema for batch document ingestion request."""

    documents: List[DocumentIngest] = Field(
        ..., description="List of documents to ingest"
    )

    @field_validator("documents")
    @classmethod
    def validate_documents_not_empty(cls, v):
        """Ensure at least one document is provided."""
        if not v:
            raise ValueError("At least one document must be provided")
        return v