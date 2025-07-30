from typing import List

from model import ChunkingStrategy
from pydantic import Field
from pydantic_settings import BaseSettings


class IngestionConfig(BaseSettings):
    """Configuration for the ingestion process."""

    # Input configuration
    input_folder: str = Field(
        default="./documents", description="Folder containing documents to ingest"
    )
    file_extensions: List[str] = Field(
        default=[".pdf", ".txt"], description="File extensions to process"
    )

    # Chunking configuration
    chunking_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.FIXED_SIZE, description="Chunking strategy to use"
    )

    chunk_size: int | None = Field(
        default=None,
        description="Size of chunks in tokens for fixed-size strategy. If None, uses embedding model's max sequence length.",
    )
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")

    # API configuration
    api_endpoint: str = Field(
        default="http://localhost:8000/ingest",
        description="API endpoint to send documents to",
    )
    api_timeout: int = Field(default=30, description="API request timeout in seconds")

    # Concurrency configuration
    #   Rate limiter set to 4 by default.
    #   This is set to 5 to demonstrate rate limiting.
    max_concurrency: int = Field(
        default=5, description="Maximum concurrent API requests"
    )
    batch_size: int = Field(
        default=50, description="Number of documents per API request"
    )

    # Retry configuration
    max_retries: int = Field(
        default=5, description="Maximum retries for non-429 errors"
    )
    retry_delay: float = Field(
        default=1.0, description="Base delay between retries in seconds"
    )

    class Config:
        env_prefix = "INGEST_"
