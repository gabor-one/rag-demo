"""
Comprehensive unit tests for the document ingestion pipeline.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent / "data-ingestion"))

try:
    from ingest import (
        ChunkingStrategy,
        DocumentChunker,
        DocumentIngest,
        DocumentIngestionPipeline,
        DocumentProcessor,
        DocumentsIngestRequest,
        IngestionConfig,
        APIClient,
    )
except ImportError as e:
    pytest.skip(f"Could not import ingest module: {e}", allow_module_level=True)


class TestDocumentIngest:
    """Test the DocumentIngest Pydantic model."""
    
    def test_valid_document(self):
        """Test creating a valid document."""
        doc = DocumentIngest(
            text="This is a test document.",
            metadata={"source": "test"}
        )
        assert doc.text == "This is a test document."
        assert doc.metadata == {"source": "test"}
    
    def test_empty_text_validation(self):
        """Test that empty text raises validation error."""
        with pytest.raises(ValueError, match="Text content cannot be empty"):
            DocumentIngest(text="")
    
    def test_whitespace_only_text_validation(self):
        """Test that whitespace-only text raises validation error."""
        with pytest.raises(ValueError, match="Text content cannot be empty"):
            DocumentIngest(text="   \n\t  ")
    
    def test_text_trimming(self):
        """Test that text is trimmed of whitespace."""
        doc = DocumentIngest(text="  This is a test.  \n")
        assert doc.text == "This is a test."
    
    def test_default_metadata(self):
        """Test that metadata defaults to empty dict."""
        doc = DocumentIngest(text="Test text")
        assert doc.metadata == {}


class TestDocumentsIngestRequest:
    """Test the DocumentsIngestRequest Pydantic model."""
    
    def test_valid_request(self):
        """Test creating a valid request."""
        docs = [
            DocumentIngest(text="Document 1"),
            DocumentIngest(text="Document 2")
        ]
        request = DocumentsIngestRequest(documents=docs)
        assert len(request.documents) == 2
    
    def test_empty_documents_validation(self):
        """Test that empty documents list raises validation error."""
        with pytest.raises(ValueError, match="At least one document must be provided"):
            DocumentsIngestRequest(documents=[])


class TestIngestionConfig:
    """Test the IngestionConfig settings model."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = IngestionConfig(api_endpoint="http://test.com")
        assert config.input_folder == "./documents"
        assert config.file_extensions == [".pdf", ".txt"]
        assert config.chunking_strategy == ChunkingStrategy.FIXED_SIZE
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.max_concurrency == 10
        assert config.batch_size == 5
        assert config.max_retries == 5
        assert config.retry_delay == 1.0
    
    def test_env_prefix(self):
        """Test that environment variables are read with correct prefix."""
        import os
        original_value = os.environ.get("INGEST_CHUNK_SIZE")
        
        try:
            os.environ["INGEST_CHUNK_SIZE"] = "500"
            config = IngestionConfig(api_endpoint="http://test.com")
            assert config.chunk_size == 500
        finally:
            if original_value is not None:
                os.environ["INGEST_CHUNK_SIZE"] = original_value
            else:
                os.environ.pop("INGEST_CHUNK_SIZE", None)


class TestDocumentChunker:
    """Test the DocumentChunker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        self.metadata = {"source": "test_file.txt"}
    
    def test_fixed_size_chunking(self):
        """Test fixed-size chunking strategy."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=5,  # 5 words per chunk
            overlap=2      # 2 words overlap
        )
        
        chunks = chunker.chunk_text(self.test_text, self.metadata)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentIngest) for chunk in chunks)
        assert all("chunking_strategy" in chunk.metadata for chunk in chunks)
        assert all(chunk.metadata["chunking_strategy"] == "fixed_size" for chunk in chunks)
    
    def test_semantic_chunking(self):
        """Test semantic chunking strategy."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=10,  # 10 words per chunk
            overlap=3       # 3 words overlap
        )
        
        chunks = chunker.chunk_text(self.test_text, self.metadata)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentIngest) for chunk in chunks)
        assert all("chunking_strategy" in chunk.metadata for chunk in chunks)
        assert all(chunk.metadata["chunking_strategy"] == "semantic" for chunk in chunks)
    
    def test_sliding_window_chunking(self):
        """Test sliding window chunking strategy."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.SLIDING_WINDOW,
            chunk_size=8,   # 8 words per chunk
            overlap=3       # 3 words overlap
        )
        
        chunks = chunker.chunk_text(self.test_text, self.metadata)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentIngest) for chunk in chunks)
        assert all("chunking_strategy" in chunk.metadata for chunk in chunks)
        assert all(chunk.metadata["chunking_strategy"] == "sliding_window" for chunk in chunks)
    
    def test_unknown_strategy_raises_error(self):
        """Test that unknown chunking strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            chunker = DocumentChunker(strategy="unknown_strategy")
            chunker.chunk_text(self.test_text, self.metadata)
    
    def test_metadata_propagation(self):
        """Test that metadata is properly propagated to chunks."""
        chunker = DocumentChunker(ChunkingStrategy.FIXED_SIZE, chunk_size=5)
        metadata = {"file": "test.txt", "category": "test"}
        
        chunks = chunker.chunk_text(self.test_text, metadata)
        
        for chunk in chunks:
            assert "file" in chunk.metadata
            assert "category" in chunk.metadata
            assert chunk.metadata["file"] == "test.txt"
            assert chunk.metadata["category"] == "test"


class TestDocumentProcessor:
    """Test the DocumentProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = DocumentProcessor()
    
    def test_process_text_file(self):
        """Test processing a text file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document content.")
            f.flush()
            
            content = self.processor.process_file(Path(f.name))
            
            assert content == "This is a test document content."
        
        # Clean up
        Path(f.name).unlink()
    
    def test_process_nonexistent_file(self):
        """Test processing a file that doesn't exist."""
        content = self.processor.process_file(Path("nonexistent_file.txt"))
        assert content is None
    
    def test_process_empty_file(self):
        """Test processing an empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("")
            f.flush()
            
            content = self.processor.process_file(Path(f.name))
            
            assert content is None
        
        # Clean up
        Path(f.name).unlink()
    
    @patch('ingest.DocumentConverter')
    def test_process_pdf_file(self, mock_converter_class):
        """Test processing a PDF file using docling."""
        # Mock the converter and its result
        mock_converter = MagicMock()
        mock_result = MagicMock()
        mock_document = MagicMock()
        
        mock_document.export_to_markdown.return_value = "PDF content as markdown"
        mock_result.document = mock_document
        mock_converter.convert.return_value = mock_result
        mock_converter_class.return_value = mock_converter
        
        # Create a temporary PDF file (just need the path)
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            pass
        
        try:
            processor = DocumentProcessor()
            content = processor.process_file(Path(f.name))
            
            assert content == "PDF content as markdown"
            mock_converter.convert.assert_called_once_with(f.name)
        finally:
            Path(f.name).unlink()


class TestAPIClient:
    """Test the APIClient class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = IngestionConfig(
            api_endpoint="http://test.com/api",
            api_timeout=30,
            max_retries=3,
            retry_delay=0.1  # Short delay for tests
        )
    
    @pytest.mark.asyncio
    async def test_successful_send(self):
        """Test successful document sending."""
        documents = [DocumentIngest(text="Test document")]
        
        # Mock successful HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.headers = {}
        
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_response
        
        async with APIClient(self.config) as client:
            client.session = mock_session
            result = await client.send_documents(documents)
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_rate_limit_retry(self):
        """Test rate limit handling with retry."""
        documents = [DocumentIngest(text="Test document")]
        
        # Mock rate limit response followed by success
        mock_rate_limit_response = AsyncMock()
        mock_rate_limit_response.status = 429
        mock_rate_limit_response.headers = {"Retry-After": "1"}
        
        mock_success_response = AsyncMock()
        mock_success_response.status = 200
        mock_success_response.headers = {}
        
        mock_session = AsyncMock()
        # First call returns rate limit, second call succeeds
        mock_session.post.return_value.__aenter__.side_effect = [
            mock_rate_limit_response,
            mock_success_response
        ]
        
        async with APIClient(self.config) as client:
            client.session = mock_session
            result = await client.send_documents(documents)
            
            assert result is True
            assert mock_session.post.call_count == 2
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test max retries exceeded scenario."""
        documents = [DocumentIngest(text="Test document")]
        
        # Mock error responses
        mock_error_response = AsyncMock()
        mock_error_response.status = 500
        mock_error_response.text.return_value = "Internal Server Error"
        
        mock_session = AsyncMock()
        mock_session.post.return_value.__aenter__.return_value = mock_error_response
        
        async with APIClient(self.config) as client:
            client.session = mock_session
            result = await client.send_documents(documents)
            
            assert result is False
            # Should retry max_retries + 1 times (initial + retries)
            assert mock_session.post.call_count == self.config.max_retries + 1


class TestDocumentIngestionPipeline:
    """Test the DocumentIngestionPipeline class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = IngestionConfig(
            api_endpoint="http://test.com/api",
            input_folder="./test_documents",
            chunk_size=10,
            batch_size=2,
            max_concurrency=2
        )
        self.pipeline = DocumentIngestionPipeline(self.config)
    
    def test_discover_files_nonexistent_folder(self):
        """Test file discovery with nonexistent folder."""
        with pytest.raises(FileNotFoundError, match="Input folder does not exist"):
            self.pipeline.discover_files()
    
    def test_discover_files_existing_folder(self):
        """Test file discovery with existing folder."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_dir = Path(temp_dir)
            (test_dir / "test1.txt").write_text("Content 1")
            (test_dir / "test2.pdf").write_text("Content 2")  # Mock PDF
            (test_dir / "test3.doc").write_text("Content 3")  # Should be ignored
            
            # Update config to use temp directory
            self.config.input_folder = str(test_dir)
            pipeline = DocumentIngestionPipeline(self.config)
            
            files = pipeline.discover_files()
            
            # Should find .txt and .pdf files only
            assert len(files) == 2
            assert any(f.name == "test1.txt" for f in files)
            assert any(f.name == "test2.pdf" for f in files)
            assert not any(f.name == "test3.doc" for f in files)
    
    @patch('ingest.DocumentProcessor.process_file')
    def test_process_files(self, mock_process_file):
        """Test file processing."""
        # Mock file processing to return test content
        mock_process_file.return_value = "This is test content for chunking."
        
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("Content")
            
            chunks = self.pipeline.process_files([test_file])
            
            assert len(chunks) > 0
            assert all(isinstance(chunk, DocumentIngest) for chunk in chunks)
            mock_process_file.assert_called_once_with(test_file)
    
    @pytest.mark.asyncio
    @patch('ingest.APIClient')
    async def test_send_chunks(self, mock_api_client_class):
        """Test chunk sending."""
        # Create test chunks
        chunks = [
            DocumentIngest(text=f"Chunk {i}", metadata={"index": str(i)})
            for i in range(5)
        ]
        
        # Mock API client
        mock_client = AsyncMock()
        mock_client.send_documents.return_value = True
        mock_api_client_class.return_value.__aenter__.return_value = mock_client
        
        await self.pipeline.send_chunks(chunks)
        
        # Should create batches of size 2 (config.batch_size)
        # 5 chunks = 3 batches (2, 2, 1)
        assert mock_client.send_documents.call_count == 3
    
    @pytest.mark.asyncio
    async def test_send_empty_chunks(self):
        """Test sending empty chunks list."""
        # Should handle empty list gracefully
        await self.pipeline.send_chunks([])
        # No exception should be raised


def test_chunking_strategies_comparison():
    """Test comparing different chunking strategies on the same text."""
    text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four. This is sentence five."
    metadata = {"source": "comparison_test"}
    
    strategies = [
        ChunkingStrategy.FIXED_SIZE,
        ChunkingStrategy.SEMANTIC,
        ChunkingStrategy.SLIDING_WINDOW
    ]
    
    results = {}
    
    for strategy in strategies:
        chunker = DocumentChunker(strategy=strategy, chunk_size=8, overlap=2)
        chunks = chunker.chunk_text(text, metadata)
        results[strategy] = chunks
        
        # Basic assertions for all strategies
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentIngest) for chunk in chunks)
        assert all(chunk.metadata["chunking_strategy"] == strategy.value for chunk in chunks)
    
    # Compare results
    for strategy, chunks in results.items():
        print(f"\n{strategy.value}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i}: {len(chunk.text.split())} words")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
