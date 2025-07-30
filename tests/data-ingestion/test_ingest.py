# This file is mostly AI generated. Very few lines were manually edited.
# Not super interesting.

"""
Unit tests for the document ingestion pipeline.

Tests cover the main DocumentIngestionPipeline class and its methods,
including file discovery, processing, chunking, and API communication.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from rag_solution.data_ingestion.ingest import DocumentIngestionPipeline
from rag_solution.data_ingestion.ingestion_config import IngestionConfig
from rag_solution.data_ingestion.model import DocumentIngest, ChunkingStrategy


@pytest.fixture
def config():
    """Create a test configuration."""
    return IngestionConfig(
        input_folder="./test_documents",
        file_extensions=[".pdf", ".txt"],
        chunking_strategy=ChunkingStrategy.FIXED_SIZE,
        chunk_size=500,
        chunk_overlap=50,
        api_endpoint="http://test.example.com/ingest",
        batch_size=10,
        max_concurrency=2,
        api_timeout=30,
        max_retries=3,
        retry_delay=1
    )


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        DocumentIngest(
            text="This is the first test document content.",
            metadata={"source_file": "test1.pdf", "page": 1}
        ),
        DocumentIngest(
            text="This is the second test document content.",
            metadata={"source_file": "test2.txt", "page": 1}
        ),
    ]


@pytest.fixture
def sample_chunks():
    """Create sample chunks for testing."""
    return [
        DocumentIngest(
            text="Chunk 1 content",
            metadata={"source_file": "test1.pdf", "chunk_id": 1}
        ),
        DocumentIngest(
            text="Chunk 2 content",
            metadata={"source_file": "test1.pdf", "chunk_id": 2}
        ),
        DocumentIngest(
            text="Chunk 3 content",
            metadata={"source_file": "test2.txt", "chunk_id": 1}
        ),
    ]


class TestDocumentIngestionPipeline:
    """Test cases for DocumentIngestionPipeline class."""

    def test_init(self, config):
        """Test pipeline initialization."""
        pipeline = DocumentIngestionPipeline(config)
        
        assert pipeline.config == config
        assert pipeline.processor is not None
        assert pipeline.chunker is not None
        assert pipeline.chunker.strategy == config.chunking_strategy
        assert pipeline.chunker.chunk_size == config.chunk_size
        assert pipeline.chunker.overlap == config.chunk_overlap

    @patch('rag_solution.data_ingestion.ingest.Path')
    def test_discover_files_success(self, mock_path, config):
        """Test successful file discovery."""
        # Setup mock path
        mock_input_path = MagicMock()
        mock_input_path.exists.return_value = True
        mock_path.return_value = mock_input_path
        
        # Setup mock files
        mock_pdf_file = MagicMock()
        mock_pdf_file.suffix.lower.return_value = ".pdf"
        mock_txt_file = MagicMock()
        mock_txt_file.suffix.lower.return_value = ".txt"
        mock_doc_file = MagicMock()
        mock_doc_file.suffix.lower.return_value = ".doc"  # Should be filtered out
        
        # Mock rglob to return different files for different extensions
        def rglob_side_effect(pattern):
            if pattern == "*.pdf":
                return [mock_pdf_file]
            elif pattern == "*.txt":
                return [mock_txt_file, mock_doc_file]  # doc file should be filtered
            return []
        
        mock_input_path.rglob.side_effect = rglob_side_effect
        
        pipeline = DocumentIngestionPipeline(config)
        files = pipeline.discover_files()
        
        # Should only return PDF and TXT files
        assert len(files) == 2
        assert mock_pdf_file in files
        assert mock_txt_file in files
        assert mock_doc_file not in files

    @patch('rag_solution.data_ingestion.ingest.Path')
    def test_discover_files_folder_not_exists(self, mock_path, config):
        """Test file discovery when input folder doesn't exist."""
        mock_input_path = MagicMock()
        mock_input_path.exists.return_value = False
        mock_path.return_value = mock_input_path
        
        pipeline = DocumentIngestionPipeline(config)
        
        with pytest.raises(FileNotFoundError) as exc_info:
            pipeline.discover_files()
        
        assert "Input folder does not exist" in str(exc_info.value)

    @patch('rag_solution.data_ingestion.ingest.Path')
    def test_discover_files_no_files(self, mock_path, config):
        """Test file discovery when no matching files found."""
        mock_input_path = MagicMock()
        mock_input_path.exists.return_value = True
        mock_input_path.rglob.return_value = []
        mock_path.return_value = mock_input_path
        
        pipeline = DocumentIngestionPipeline(config)
        files = pipeline.discover_files()
        
        assert files == []

    @patch.object(DocumentIngestionPipeline, '__init__', lambda x, y: None)
    def test_process_files_success(self, config, sample_documents, sample_chunks):
        """Test successful file processing and chunking."""
        pipeline = DocumentIngestionPipeline.__new__(DocumentIngestionPipeline)
        pipeline.config = config
        
        # Mock processor and chunker
        mock_processor = MagicMock()
        mock_processor.process_file.return_value = sample_documents
        pipeline.processor = mock_processor
        
        mock_chunker = MagicMock()
        mock_chunker.chunk_text.side_effect = [
            [sample_chunks[0], sample_chunks[1]],  # First document -> 2 chunks
            [sample_chunks[2]]  # Second document -> 1 chunk
        ]
        pipeline.chunker = mock_chunker
        
        files = [Path("test1.pdf"), Path("test2.txt")]
        chunks = pipeline.process_files(files)
        
        assert len(chunks) == 3
        assert chunks == sample_chunks
        mock_processor.process_file.assert_called_once_with(files)
        assert mock_chunker.chunk_text.call_count == 2

    @patch.object(DocumentIngestionPipeline, '__init__', lambda x, y: None)
    def test_process_files_chunking_error(self, config, sample_documents):
        """Test file processing when chunking fails for some documents."""
        pipeline = DocumentIngestionPipeline.__new__(DocumentIngestionPipeline)
        pipeline.config = config
        
        # Mock processor
        mock_processor = MagicMock()
        mock_processor.process_file.return_value = sample_documents
        pipeline.processor = mock_processor
        
        # Mock chunker to fail on first document but succeed on second
        mock_chunker = MagicMock()
        mock_chunker.chunk_text.side_effect = [
            Exception("Chunking failed"),  # First document fails
            [DocumentIngest(text="Chunk", metadata={"source_file": "test2.txt"})]  # Second succeeds
        ]
        pipeline.chunker = mock_chunker
        
        files = [Path("test1.pdf"), Path("test2.txt")]
        
        with patch('rag_solution.data_ingestion.ingest.logger') as mock_logger:
            chunks = pipeline.process_files(files)
        
        assert len(chunks) == 1
        assert chunks[0].metadata["source_file"] == "test2.txt"
        mock_logger.error.assert_called_once()

    @patch.object(DocumentIngestionPipeline, '__init__', lambda x, y: None)
    @pytest.mark.asyncio
    async def test_send_chunks_success(self, config, sample_chunks):
        """Test successful chunk sending."""
        pipeline = DocumentIngestionPipeline.__new__(DocumentIngestionPipeline)
        pipeline.config = config
        
        # Mock APIClient
        mock_client = AsyncMock()
        mock_client.send_documents.return_value = True
        
        with patch('rag_solution.data_ingestion.ingest.APIClient') as mock_api_class:
            mock_api_class.return_value.__aenter__.return_value = mock_client
            mock_api_class.return_value.__aexit__.return_value = None
            
            await pipeline.send_chunks(sample_chunks)
        
        # Should be called once since batch_size=10 and we have 3 chunks
        mock_client.send_documents.assert_called_once()

    @patch.object(DocumentIngestionPipeline, '__init__', lambda x, y: None)
    @pytest.mark.asyncio
    async def test_send_chunks_empty_list(self, config):
        """Test sending empty chunk list."""
        pipeline = DocumentIngestionPipeline.__new__(DocumentIngestionPipeline)
        pipeline.config = config
        
        with patch('rag_solution.data_ingestion.ingest.logger') as mock_logger:
            await pipeline.send_chunks([])
        
        mock_logger.warning.assert_called_with("No chunks to send")

    @patch.object(DocumentIngestionPipeline, '__init__', lambda x, y: None)
    @pytest.mark.asyncio
    async def test_send_chunks_batching(self, config):
        """Test chunk batching with multiple batches."""
        pipeline = DocumentIngestionPipeline.__new__(DocumentIngestionPipeline)
        config.batch_size = 2  # Force multiple batches
        pipeline.config = config
        
        # Create 5 chunks to test batching (should create 3 batches: 2, 2, 1)
        chunks = [
            DocumentIngest(text=f"Chunk {i}", metadata={"chunk_id": i})
            for i in range(5)
        ]
        
        mock_client = AsyncMock()
        mock_client.send_documents.return_value = True
        
        with patch('rag_solution.data_ingestion.ingest.APIClient') as mock_api_class:
            mock_api_class.return_value.__aenter__.return_value = mock_client
            mock_api_class.return_value.__aexit__.return_value = None
            
            await pipeline.send_chunks(chunks)
        
        # Should be called 3 times (3 batches)
        assert mock_client.send_documents.call_count == 3

    @patch.object(DocumentIngestionPipeline, '__init__', lambda x, y: None)
    @pytest.mark.asyncio
    async def test_send_chunks_partial_failure(self, config, sample_chunks):
        """Test handling of partial batch failures."""
        pipeline = DocumentIngestionPipeline.__new__(DocumentIngestionPipeline)
        config.batch_size = 1  # One chunk per batch
        pipeline.config = config
        
        mock_client = AsyncMock()
        # First batch succeeds, second fails, third succeeds
        mock_client.send_documents.side_effect = [True, False, True]
        
        with patch('rag_solution.data_ingestion.ingest.APIClient') as mock_api_class:
            mock_api_class.return_value.__aenter__.return_value = mock_client
            mock_api_class.return_value.__aexit__.return_value = None
            
            with patch('rag_solution.data_ingestion.ingest.logger') as mock_logger:
                await pipeline.send_chunks(sample_chunks)
        
        assert mock_client.send_documents.call_count == 3
        mock_logger.warning.assert_called_with("1 batches failed to process")

    @patch.object(DocumentIngestionPipeline, 'discover_files')
    @patch.object(DocumentIngestionPipeline, 'process_files')
    @patch.object(DocumentIngestionPipeline, 'send_chunks')
    @pytest.mark.asyncio
    async def test_run_success(self, mock_send_chunks, mock_process_files, 
                              mock_discover_files, config, sample_chunks):
        """Test successful pipeline run."""
        mock_discover_files.return_value = [Path("test1.pdf"), Path("test2.txt")]
        mock_process_files.return_value = sample_chunks
        mock_send_chunks.return_value = None
        
        pipeline = DocumentIngestionPipeline(config)
        
        with patch('rag_solution.data_ingestion.ingest.logger') as mock_logger:
            await pipeline.run()
        
        mock_discover_files.assert_called_once()
        mock_process_files.assert_called_once()
        mock_send_chunks.assert_called_once_with(sample_chunks)
        mock_logger.info.assert_any_call("Starting document ingestion pipeline")
        mock_logger.info.assert_any_call("Ingestion pipeline completed")

    @patch.object(DocumentIngestionPipeline, 'discover_files')
    @pytest.mark.asyncio
    async def test_run_no_files_found(self, mock_discover_files, config):
        """Test pipeline run when no files are found."""
        mock_discover_files.return_value = []
        
        pipeline = DocumentIngestionPipeline(config)
        
        with patch('rag_solution.data_ingestion.ingest.logger') as mock_logger:
            await pipeline.run()
        
        mock_logger.warning.assert_called_with("No files found to process")

    @patch.object(DocumentIngestionPipeline, 'discover_files')
    @patch.object(DocumentIngestionPipeline, 'process_files')
    @pytest.mark.asyncio
    async def test_run_no_chunks_generated(self, mock_process_files, 
                                          mock_discover_files, config):
        """Test pipeline run when no chunks are generated."""
        mock_discover_files.return_value = [Path("test1.pdf")]
        mock_process_files.return_value = []
        
        pipeline = DocumentIngestionPipeline(config)
        
        with patch('rag_solution.data_ingestion.ingest.logger') as mock_logger:
            await pipeline.run()
        
        mock_logger.warning.assert_called_with("No chunks generated")

    @patch.object(DocumentIngestionPipeline, 'discover_files')
    @pytest.mark.asyncio
    async def test_run_exception_handling(self, mock_discover_files, config):
        """Test pipeline exception handling."""
        mock_discover_files.side_effect = Exception("Test error")
        
        pipeline = DocumentIngestionPipeline(config)
        
        with pytest.raises(Exception) as exc_info:
            with patch('rag_solution.data_ingestion.ingest.logger') as mock_logger:
                await pipeline.run()
        
        assert "Test error" in str(exc_info.value)
        mock_logger.exception.assert_called_once()


class TestMainFunction:
    """Test cases for the main function."""

    @patch('rag_solution.data_ingestion.ingest.IngestionConfig')
    @patch.object(DocumentIngestionPipeline, 'run')
    @pytest.mark.asyncio
    async def test_main_function(self, mock_run, mock_config_class):
        """Test the main function."""
        from rag_solution.data_ingestion.ingest import main
        
        mock_config = MagicMock()
        mock_config_class.return_value = mock_config
        mock_run.return_value = None
        
        await main()
        
        mock_config_class.assert_called_once()
        mock_run.assert_called_once()


class TestIntegration:
    """Integration tests that test multiple components together."""

    @pytest.mark.asyncio
    async def test_pipeline_integration_with_mocks(self, config):
        """Test pipeline integration with mocked dependencies."""
        # Create a real pipeline instance
        pipeline = DocumentIngestionPipeline(config)
        
        # Mock the file system
        test_files = [Path("test1.pdf"), Path("test2.txt")]
        
        # Mock the processor to return documents
        mock_documents = [
            DocumentIngest(text="Document 1 content", metadata={"source_file": "test1.pdf"}),
            DocumentIngest(text="Document 2 content", metadata={"source_file": "test2.txt"}),
        ]
        
        # Mock the chunker to return chunks
        mock_chunks = [
            DocumentIngest(text="Chunk 1", metadata={"source_file": "test1.pdf", "chunk_id": 1}),
            DocumentIngest(text="Chunk 2", metadata={"source_file": "test2.txt", "chunk_id": 1}),
        ]
        
        with patch.object(pipeline, 'discover_files', return_value=test_files), \
             patch.object(pipeline.processor, 'process_file', return_value=mock_documents), \
             patch.object(pipeline.chunker, 'chunk_text', side_effect=[[mock_chunks[0]], [mock_chunks[1]]]), \
             patch('rag_solution.data_ingestion.ingest.APIClient') as mock_api_class:
            
            # Mock API client
            mock_client = AsyncMock()
            mock_client.send_documents.return_value = True
            mock_api_class.return_value.__aenter__.return_value = mock_client
            mock_api_class.return_value.__aexit__.return_value = None
            
            # Run the pipeline
            await pipeline.run()
            
            # Verify the flow
            mock_client.send_documents.assert_called_once()
            sent_chunks = mock_client.send_documents.call_args[0][0]
            assert len(sent_chunks) == 2
            assert sent_chunks[0].text == "Chunk 1"
            assert sent_chunks[1].text == "Chunk 2"
