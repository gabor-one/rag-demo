# This file is mostly AI generated. Very few lines were manually edited.
# Not super interesting.

from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import pytest

from rag_solution.data_ingestion.document_processor import DocumentProcessor
from rag_solution.data_ingestion.model import DocumentIngest


class TestDocumentProcessor:
    """Test suite for DocumentProcessor class."""

    @pytest.fixture
    def mock_converter(self):
        """Mock DocumentConverter for testing."""
        converter = Mock()
        converter.convert_all = Mock()
        return converter

    @pytest.fixture
    def sample_docling_result(self):
        """Create a sample docling conversion result."""
        result = Mock()
        result.input = Mock()
        result.input.file = Mock()
        result.input.file.name = "test_document.pdf"
        result.document = Mock()
        result.document.export_to_markdown = Mock(return_value="# Test Document\n\nThis is test content.")
        result.errors = []
        return result

    @pytest.fixture
    def sample_error_result(self):
        """Create a sample docling result with errors."""
        result = Mock()
        result.input = Mock()
        result.input.file = Mock()
        result.input.file.name = "error_document.pdf"
        result.document = Mock()
        result.document.export_to_markdown = Mock(return_value="")
        result.errors = ["Failed to process document"]
        return result

    def test_init_creates_converter_with_correct_options(self):
        """Test that DocumentProcessor initializes with correct options."""
        with patch('rag_solution.data_ingestion.document_processor.DocumentConverter') as mock_converter_class, \
             patch('rag_solution.data_ingestion.document_processor._get_default_option') as mock_default_option, \
             patch('os.cpu_count', return_value=4):
            
            mock_default_option.return_value = Mock()
            mock_default_option.return_value.pipeline_options = Mock()
            
            processor = DocumentProcessor()
            
            # Verify processor was created
            assert processor is not None
            assert processor.converter is not None
            
            # Verify DocumentConverter was called with format options
            mock_converter_class.assert_called_once()
            call_args = mock_converter_class.call_args
            assert 'format_options' in call_args.kwargs
            
            # Verify accelerator options are set correctly
            format_options = call_args.kwargs['format_options']
            assert len(format_options) == 2  # PDF and JSON_DOCLING formats

    def test_process_file_single_pdf(self, sample_docling_result):
        """Test processing a single PDF file."""
        with patch('rag_solution.data_ingestion.document_processor.DocumentConverter') as mock_converter_class:
            mock_converter = Mock()
            mock_converter.convert_all = Mock(return_value=[sample_docling_result])
            mock_converter_class.return_value = mock_converter
            
            processor = DocumentProcessor()
            file_paths = [Path("test_document.pdf")]
            
            results = processor.process_file(file_paths)
            
            assert len(results) == 1
            assert isinstance(results[0], DocumentIngest)
            assert results[0].text == "# Test Document\n\nThis is test content."
            assert results[0].metadata["source_file"] == "test_document.pdf"
            mock_converter.convert_all.assert_called_once()

    def test_process_file_single_txt(self):
        """Test processing a single TXT file."""
        with patch('rag_solution.data_ingestion.document_processor.DocumentConverter'), \
             patch('builtins.open', mock_open(read_data="This is plain text content.")):
            
            processor = DocumentProcessor()
            file_paths = [Path("test_document.txt")]
            
            results = processor.process_file(file_paths)
            
            assert len(results) == 1
            assert isinstance(results[0], DocumentIngest)
            assert results[0].text == "This is plain text content."
            assert results[0].metadata["source_file"] == "test_document.txt"

    def test_process_file_mixed_formats(self, sample_docling_result):
        """Test processing mixed file formats (PDF and TXT)."""
        with patch('rag_solution.data_ingestion.document_processor.DocumentConverter') as mock_converter_class, \
             patch('builtins.open', mock_open(read_data="Plain text content")):
            
            mock_converter = Mock()
            mock_converter.convert_all = Mock(return_value=[sample_docling_result])
            mock_converter_class.return_value = mock_converter
            
            processor = DocumentProcessor()
            file_paths = [Path("document.pdf"), Path("document.txt")]
            
            results = processor.process_file(file_paths)
            
            assert len(results) == 2
            
            # Check TXT file result
            txt_result = next(r for r in results if r.metadata["source_file"] == "document.txt")
            assert txt_result.text == "Plain text content"
            
            # Check PDF file result
            pdf_result = next(r for r in results if r.metadata["source_file"] == "test_document.pdf")
            assert pdf_result.text == "# Test Document\n\nThis is test content."

    def test_process_file_batch_processing(self, sample_docling_result):
        """Test batch processing with multiple files."""
        with patch('rag_solution.data_ingestion.document_processor.DocumentConverter') as mock_converter_class:
            mock_converter = Mock()
            # Return different results for each batch
            mock_converter.convert_all = Mock(side_effect=[
                [sample_docling_result],  # First batch
                [sample_docling_result],  # Second batch
            ])
            mock_converter_class.return_value = mock_converter
            
            processor = DocumentProcessor()
            # Create 25 files to test batching (batch_size = 20)
            file_paths = [Path(f"document_{i}.pdf") for i in range(25)]
            
            results = processor.process_file(file_paths)
            
            # Should call convert_all twice (2 batches)
            assert mock_converter.convert_all.call_count == 2
            assert len(results) == 2  # One result per batch

    def test_process_file_with_errors(self, sample_error_result):
        """Test processing files with conversion errors."""
        with patch('rag_solution.data_ingestion.document_processor.DocumentConverter') as mock_converter_class, \
             patch('rag_solution.data_ingestion.document_processor.logger') as mock_logger:
            
            mock_converter = Mock()
            mock_converter.convert_all = Mock(return_value=[sample_error_result])
            mock_converter_class.return_value = mock_converter
            
            processor = DocumentProcessor()
            file_paths = [Path("error_document.pdf")]
            
            results = processor.process_file(file_paths)
            
            # Should return empty list for files with errors
            assert len(results) == 0
            
            # Should log warning about errors
            mock_logger.warning.assert_called_once()
            assert "Errors processing error_document.pdf" in str(mock_logger.warning.call_args)

    def test_process_file_txt_read_error(self):
        """Test handling of TXT file read errors."""
        with patch('rag_solution.data_ingestion.document_processor.DocumentConverter'), \
             patch('builtins.open', side_effect=FileNotFoundError("File not found")), \
             patch('rag_solution.data_ingestion.document_processor.logger') as mock_logger:
            
            processor = DocumentProcessor()
            file_paths = [Path("missing_file.txt")]
            
            results = processor.process_file(file_paths)
            
            # Should return empty list when file can't be read
            assert len(results) == 0
            
            # Should log warning about read failure
            mock_logger.warning.assert_called_once()
            assert "Failed to read" in str(mock_logger.warning.call_args)

    def test_process_file_empty_list(self):
        """Test processing an empty list of files."""
        with patch('rag_solution.data_ingestion.document_processor.DocumentConverter'):
            processor = DocumentProcessor()
            
            results = processor.process_file([])
            
            assert len(results) == 0

    def test_process_file_strips_whitespace(self):
        """Test that text content is properly stripped of whitespace."""
        with patch('rag_solution.data_ingestion.document_processor.DocumentConverter'), \
             patch('builtins.open', mock_open(read_data="  \n  Content with whitespace  \n  ")):
            
            processor = DocumentProcessor()
            file_paths = [Path("whitespace.txt")]
            
            results = processor.process_file(file_paths)
            
            assert len(results) == 1
            assert results[0].text == "Content with whitespace"

    def test_process_file_txt_encoding_handling(self):
        """Test handling of different text file encodings."""
        with patch('rag_solution.data_ingestion.document_processor.DocumentConverter'), \
             patch('builtins.open', mock_open(read_data="Content with special chars: café")):
            
            processor = DocumentProcessor()
            file_paths = [Path("encoded.txt")]
            
            results = processor.process_file(file_paths)
            
            assert len(results) == 1
            assert "café" in results[0].text

    def test_process_file_progress_bar(self, sample_docling_result):
        """Test that progress bar is displayed during processing."""
        with patch('rag_solution.data_ingestion.document_processor.DocumentConverter') as mock_converter_class, \
             patch('rag_solution.data_ingestion.document_processor.sync_tqdm') as mock_tqdm:
            
            mock_converter = Mock()
            mock_converter.convert_all = Mock(return_value=[sample_docling_result])
            mock_converter_class.return_value = mock_converter
            
            # Mock tqdm to return an iterable
            mock_tqdm.return_value = [0]  # Single batch
            
            processor = DocumentProcessor()
            file_paths = [Path("document.pdf")]
            
            processor.process_file(file_paths)
            
            # Verify tqdm was called with correct parameters
            mock_tqdm.assert_called_once()
            call_args = mock_tqdm.call_args
            assert "Processing files" in str(call_args)
            assert "unit" in call_args.kwargs
            assert call_args.kwargs["unit"] == "batch"

    def test_process_file_large_batch(self):
        """Test processing a large number of files exceeding batch size."""
        with patch('rag_solution.data_ingestion.document_processor.DocumentConverter') as mock_converter_class, \
             patch('builtins.open', mock_open(read_data="content")):
            
            mock_converter = Mock()
            mock_converter.convert_all = Mock(return_value=[])
            mock_converter_class.return_value = mock_converter
            
            processor = DocumentProcessor()
            # Create 45 files (will require 3 batches of 20, 20, 5)
            file_paths = [Path(f"file_{i}.txt") for i in range(45)]
            
            results = processor.process_file(file_paths)
            
            # All TXT files should be processed directly, no docling calls
            assert mock_converter.convert_all.call_count == 0
            assert len(results) == 45

    @pytest.mark.parametrize("file_extension", [".pdf", ".docx", ".json"])
    def test_process_file_non_txt_extensions(self, file_extension, sample_docling_result):
        """Test processing files with various non-TXT extensions."""
        with patch('rag_solution.data_ingestion.document_processor.DocumentConverter') as mock_converter_class:
            mock_converter = Mock()
            mock_converter.convert_all = Mock(return_value=[sample_docling_result])
            mock_converter_class.return_value = mock_converter
            
            processor = DocumentProcessor()
            file_paths = [Path(f"document{file_extension}")]
            
            results = processor.process_file(file_paths)
            
            # Should use docling for non-TXT files
            mock_converter.convert_all.assert_called_once()
            assert len(results) == 1

    def test_process_file_case_insensitive_txt(self):
        """Test that TXT file detection is case insensitive."""
        with patch('rag_solution.data_ingestion.document_processor.DocumentConverter') as mock_converter_class, \
             patch('builtins.open', mock_open(read_data="content")):
            
            mock_converter = Mock()
            mock_converter_class.return_value = mock_converter
            
            processor = DocumentProcessor()
            file_paths = [Path("document.TXT"), Path("document.Txt")]
            
            results = processor.process_file(file_paths)
            
            # Should not call docling for uppercase TXT extensions
            mock_converter.convert_all.assert_not_called()
            assert len(results) == 2

    def test_document_ingest_creation(self, sample_docling_result):
        """Test that DocumentIngest objects are created correctly."""
        with patch('rag_solution.data_ingestion.document_processor.DocumentConverter') as mock_converter_class:
            mock_converter = Mock()
            mock_converter.convert_all = Mock(return_value=[sample_docling_result])
            mock_converter_class.return_value = mock_converter
            
            processor = DocumentProcessor()
            file_paths = [Path("test.pdf")]
            
            results = processor.process_file(file_paths)
            
            assert len(results) == 1
            doc = results[0]
            
            # Verify DocumentIngest structure
            assert isinstance(doc, DocumentIngest)
            assert hasattr(doc, 'text')
            assert hasattr(doc, 'metadata')
            assert isinstance(doc.metadata, dict)
            assert 'source_file' in doc.metadata
            assert doc.metadata['source_file'] == 'test_document.pdf'

    def test_process_file_logging(self):
        """Test that appropriate logging occurs during processing."""
        with patch('rag_solution.data_ingestion.document_processor.DocumentConverter'), \
             patch('rag_solution.data_ingestion.document_processor.logger') as mock_logger:
            
            processor = DocumentProcessor()
            file_paths = [Path("file1.txt"), Path("file2.txt")]
            
            with patch('builtins.open', mock_open(read_data="content")):
                processor.process_file(file_paths)
            
            # Should log processing start
            mock_logger.info.assert_called_with("Processing 2 files.")

    def test_dummy_result_class_functionality(self):
        """Test the internal DummyResult class for TXT files."""
        with patch('rag_solution.data_ingestion.document_processor.DocumentConverter'), \
             patch('builtins.open', mock_open(read_data="test content")):
            
            processor = DocumentProcessor()
            file_paths = [Path("test.txt")]
            
            results = processor.process_file(file_paths)
            
            assert len(results) == 1
            result = results[0]
            
            # Verify the result mimics docling structure
            assert result.text == "test content"
            assert result.metadata["source_file"] == "test.txt"
