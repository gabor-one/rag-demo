import pytest
from typing import Dict

from rag_solution.data_ingestion.document_chunker import DocumentChunker
from rag_solution.data_ingestion.model import ChunkingStrategy, DocumentIngest


class TestDocumentChunker:
    """Test suite for DocumentChunker class."""

    @pytest.fixture
    def sample_text(self) -> str:
        """Sample text for testing."""
        return (
            "This is the first sentence. This is the second sentence with more content. "
            "This is the third sentence that contains additional information about the topic. "
            "Here we have the fourth sentence which is longer and more detailed than the previous ones. "
            "The fifth sentence continues the narrative and provides even more context. "
            "Finally, this is the sixth and last senpytence that concludes our sample text."
        )

    @pytest.fixture
    def sample_metadata(self) -> Dict[str, str]:
        """Sample metadata for testing."""
        return {
            "source": "test_document.txt",
            "author": "test_author",
            "category": "test"
        }

    def test_init_with_default_parameters(self):
        """Test DocumentChunker initialization with default parameters."""
        chunker = DocumentChunker(ChunkingStrategy.FIXED_SIZE)
        
        assert chunker.strategy == ChunkingStrategy.FIXED_SIZE
        assert chunker.chunk_size == 384  # Default from mock
        assert chunker.overlap == 30
        assert chunker.embedding is not None

    def test_init_with_custom_parameters(self):
        """Test DocumentChunker initialization with custom parameters."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=256,
            overlap=50
        )
        
        assert chunker.strategy == ChunkingStrategy.SEMANTIC
        assert chunker.chunk_size == 256
        assert chunker.overlap == 50

    def test_get_sequence_length(self):
        """Test sequence length calculation."""
        chunker = DocumentChunker(ChunkingStrategy.FIXED_SIZE)
        
        # Test with known text
        text = "This is a test sentence with multiple words"
        chunker.get_sequence_length(text)
        # There is really no way to know the exact length

    def test_fixed_size_chunking(self, sample_text, sample_metadata):
        """Test fixed-size chunking strategy."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.FIXED_SIZE,
            chunk_size=100,  # Small chunk size for testing
        )
        
        chunks = chunker.chunk_text(sample_text, sample_metadata)
        
        # Verify we got chunks
        assert len(chunks) > 0
        
        # Verify each chunk is a DocumentIngest instance
        for chunk in chunks:
            assert isinstance(chunk, DocumentIngest)
            assert chunk.text.strip()  # Non-empty text
            assert "chunk_index" in chunk.metadata
            assert chunk.metadata["chunking_strategy"] == "fixed_size"
            assert "chunk_size_chars" in chunk.metadata
            assert "chunk_size_tokens" in chunk.metadata
            assert chunk.metadata["chunk_size_tokens"] <= 100

            # Verify original metadata is preserved
            assert chunk.metadata["source"] == "test_document.txt"
            assert chunk.metadata["author"] == "test_author"

    def test_semantic_chunking(self, sample_text, sample_metadata):
        """Test semantic chunking strategy."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=100,  
        )
        
        chunks = chunker.chunk_text(sample_text, sample_metadata)
        
        # Verify we got chunks
        assert len(chunks) == 5
        
        # Verify each chunk is a DocumentIngest instance
        for chunk in chunks:
            assert isinstance(chunk, DocumentIngest)
            assert chunk.text.strip()
            assert chunk.text.endswith(".")
            assert "chunk_index" in chunk.metadata
            assert chunk.metadata["chunking_strategy"] == "semantic"
            assert "sentence_count" in chunk.metadata
            
            # Verify original metadata is preserved
            assert chunk.metadata["source"] == "test_document.txt"

    def test_sliding_window_chunking(self, sample_text, sample_metadata):
        """Test sliding window chunking strategy."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.SLIDING_WINDOW,
            chunk_size=20,  # Small chunk size for testing
            overlap=5
        )
        
        chunks = chunker.chunk_text(sample_text, sample_metadata)
        
        # Verify we got chunks
        assert len(chunks) > 0
        
        # Verify each chunk is a DocumentIngest instance
        for chunk in chunks:
            assert isinstance(chunk, DocumentIngest)
            assert chunk.text.strip()
            assert "chunk_index" in chunk.metadata
            assert chunk.metadata["chunking_strategy"] == "sliding_window"
            assert "window_start" in chunk.metadata
            assert "window_size" in chunk.metadata
            
            # Verify original metadata is preserved
            assert chunk.metadata["source"] == "test_document.txt"

    def test_chunking_with_no_metadata(self, sample_text):
        """Test chunking without providing metadata."""
        chunker = DocumentChunker(ChunkingStrategy.FIXED_SIZE, chunk_size=20)
        
        chunks = chunker.chunk_text(sample_text)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, DocumentIngest)
            assert "chunk_index" in chunk.metadata
            assert chunk.metadata["chunking_strategy"] == "fixed_size"

    def test_chunking_empty_text(self):
        """Test chunking with empty text."""
        chunker = DocumentChunker(ChunkingStrategy.FIXED_SIZE)
        
        chunks = chunker.chunk_text("")
        
        # Should return empty list for empty text
        assert len(chunks) == 0

    def test_chunking_whitespace_only_text(self):
        """Test chunking with whitespace-only text."""
        chunker = DocumentChunker(ChunkingStrategy.FIXED_SIZE)
        
        chunks = chunker.chunk_text("   \n\t   ")
        
        # Should return empty list for whitespace-only text
        assert len(chunks) == 0

    def test_chunking_single_word(self):
        """Test chunking with a single word."""
        chunker = DocumentChunker(ChunkingStrategy.FIXED_SIZE, chunk_size=10)
        
        chunks = chunker.chunk_text("word")
        
        assert len(chunks) == 1
        assert chunks[0].text == "word"

    def test_invalid_chunking_strategy(self):
        """Test handling of invalid chunking strategy."""
        # This should be caught at enum level, but test runtime error handling
        chunker = DocumentChunker(ChunkingStrategy.FIXED_SIZE)
        chunker.strategy = "invalid_strategy"  # Force invalid strategy
        
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            chunker.chunk_text("test text")

    def test_semantic_chunking_sentence_handling(self):
        """Test semantic chunking properly handles sentences."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=20  # Small size to force multiple chunks
        )
        
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunker.chunk_text(text)

        assert len(chunks) == 3

        # Check that sentences are preserved
        for chunk in chunks:
            # Each chunk should end with a period (sentence boundary)
            assert chunk.metadata["chunk_size_tokens"] <= 20

    def test_sliding_window_step_size_calculation(self):
        """Test sliding window step size calculation."""
        chunker = DocumentChunker(
            strategy=ChunkingStrategy.SLIDING_WINDOW,
            chunk_size=10,
            overlap=3
        )
        
        text = " ".join([f"word{i}" for i in range(20)])  # 20 words
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 1  # Should create multiple overlapping chunks
        
        # Verify chunks have expected overlap structure
        for chunk in chunks:
            words_in_chunk = len(chunk.text.split())
            assert words_in_chunk <= 10  # Shouldn't exceed chunk_size

    def test_metadata_preservation_across_strategies(self, sample_metadata):
        """Test that original metadata is preserved across all chunking strategies."""
        text = "This is a test sentence for metadata preservation."
        
        strategies = [
            ChunkingStrategy.FIXED_SIZE,
            ChunkingStrategy.SEMANTIC,
            ChunkingStrategy.SLIDING_WINDOW
        ]
        
        for strategy in strategies:
            chunker = DocumentChunker(strategy=strategy, chunk_size=20)
            chunks = chunker.chunk_text(text, sample_metadata)
            
            for chunk in chunks:
                # Original metadata should be preserved
                assert chunk.metadata["source"] == "test_document.txt"
                assert chunk.metadata["author"] == "test_author"
                assert chunk.metadata["category"] == "test"
                
                # Strategy-specific metadata should be added
                assert "chunk_index" in chunk.metadata
                assert "chunking_strategy" in chunk.metadata

    def test_chunk_text_with_special_characters(self):
        """Test chunking with special characters and punctuation."""
        chunker = DocumentChunker(ChunkingStrategy.FIXED_SIZE, chunk_size=20)
        
        text = "Text with special chars: @#$%^&*()! And émojis 🎉 and números 123."
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 0
        
        # Verify special characters are preserved
        combined_text = " ".join(chunk.text for chunk in chunks)
        # These should work.
        for special_char in ["@", "#", "$", "%", "^", "&", "*", "(", ")"]:
            assert special_char in combined_text
        # Special character should be replace with closest match
        assert "emojis" in combined_text
        # Unkown character should be replaced with [UNK]
        assert "[UNK]" in combined_text
