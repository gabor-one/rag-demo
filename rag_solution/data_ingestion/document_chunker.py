from typing import Dict, List, Optional

from loguru import logger
from pydantic import ValidationError
from sentence_transformers import SentenceTransformer

from model import ChunkingStrategy, DocumentIngest


class DocumentChunker:
    """Handles different document chunking strategies."""

    def __init__(
        self,
        strategy: ChunkingStrategy,
        chunk_size: int | None = None,
        overlap: int = 30,
    ):
        """
        Initialize the DocumentChunker with a specific strategy and parameters.
        Args:
            strategy (ChunkingStrategy): The chunking strategy to use.
            chunk_size (int | None): The size of each chunk in characters. If None, uses max_sequence_length of the embedding model.
            overlap (int): The number of characters to overlap between chunks.
        """
        self.embedding = SentenceTransformer("all-mpnet-base-v2")

        self.strategy = strategy
        self.chunk_size = chunk_size or self.embedding.max_seq_length
        self.overlap = overlap

    def get_sequence_length(self, text: str) -> int:
        return self.embedding.tokenize(text)["input_ids"].shape[0]

    def chunk_text(
        self, text: str, metadata: Optional[Dict[str, str]] = None
    ) -> List[DocumentIngest]:
        """Chunk text based on the configured strategy."""
        if metadata is None:
            metadata = {}

        if self.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._fixed_size_chunking(text, metadata)
        elif self.strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic_chunking(text, metadata)
        elif self.strategy == ChunkingStrategy.SLIDING_WINDOW:
            return self._sliding_window_chunking(text, metadata)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")

    def _fixed_size_chunking(
        self, text: str, metadata: Dict[str, str]
    ) -> List[DocumentIngest]:
        """Split text into fixed-size chunks by token count, breaking at spaces."""
        chunks = []
        start = 0
        text = text.strip()
        # Use the models tokenizer to get proper string tokens, aka words
        words = self.embedding.tokenizer.tokenize(text)
        chunk_index = 0

        while start < len(words):
            # Start with a reasonable estimate and adjust
            # Sadly tokenizer.tokenize is not perfect, so we need to approach step by step.
            # We could move ahead more to start, but the sake of safety we start with 1.
            end = start + 1

            # Find the right end position by checking token length
            while end <= len(words):
                # Sanity check.
                chunk_text = self.embedding.tokenizer.convert_tokens_to_string(
                    words[start:end]
                )
                token_length = self.get_sequence_length(chunk_text)

                if token_length <= self.chunk_size:
                    # This chunk fits, try to add more words
                    if end == len(words):
                        # We've reached the end of the text
                        break
                    if token_length == self.chunk_size:
                        # Perfect fit, no need to extend
                        break
                    end += 1
                else:
                    # This chunk is too big, use the previous end
                    end -= 1
                    break

            chunk_text = self.embedding.tokenizer.convert_tokens_to_string(
                words[start:end]
            ).strip()
            if not chunk_text:
                break

            token_length = self.get_sequence_length(chunk_text)
            chunk_metadata = metadata.copy()
            chunk_metadata.update(
                {
                    "chunk_index": chunk_index,
                    "chunking_strategy": "fixed_size",
                    "chunk_size_chars": len(chunk_text),
                    "chunk_size_tokens": token_length,
                }
            )

            try:
                chunks.append(DocumentIngest(text=chunk_text, metadata=chunk_metadata))
            except ValidationError as e:
                logger.warning(f"Skipping invalid chunk: {e}")

            start = end
            chunk_index += 1

        return chunks

    def _semantic_chunking(
        self, text: str, metadata: Dict[str, str]
    ) -> List[DocumentIngest]:
        """Split text into semantic chunks (simplified version using sentences)."""
        # Simple semantic chunking based on sentences
        sentences = text.split(". ")
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())

            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = ". ".join(current_chunk) + (
                    "." if not current_chunk[-1].endswith(".") else ""
                )

                chunk_metadata = metadata.copy()
                chunk_metadata.update(
                    {
                        "chunk_index": str(len(chunks)),
                        "chunking_strategy": "semantic",
                        "sentence_count": str(len(current_chunk)),
                    }
                )

                try:
                    chunks.append(
                        DocumentIngest(text=chunk_text, metadata=chunk_metadata)
                    )
                except ValidationError as e:
                    logger.warning(f"Skipping invalid chunk: {e}")

                # Start new chunk with overlap
                if self.overlap > 0 and len(current_chunk) > 1:
                    overlap_sentences = current_chunk[-1:]
                    current_chunk = overlap_sentences + [sentence]
                    current_length = sum(len(s.split()) for s in current_chunk)
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add remaining chunk
        if current_chunk:
            chunk_text = ". ".join(current_chunk) + (
                "." if not current_chunk[-1].endswith(".") else ""
            )
            chunk_metadata = metadata.copy()
            chunk_metadata.update(
                {
                    "chunk_index": str(len(chunks)),
                    "chunking_strategy": "semantic",
                    "sentence_count": str(len(current_chunk)),
                }
            )

            try:
                chunks.append(DocumentIngest(text=chunk_text, metadata=chunk_metadata))
            except ValidationError as e:
                logger.warning(f"Skipping invalid chunk: {e}")

        return chunks

    def _sliding_window_chunking(
        self, text: str, metadata: Dict[str, str]
    ) -> List[DocumentIngest]:
        """Split text using sliding window approach."""
        words = text.split()
        chunks = []

        step_size = max(1, self.chunk_size - self.overlap)

        for i in range(0, len(words), step_size):
            chunk_words = words[i : i + self.chunk_size]
            if (
                len(chunk_words) < self.chunk_size // 2
            ):  # Skip very small chunks at the end
                break

            chunk_text = " ".join(chunk_words)

            chunk_metadata = metadata.copy()
            chunk_metadata.update(
                {
                    "chunk_index": str(len(chunks)),
                    "chunking_strategy": "sliding_window",
                    "window_start": str(i),
                    "window_size": str(len(chunk_words)),
                }
            )

            try:
                chunks.append(DocumentIngest(text=chunk_text, metadata=chunk_metadata))
            except ValidationError as e:
                logger.warning(f"Skipping invalid chunk: {e}")
                continue

        return chunks
