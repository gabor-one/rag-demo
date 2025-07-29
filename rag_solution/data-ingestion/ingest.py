"""
Document ingestion script using docling with configurable chunking strategies.

This script processes PDF, Docling JSON and TXT files from a folder, chunks them using various strategies,
and sends them to a configurable API endpoint with retry logic and data validation.
"""

import asyncio
import os
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    _get_default_option,
)
from ingestion_config import IngestionConfig
from loguru import logger
from model import (
    ChunkingStrategy,
    DocumentIngest,
    DocumentsIngestRequest,
)
from pydantic import ValidationError
from tqdm import tqdm as sync_tqdm
from tqdm.asyncio import tqdm


class DocumentChunker:
    """Handles different document chunking strategies."""

    def __init__(
        self, strategy: ChunkingStrategy, chunk_size: int = 1000, overlap: int = 200
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap

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
        """Split text into fixed-size chunks by character count, breaking at spaces."""
        chunks = []
        start = 0
        text = text.strip()
        n = len(text)
        chunk_index = 0

        while start < n:
            # Find the max end index for this chunk
            end = min(start + self.chunk_size, n)
            if end < n:
                # Try to break at the last space before end
                space = text.rfind(" ", start, end)
                if space > start:
                    end = space
            chunk_text = text[start:end].strip()
            if not chunk_text:
                break

            chunk_metadata = metadata.copy()
            chunk_metadata.update(
                {
                    "chunk_index": str(chunk_index),
                    "chunking_strategy": "fixed_size",
                    "chunk_size": str(len(chunk_text)),
                }
            )

            try:
                chunks.append(DocumentIngest(text=chunk_text, metadata=chunk_metadata))
            except ValidationError as e:
                logger.warning(f"Skipping invalid chunk: {e}")

            start = end
            while start < n and text[start] == " ":
                start += 1

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


class DocumentProcessor:
    """Handles document processing using docling."""

    def __init__(self):
        # Configure docling options

        accelerator_options = AcceleratorOptions(
            num_threads=os.cpu_count(), device=AcceleratorDevice.AUTO
        )

        pipeline_options = PdfPipelineOptions()
        # For the sake of performance we turn them off
        # Low-cost GPU hosts (like T4) can be used to run these models and improve quality.
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = False
        pipeline_options.accelerator_options = accelerator_options

        json_docling_option = _get_default_option(InputFormat.JSON_DOCLING)
        json_docling_option.pipeline_options.accelerator_options = accelerator_options

        # We can define configs for different file types.
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                InputFormat.JSON_DOCLING: json_docling_option,
            }
        )

    def process_file(self, file_paths: list[Path]) -> list[DocumentIngest]:
        """Process a single file and extract text content."""

        logger.info(f"Processing {len(file_paths)} files.")

        # Use docling to convert files
        # Process files in batches of 20 and show progress bar
        results = []
        batch_size = 20
        total_files = len(file_paths)

        for i in sync_tqdm(
            range(0, total_files, batch_size), desc="Processing files", unit="batch"
        ):
            batch = file_paths[i : i + batch_size]
            # Separate .txt files and others
            txt_files = [f for f in batch if f.suffix.lower() == ".txt"]
            other_files = [f for f in batch if f.suffix.lower() != ".txt"]

            # Process .txt files: read content and add to results directly
            for txt_file in txt_files:
                try:
                    with open(txt_file, "r", encoding="utf-8") as f:
                        text = f.read()

                    # Mimic docling result structure for consistency
                    class DummyResult:
                        def __init__(self, file, text):
                            self.input = type("Input", (), {"file": file})()
                            self.document = type(
                                "Document",
                                (),
                                {"export_to_markdown": lambda self: text},
                            )()
                            self.errors = []

                    results.append(DummyResult(txt_file, text))
                except Exception as e:
                    logger.warning(f"Failed to read {txt_file}: {e}")

            # Process other files with docling
            if other_files:
                batch_results = list(self.converter.convert_all(other_files))
                results.extend(batch_results)

        # Warn about any errors.
        # Custom handling can be added here if needed.
        for doc in results:
            if doc.errors:
                logger.warning(f"Errors processing {doc.input.file.name}: {doc.errors}")

        documents = [
            # Using markdown as conversion format for better text extraction
            DocumentIngest(
                text=result.document.export_to_markdown().strip(),
                metadata={
                    "source_file": result.input.file.name,
                },
            )
            for result in results
            if not result.errors
        ]

        return documents


class APIClient:
    """Handles API communication with retry logic."""

    def __init__(self, config: IngestionConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=self.config.api_timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def send_documents(self, documents: List[DocumentIngest]) -> bool:
        """Send documents to the API endpoint with retry logic."""
        if not self.session:
            raise RuntimeError("APIClient not properly initialized")

        request_data = DocumentsIngestRequest(documents=documents)

        for attempt in range(self.config.max_retries + 1):
            try:
                async with self.session.post(
                    self.config.api_endpoint,
                    json=request_data.model_dump(),
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status == 200:
                        logger.debug(f"Successfully sent {len(documents)} documents")
                        return True
                    elif response.status == 429:
                        # Rate limit - retry with exponential backoff
                        # Let the server tell us when to retry
                        # SlowApi fills this header on 429
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            delay = int(retry_after)
                        else:
                            delay = (2**attempt) * self.config.retry_delay

                        logger.warning(
                            f"Rate limited. Retrying after {delay} seconds..."
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        response_text = await response.text()
                        logger.error(f"API error {response.status}: {response_text}")

                        if attempt < self.config.max_retries:
                            delay = (2**attempt) * self.config.retry_delay
                            logger.info(
                                f"Retrying in {delay} seconds... (attempt {attempt + 1}/{self.config.max_retries})"
                            )
                            await asyncio.sleep(delay)
                        else:
                            logger.error("Max retries exceeded for batch")
                            return False

            except (asyncio.TimeoutError, Exception) as e:
                if e is asyncio.TimeoutError:
                    logger.error(f"Request timed out (attempt {attempt + 1}): {e}")
                else:
                    logger.error(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries:
                    delay = (2**attempt) * self.config.retry_delay
                    await asyncio.sleep(delay)
                else:
                    return False

        return False


class DocumentIngestionPipeline:
    """Main pipeline for document ingestion."""

    def __init__(self, config: IngestionConfig):
        self.config = config
        self.processor = DocumentProcessor()
        self.chunker = DocumentChunker(
            strategy=config.chunking_strategy,
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
        )

    def discover_files(self) -> List[Path]:
        """Discover files to process in the input folder."""
        input_path = Path(self.config.input_folder)
        if not input_path.exists():
            raise FileNotFoundError(f"Input folder does not exist: {input_path}")

        files = []
        # Only select .pdf and .txt files (case-insensitive)
        for ext in self.config.file_extensions:
            files.extend(
                f
                for f in input_path.rglob(f"*{ext}")
                if f.suffix.lower() in [".pdf", ".txt"]
            )

        logger.info(f"Discovered {len(files)} files to process")
        return files

    def process_files(self, files: List[Path]) -> List[DocumentIngest]:
        """Process all files and generate chunks."""
        all_chunks = []

        documents = self.processor.process_file(files)

        for document in documents:
            # Generate chunks
            try:
                chunks = self.chunker.chunk_text(document.text, document.metadata)
                all_chunks.extend(chunks)
                logger.debug(
                    f"Generated {len(chunks)} chunks from {document.metadata['source_file']}"
                )
            except Exception as e:
                logger.error(
                    f"Error chunking file {document.metadata['source_file']}: {e}"
                )
                continue

        logger.info(f"Total chunks generated: {len(all_chunks)}")
        return all_chunks

    async def send_chunks(self, chunks: List[DocumentIngest]) -> None:
        """Send chunks to API in batches with concurrency control."""
        if not chunks:
            logger.warning("No chunks to send")
            return

        # Create batches
        batches = [
            chunks[i : i + self.config.batch_size]
            for i in range(0, len(chunks), self.config.batch_size)
        ]

        logger.info(f"Sending {len(chunks)} chunks in {len(batches)} batches")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        async def send_batch(batch: List[DocumentIngest], batch_index: int) -> bool:
            async with semaphore:
                async with APIClient(self.config) as client:
                    return await client.send_documents(batch)

        # Send all batches concurrently with TQDM progress bar
        tasks = [send_batch(batch, i) for i, batch in enumerate(batches)]
        results = []
        for coro in tqdm.as_completed(tasks, total=len(tasks), desc="Sending batches"):
            result = await coro
            results.append(result)

        # Count successes and failures
        successes = sum(1 for result in results if result is True)
        failures = len(results) - successes

        logger.info(
            f"Ingestion complete: {successes} successful batches, {failures} failed batches"
        )

        if failures > 0:
            logger.warning(f"{failures} batches failed to process")

    async def run(self) -> None:
        """Run the complete ingestion pipeline."""
        try:
            logger.info("Starting document ingestion pipeline")
            logger.info(f"Configuration: {self.config.model_dump()}")

            # Discover files
            files = self.discover_files()
            if not files:
                logger.warning("No files found to process")
                return

            # Process files and generate chunks
            chunks = self.process_files(files)
            if not chunks:
                logger.warning("No chunks generated")
                return

            # Send chunks to API
            await self.send_chunks(chunks)

            logger.info("Ingestion pipeline completed")

        except Exception as e:
            logger.exception(f"Pipeline failed: {e}")
            raise


async def main():
    """Main entry point."""
    config = IngestionConfig()
    pipeline = DocumentIngestionPipeline(config)
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())
