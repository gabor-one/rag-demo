"""
Document ingestion script using docling with configurable chunking strategies.

This script processes PDF, Docling JSON and TXT files from a folder, chunks them using various strategies,
and sends them to a configurable API endpoint with retry logic and data validation.
"""

import asyncio
from pathlib import Path
from typing import List

from api_client import APIClient
from document_chunker import DocumentChunker
from document_processor import DocumentProcessor
from ingestion_config import IngestionConfig
from loguru import logger
from model import (
    DocumentIngest,
)
from tqdm.asyncio import tqdm


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
