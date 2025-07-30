#!/usr/bin/env python3
"""
Example usage of the document ingestion pipeline.

This script demonstrates how to configure and run the document ingestion
pipeline with different settings.
"""

import asyncio
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

# Import the ingestion modules
from ingest import (
    DocumentIngestionPipeline,
    IngestionConfig,
)
from loguru import logger
from model import ChunkingStrategy

logger.remove()
logger.add(sys.stdout, level="INFO")


async def example_basic_usage(chunking_strategy=ChunkingStrategy.FIXED_SIZE):
    """Basic usage example with minimal configuration."""
    logger.info("=== Basic Usage Example ===")

    # Create a temporary directory with sample documents
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create sample documents
        (temp_path / "iphone_16.txt").write_text(
            "The iPhone 16 is Apple's latest flagship smartphone, featuring a sleek design, advanced camera system, and a powerful new processor. "
            "It boasts a brighter, more energy-efficient display, improved battery life, and enhanced AI capabilities for photography and productivity. "
            "With iOS 18, users enjoy new customization options, privacy features, and seamless integration with the Apple ecosystem. "
            "The iPhone 16 sets a new standard for performance, innovation, and user experience."
        )

        (temp_path / "toyota_corolla.txt").write_text(
            "The Toyota Corolla is a compact car renowned for its reliability, fuel efficiency, and affordability. "
            "First introduced in 1966, the Corolla has become one of the best-selling vehicles worldwide. "
            "It features a comfortable interior, advanced safety features, and a reputation for low maintenance costs. "
            "The latest models offer modern technology, efficient hybrid options, and a smooth driving experience, making the Corolla a popular choice for drivers seeking practicality and value."
        )

        # Configure the ingestion pipeline
        config = IngestionConfig(
            input_folder=str(temp_path),
            chunking_strategy=chunking_strategy,
        )

        # Create and run the pipeline
        pipeline = DocumentIngestionPipeline(config)

        try:
            await pipeline.run()
            print("✓ Basic example completed successfully!")
        except Exception as e:
            print(f"✗ Basic example failed: {e}")


async def large_pdf_dataset_ingestion_from_HF(
    extract_tar_count=1, max_extracted_files=20, chunking_strategy=ChunkingStrategy.FIXED_SIZE
):
    """This example demonstrates how to ingest a large PDF dataset from Hugging Face."""
    logger.info("=== Large PDF Dataset Ingestion from Hugging Face ===")

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    ## Prepare a large amount of test data from Hugging Face

    raw_data_dir = Path(__file__).parent.parent / "data" / "raw_data"

    # The whole repo is 1.5T... Maybe not now.
    allowed_patterns = [f"pdfa-eng-train-{i:04d}.tar" for i in range(extract_tar_count)]

    hf = HfApi()
    hf.snapshot_download(
        "pixparse/pdfa-eng-wds",
        repo_type="dataset",
        local_dir_use_symlinks=False,
        local_dir=raw_data_dir,
        allow_patterns=allowed_patterns,
        resume_download=True,
    )

    # Extract all tar files from raw_data_dir to another directory

    extracted_dir = raw_data_dir.parent / "extracted_files"
    extracted_dir.mkdir(exist_ok=True)

    # Extract each tar file if not already extracted
    extracted_files = 0
    skipped_count = 0
    for tar_path in raw_data_dir.glob("*.tar"):
        with tarfile.open(tar_path, "r") as tar:
            # Get all member names in the tar file
            tar_filenames = [
                member.name for member in tar.getmembers() if member.isfile()
            ]
            # Check if all files already exist in extracted_dir
            if all((extracted_dir / name).exists() for name in tar_filenames):
                skipped_count += 1
                extracted_files += len(tar_filenames)
                continue

            budget_left = max_extracted_files - extracted_files
            if budget_left <= 0:
                break

            members_to_extract = tar_filenames[:budget_left]
            tar.extractall(path=extracted_dir, members=members_to_extract)
            extracted_files += len(members_to_extract)

    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} tar files that were already extracted.")

    ## Run the ingestion pipeline on the extracted files

    config = IngestionConfig(
        input_folder=str(extracted_dir),
        chunking_strategy=chunking_strategy,
        chunk_size=None,
    )

    pipeline = DocumentIngestionPipeline(config)

    try:
        await pipeline.run()
        logger.info("✓ Large PDF dataset ingestion completed successfully!")
    except Exception as e:
        logger.error(f"✗ Large PDF dataset ingestion failed: {e}")
    finally:
        # Clean up extracted_dir
        shutil.rmtree(extracted_dir)


async def main():
    logger.info("Document Ingestion Pipeline Examples")
    logger.info("====================================")

    ## Select the chunking strategy for the examples
    chunking_strategy = ChunkingStrategy.SEMANTIC

    ### Select example to run
    # Generate small text files and ingest them.
    # await example_basic_usage(chunking_strategy=chunking_strategy)

    # This download a large PDF dataset from Hugging Face and ingests it.
    # Full dataset is 1.5TB so be mindful of your disk space.
    # Also while there is internal batching of subtasks, all the chunks are collected in memory.
    #   Streaming can be implemented if larger-than-memory dataset processing is needed.
    await large_pdf_dataset_ingestion_from_HF(
        extract_tar_count=1, max_extracted_files=20, chunking_strategy=chunking_strategy
    )

    logger.info("\n" + "=" * 50)
    logger.info("Ingestion completed!")


if __name__ == "__main__":
    asyncio.run(main())
