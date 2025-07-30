import os
from pathlib import Path
from loguru import logger
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    _get_default_option,
)
from tqdm import tqdm as sync_tqdm

from model import DocumentIngest

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