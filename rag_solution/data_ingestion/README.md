# Document Ingestion Pipeline

A comprehensive Python script for ingesting PDF and text documents using docling, with configurable chunking strategies and asynchronous API submission.

## Features

- **Multiple File Formats**: Support for PDF, Docling JSON and TXT files using docling for advanced document processing
- **Configurable Chunking Strategies**:
  - Fixed-size chunking with word-based splitting
  - Semantic chunking based on sentence boundaries
  - Sliding window chunking with configurable overlap
- **Asynchronous Processing**: Parallel document processing and API submission
- **Robust Retry Logic**: 
  - Infinite retries for 429 (rate limit) responses
  - Configurable retries (default 5) for other error codes
  - Exponential backoff for all retry scenarios
- **Data Validation**: Comprehensive validation using Pydantic models
- **Quality Checks**: Content validation and metadata enrichment
- **Configurable Settings**: Environment variable support with sensible defaults

## System Dependencies

- UV: Python package and project manager. [Installation manual](https://docs.astral.sh/uv/getting-started/installation/).

## Installation

Dependencies are in "pipeline" dependency group. For the sake of simplicity no new project has been created for this script.

```bash
uv sync --group pipeline
```

### Required Dependencies

| Package              | Purpose                                                      |
|----------------------|--------------------------------------------------------------|
| `aiohttp`            | Asynchronous HTTP client for API requests                    |
| `docling`            | Advanced document parsing and chunking                       |
| `hf-transfer`        | Efficient file transfer from Hugging Face                    |
| `huggingface-hub`    | Access to Hugging Face models and datasets                   |
| `pydantic`           | Data validation and settings management                      |
| `pydantic-settings`  | Environment-based configuration for Pydantic models          |
| `tqdm`               | Progress bars for file and batch processing                  |

## Usage

### Basic Usage

Pipeline can be configured via IngestionConfig class or environment variables.

### Configure from Code

```python
import asyncio
from ingest import DocumentIngestionPipeline, IngestionConfig, ChunkingStrategy

async def main():
    config = IngestionConfig(
        input_folder="./documents",
        api_endpoint="https://localhost:8000/ingest",
        chunking_strategy=ChunkingStrategy.FIXED_SIZE,
        chunk_size=512,
        chunk_overlap=50
    )
    
    pipeline = DocumentIngestionPipeline(config)
    await pipeline.run()

asyncio.run(main())
```

### Environment Variables

All configuration options can be set via environment variables with the `INGEST_` prefix. Below are examples for the most common settings:

```bash
export INGEST_INPUT_FOLDER="./my-documents"
export INGEST_FILE_EXTENSIONS='[".pdf", ".txt", ".json"]'
export INGEST_CHUNKING_STRATEGY="fixed_size"      # Options: fixed_size, semantic, sliding_window
export INGEST_CHUNK_SIZE="512"
export INGEST_CHUNK_OVERLAP="50"
export INGEST_API_ENDPOINT="http://localhost:8000/ingest"
export INGEST_API_TIMEOUT="30"
export INGEST_MAX_CONCURRENCY="5"
export INGEST_BATCH_SIZE="50"
export INGEST_MAX_RETRIES="5"
export INGEST_RETRY_DELAY="1.0"
```

- All variables are optional except `INGEST_API_ENDPOINT`, which should be set to your API URL.
- `INGEST_FILE_EXTENSIONS` expects a JSON array as a string (e.g., `'[".pdf", ".txt"]'`).
- Default values are used if environment variables are not set.
- See the configuration table below for all available options and their defaults.

See "Configuration Options" chapter for all description.

### Running the Script Directly

```bash
# Set required environment variable
export INGEST_API_ENDPOINT="https://your-api.com/ingest"

# Run the ingestion pipeline
uv run rag_solution/data_ingestion/ingest.py
```

### Example Usage Script

Run the example script to see different configurations in action:

```bash
uv run rag_solution/data_ingestion/example_usage.py
```

## Configuration Options

| Parameter | Environment Variable | Default | Description |
|-----------|---------------------|---------|-------------|
| `input_folder` | `INGEST_INPUT_FOLDER` | `"./documents"` | Folder containing documents to ingest |
| `file_extensions` | `INGEST_FILE_EXTENSIONS` | `[".pdf", ".txt"]` | File extensions to process |
| `chunking_strategy` | `INGEST_CHUNKING_STRATEGY` | `"fixed_size"` | Chunking strategy (`fixed_size`, `semantic`, `sliding_window`) |
| `chunk_size` | `INGEST_CHUNK_SIZE` | `1000` | Size of chunks (words for fixed-size, max words for others) |
| `chunk_overlap` | `INGEST_CHUNK_OVERLAP` | `200` | Overlap between chunks (words) |
| `api_endpoint` | `INGEST_API_ENDPOINT` | **Required** | API endpoint to send documents to |
| `api_timeout` | `INGEST_API_TIMEOUT` | `30` | API request timeout (seconds) |
| `max_concurrency` | `INGEST_MAX_CONCURRENCY` | `10` | Maximum concurrent API requests |
| `batch_size` | `INGEST_BATCH_SIZE` | `5` | Number of documents per API request |
| `max_retries` | `INGEST_MAX_RETRIES` | `5` | Maximum retries for non-429 errors |
| `retry_delay` | `INGEST_RETRY_DELAY` | `1.0` | Base delay between retries (seconds) |


## Chunking Strategies

### Fixed Size (`fixed_size`)
- Splits text into chunks of approximately equal character count
- Simple and predictable chunk sizes
- Good for uniform processing requirements
- Good for maximising context window

### Semantic (`semantic`)
- Splits text at sentence boundaries
- Maintains semantic coherence within chunks
- Better for natural language processing tasks
- Truncates to max chunk size, thus it might lose some information

### Sliding Window (`sliding_window`)
- Creates overlapping chunks with configurable step size
- Preserves context across chunk boundaries
- Useful for tasks requiring context preservation

### Not implemented but the cutting edge approach
#### Semantic distance
- Tries to create chunks with maxiumum neighbour pair-wise distance
- Theoretically it will create chunk which is consistent information content wise
- Experimental
- Ask me about this if you are interested

## Error Handling

- **Rate Limiting (429)**: Infinite retries with exponential backoff and `Retry-After` header support (which is populated by API)
- **Other Errors**: Configurable retries (default 5) with exponential backoff
- **Validation Errors**: Invalid chunks are logged and skipped
- **File Processing Errors**: Failed files are logged and processing continues
- **Network Errors**: Timeout and connection errors trigger retry logic
- **Progress Bars**: Progress bars are shown for each stage

## Testing

Run the comprehensive test suite:

```bash
# Install test dependencies
uv sync --all-group

# Run tests
python -m pytest tests/test_ingest.py -v
```

### Test Coverage

- Pydantic model validation
- All chunking strategies
- Document processing (text and PDF)
- API client with retry logic
- Pipeline integration
- Configuration management
- Error handling scenarios


## Performance Considerations

- **Concurrency**: Adjust `max_concurrency` based on API rate limits. API is rate limited to 4 QPS per IP by default.
- **Batch Size**: Larger batches reduce API calls but increase memory usage. API server scales to your CPU cores by default.
- **Chunk Size**: Smaller chunks increase granularity but create more API requests. Better in horizontal scale out scenario.
- **Overlap**: Larger overlaps preserve more context but increase data volume.

## Example Use Cases

`example_usage.py` contains some demos that you can try! Just comment out the right line in main.

1. **Large PDF dataset ingestion**: Downloads a bunch of PDFs from HuggingFace and ingests it
2. **Basic usage example with minimal configuration**: Simple TXT ingestion.
