import asyncio
import concurrent
import concurrent.futures
import multiprocessing
import threading
from itertools import chain
from typing import Iterable, Literal

from loguru import logger
from pymilvus import model

from rag_solution.models.db import CreateDocument
from rag_solution.settings import settings

# Embedding functions
dense_embedding_function: model.dense.SentenceTransformerEmbeddingFunction | None = None

def _encode_dense(
    documents: list[CreateDocument], type: Literal["Document", "Query"]
) -> list[tuple[CreateDocument, list[float]]]:
    """
    Encodes a list of documents into dense vector embeddings.
    Args:
        documents (list[CreateDocument]): A list of documents to encode. Each document should have a "text" field.
        type (Literal["Document", "Query"]): Specifies whether to encode as "Document" or "Query".
    Returns:
        list[tuple[CreateDocument, list[float]]]: A list of tuples, each containing the original document and its corresponding embedding vector.
    Raises:
        ValueError: If the type is not "Document" or "Query".
    """
    global dense_embedding_function
    if type == "Document":
        embeddings = dense_embedding_function.encode_documents(
            [doc["text"] for doc in documents]
        )
    elif type == "Query":
        embeddings = dense_embedding_function.encode_queries(
            [doc["text"] for doc in documents]
        )
    else:
        raise ValueError("Type must be either 'Document' or 'Query'.")
    return list(zip(documents, embeddings))


async def encode_dense_parallel_executor(
    documents: list[CreateDocument], type: Literal["Document", "Query"]
) -> Iterable[tuple[CreateDocument, list[float]]]:
    """
    Asynchronously encodes a list of documents in parallel using a worker pool.
    Splits the input documents into chunks based on the executor pool size, and processes each chunk concurrently
    using the worker pool. The encoding is performed by the `_encode_dense` function, and the results are gathered
    and flattened into a single iterable of (document, embedding) tuples.
    Args:
        documents (list[CreateDocument]): The list of documents to encode.
        type (Literal["Document", "Query"]): The type of encoding to perform, either "Document" or "Query".
    Returns:
        Iterable[tuple[CreateDocument, list[float]]]: An iterable of tuples, each containing a document and its corresponding embedding vector.
    """
    loop = asyncio.get_running_loop()

    # Split documents into self.executor_pool_size chunks for parallel processing
    chunks = [
        documents[i :: SingletonWorkerPool.executor_pool_size()]
        for i in range(SingletonWorkerPool.executor_pool_size())
    ]
    chunks = [chunk for chunk in chunks if chunk]  # Remove empty chunks

    # Run encode_documents in parallel for each chunk
    encode_results = await asyncio.gather(
        *[
            loop.run_in_executor(
                SingletonWorkerPool.get_pool(), _encode_dense, chunk, type
            )
            for chunk in chunks
        ]
    )
    # Flatten the list of lists and return as an iterator
    return chain.from_iterable(encode_results)


def _init_worker():
    """
    Initializes the worker by loading the dense embedding function onto the device.
    Expected to be called by the worker processes.
    """
    global dense_embedding_function
    dense_embedding_function = model.dense.SentenceTransformerEmbeddingFunction(
        model_name="all-mpnet-base-v2",  # Good model with large context window
        # Uses most performant device available if None
        device="cpu",
    )


class SingletonWorkerPool:
    """
    Global worker pool for parallel processing.
    This is needed otherwise requests would compete for GPUs/CPU.
    Pass CUDA device address to init_worker if used with GPUs (not implemented).
    This essentially creates a queue for embedding requests while keeping necessary resources initialized.
    """

    _lock = threading.Lock()
    _pool: concurrent.futures.ProcessPoolExecutor | None = None

    # Limit max executor to 4 as dense embedding function model is fairly large
    # all-mpnet-base-v2: 420 MB
    _executor_pool_size: int | None = None

    @classmethod
    def executor_pool_size(cls) -> int:
        if cls._executor_pool_size is None:
            cls._executor_pool_size = settings.MAX_EXECUTOR_POOL_SIZE or (
                multiprocessing.cpu_count() - 1
            )
        return cls._executor_pool_size

    @classmethod
    def get_pool(cls) -> concurrent.futures.ProcessPoolExecutor:
        if cls._pool is None:
            with cls._lock:
                if cls._pool is None:
                    cls._pool = concurrent.futures.ProcessPoolExecutor(
                        max_workers=cls.executor_pool_size(), initializer=_init_worker
                    )
                    logger.info(
                        f"Initialized embedding worker pool with {cls.executor_pool_size()} workers."
                    )
        return cls._pool

    @classmethod
    def is_initialized(cls) -> bool:
        return cls._pool is not None

    @classmethod
    def shutdown(cls):
        if cls._pool is not None:
            with cls._lock:
                if cls._pool is not None:
                    cls._pool.shutdown(wait=True, cancel_futures=True)
                    cls._pool = None
                    logger.info("Shut down embedding worker pool.")
