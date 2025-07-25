import asyncio
import concurrent.futures
from itertools import chain
import threading
from typing import Iterable, Literal, TypedDict
import multiprocessing
import concurrent
from pymilvus import (
    AnnSearchRequest,
    AsyncMilvusClient,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    WeightedRanker,
    connections,
    model,
    utility,
)
from loguru import logger
from rag_solution.settings import settings

# Constants
COLLECTION_NAME = "hybrid_search_collection"


# Models
class CreateDocument(TypedDict):
    text: str
    metadata: dict[str, str] | None


class ResultDocument(TypedDict):
    text: str
    metadata: dict[str, str] | None
    pk: int


class SearchResult(TypedDict):
    id: int
    distance: float
    entity: ResultDocument


# Embedding functions
dense_embedding_function: model.dense.SentenceTransformerEmbeddingFunction | None = None


def init_worker():
    global dense_embedding_function
    dense_embedding_function = model.dense.SentenceTransformerEmbeddingFunction(
        model_name="all-mpnet-base-v2",  # Good model with large context window
        # Uses most performant device available if None
        device="cpu",
    )

# Global worker pool for parallel processing
#  In case of GPUs: one worker per GPU
#  This is needed otherwise requests would compete for GPUs/CPU (pass CUDA device address to init_worker)
#  This essentially creates a queue for embedding requests while keeping necessary resources initialized.
class SingletonWorkerPool:
    _lock = threading.Lock()
    _pool: concurrent.futures.ProcessPoolExecutor | None = None

    # Limit max executor to 4 as dense embedding function model is fairly large
    # all-mpnet-base-v2: 420 MB
    _executor_pool_size: int | None = None

    @classmethod
    def executor_pool_size(cls) -> int:
        if cls._executor_pool_size is None:
            cls._executor_pool_size = settings.MAX_EXECUTOR_POOL_SIZE or (multiprocessing.cpu_count() - 1)
        return cls._executor_pool_size

    @classmethod
    def get_pool(cls) -> concurrent.futures.ProcessPoolExecutor:
        if cls._pool is None:
            with cls._lock:
                if cls._pool is None:
                    cls._pool = concurrent.futures.ProcessPoolExecutor(
                        max_workers=cls.executor_pool_size(),
                        initializer=init_worker
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

def encode_dense(
    documents: list[CreateDocument], type: Literal["Document", "Query"]
) -> list[tuple[CreateDocument, list[float]]]:
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


# MilvusDB
class MilvusDB:
    def __init__(self, uri: str):
        self.uri: str = uri
        self.async_client: AsyncMilvusClient | None = None

    async def connect(self):
        # High Performance Astnc client
        self.async_client = AsyncMilvusClient(
            uri=self.uri,
        )
        # Low performance High comfort managed sync client pool
        connections.connect(
            alias="default",
            uri=self.uri,
        )

        if not utility.has_collection(COLLECTION_NAME):
            await self.create_collection()

        logger.info(f"Connected to Milvus at {self.uri}")

    async def disconnect(self):
        await self.async_client.close()
        connections.disconnect("default")
        logger.info(f"Disconnected from Milvus at {self.uri}")

    async def create_collection(self):
        if utility.has_collection(COLLECTION_NAME):
            raise ValueError("Collection already exists.")

        fields = [
            # Use auto generated id as primary key
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            # Set for all-mpnet-base-v2
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=384),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            # Set for all-mpnet-base-v2
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        schema = CollectionSchema(fields, enable_dynamic_field=True)

        # This will automaticallty create sparse embedding when writing to the collection
        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["sparse_vector"],
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)

        await self.async_client.create_collection(
            COLLECTION_NAME, schema=schema, consistency_level="Strong"
        )

        index_params = self.async_client.prepare_index_params()
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            index_name="sparse_vector_index",
            metric_type="BM25",
        )
        index_params.add_index(
            field_name="dense_vector", index_type="FLAT", metric_type="COSINE"
        )
        await self.async_client.create_index(
            COLLECTION_NAME,
            index_params,
        )

        await self.async_client.load_collection(COLLECTION_NAME)
        logger.info(f"Created collection {COLLECTION_NAME} with schema: {schema}")

    async def encode_dense_parallel_executor(
        self, documents: list[CreateDocument], type: Literal["Document", "Query"]
    ) -> Iterable[tuple[CreateDocument, list[float]]]:
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
                loop.run_in_executor(SingletonWorkerPool.get_pool(), encode_dense, chunk, type)
                for chunk in chunks
            ]
        )
        # Flatten the list of lists and return as an iterator
        return chain.from_iterable(encode_results)

    async def hybrid_search(
        self,
        query: str,
        sparse_weight: float = 0.7,
        dense_weight: float = 1.0,
        similarity_threshold: float | None = 0.6,
        filter: str | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        assert self.async_client is not None, "DB is not initialized."

        encode_results = await self.encode_dense_parallel_executor([CreateDocument(text=query)], "Query")
        query_dense_embedding = list(encode_results)[0][
            1
        ]  # Get the first (and only) query embedding

        dense_req = AnnSearchRequest(
            data=[query_dense_embedding],
            anns_field="dense_vector",
            param={},
            limit=limit,
            expr=filter,
        )
        sparse_req = AnnSearchRequest(
            data=[query],
            anns_field="sparse_vector",
            param={},
            limit=limit,
            expr=filter,
        )
        ranker = WeightedRanker(sparse_weight, dense_weight)
        results = (
            await self.async_client.hybrid_search(
                COLLECTION_NAME,
                [sparse_req, dense_req],
                ranker=ranker,
                limit=limit,
                output_fields=["text", "pk", "metadata"],
            )
        )[0]

        logger.info(
            f"Hybrid search for query '{query}' returned {len(results)} results."
        )

        if similarity_threshold is not None:
            # Filter results based on the similarity threshold
            results = [
                result
                for result in results
                if result["distance"] >= similarity_threshold
            ]

            logger.info(
                f"Filtered results to {len(results)} based on similarity threshold {similarity_threshold}."
            )

        return results

    async def insert_documents(self, documents: list[CreateDocument]):
        assert self.async_client is not None, "DB is not initialized."

        encode_results = await self.encode_dense_parallel_executor(
            documents, "Document"
        )

        document_dense_embeddings = [
            {
                "text": document["text"],
                "dense_vector": embedding,
                "metadata": document["metadata"] if "metadata" in document else {},
            }
            for (document, embedding) in encode_results
        ]

        await self.async_client.insert(COLLECTION_NAME, document_dense_embeddings)
        logger.info(
            f"Inserted {len(document_dense_embeddings)} documents into collection {COLLECTION_NAME}."
        )

    async def delete_document_by_id(self, doc_ids: list[int]):
        assert self.async_client is not None, "DB is not initialized."
        await self.async_client.delete(COLLECTION_NAME, ids=doc_ids)
        logger.info(f"Deleted documents with IDs {doc_ids} from collection {COLLECTION_NAME}.")

    async def list_all_documents(self) -> list[ResultDocument]:
        assert self.async_client is not None, "DB is not initialized."
        results: list[ResultDocument] = []
        offset = 0
        limit = 100
        while True:
            res = await self.async_client.query(
                COLLECTION_NAME,
                offset=offset,
                limit=limit,
                output_fields=["text", "metadata"],
            )

            if not res:
                break

            results.extend(res)
            if len(res) < limit:
                break
            offset += limit
        return results

    async def drop_all_documents(self):
        assert self.async_client is not None, "DB is not initialized."
        if utility.has_collection(COLLECTION_NAME):
            await self.async_client.drop_collection(COLLECTION_NAME)
            await self.create_collection()
        else:
            raise ValueError("Collection does not exist.")
        
    def is_connection_ready(self) -> bool:
        return utility.has_collection(COLLECTION_NAME)


# Dependency injection for MilvusDB
async def get_db():
    db = MilvusDB(settings.MILVUS_URI)
    await db.connect()
    return db