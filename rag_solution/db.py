import os

from loguru import logger
from pymilvus import (
    AnnSearchRequest,
    AsyncMilvusClient,
    CollectionSchema,
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    MilvusException,
    WeightedRanker,
)

from rag_solution.models.db import CreateDocument, ResultDocument, SearchResult
from rag_solution.settings import settings
from rag_solution.singleton_worker_pool import encode_dense_parallel_executor

# Constants
COLLECTION_NAME = "hybrid_search_collection"


class MilvusDB:
    """
    Asynchronous client wrapper for managing a Milvus vector database.
    This class provides high-performance async operations and a managed sync client pool for interacting with Milvus.
    It supports collection creation, document insertion, hybrid search (dense + sparse), and collection management.
    Args:
        uri (str): The URI of the Milvus server.
    Attributes:
        uri (str): The URI of the Milvus server.
        async_client (AsyncMilvusClient | None): The asynchronous Milvus client instance.
    Methods:
        connect():
            Asynchronously connects to Milvus, initializes clients, and creates the collection if it does not exist.
        disconnect():
            Asynchronously disconnects from Milvus and closes clients.
        create_collection():
            Asynchronously creates a new collection with the required schema and indexes.
        hybrid_search(query, sparse_weight=0.7, dense_weight=1.0, similarity_threshold=0.6, filter=None, limit=10):
            Performs a hybrid search using both sparse and dense embeddings, returning ranked results.
        insert_documents(documents):
            Asynchronously inserts documents into the collection after encoding their dense embeddings.
        delete_document_by_id(doc_ids):
            Asynchronously deletes documents from the collection by their primary key IDs.
        list_all_documents():
            Asynchronously retrieves all documents from the collection.
        drop_all_documents():
            Drops the entire collection and recreates it.
        is_connection_ready():
            Checks if the collection exists and the connection is ready.
    Raises:
        ValueError: If attempting to create a collection that already exists or drop a non-existent collection.
    """

    def __init__(self, uri: str):
        self.uri: str = uri
        self.async_client: AsyncMilvusClient | None = None

    async def connect(self):
        logger.info(f"Connecting to Milvus at {self.uri}...")
        # Create parent directories if using local milvus lite.
        if self.uri.endswith(".db"):
            os.makedirs(os.path.dirname(self.uri), exist_ok=True)

        self.async_client = AsyncMilvusClient(
            uri=self.uri,
        )

        if not await self.try_load_collection():
            await self.create_collection()

        logger.info(f"Connected to Milvus at {self.uri}")

    async def try_load_collection(self) -> bool:
        """
        Check if the collection exists in the Milvus database, load it if it does.
        Returns:
            bool: True if the collection exists and loaded, False otherwise.
        """
        try:
            await self.async_client.load_collection(COLLECTION_NAME)
            return True
        except MilvusException as e:
            if e.code == 100:
                "collection not found error. It is ok."
                pass
            else:
                logger.exception(f"Failed to load collection {COLLECTION_NAME}")
        return False

    async def disconnect(self):
        await self.async_client.close()
        logger.info(f"Disconnected from Milvus at {self.uri}")

    async def create_collection(self):
        if await self.try_load_collection():
            raise ValueError("Collection already exists.")

        fields = [
            # Use auto generated id as primary key
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            # Set for all-mpnet-base-v2
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=384,
                enable_analyzer=True,
            ),
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

        await self.async_client.create_collection(
            COLLECTION_NAME,
            schema=schema,
            consistency_level="Strong",
            index_params=index_params
        )

        await self.async_client.load_collection(COLLECTION_NAME)
        logger.info(f"Created collection {COLLECTION_NAME} with schema: {schema}")

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

        encode_results = await encode_dense_parallel_executor(
            documents=[CreateDocument(text=query)], type="Query"
        )
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

        encode_results = await encode_dense_parallel_executor(
            documents=documents, type="Document"
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
        logger.info(
            f"Deleted documents with IDs {doc_ids} from collection {COLLECTION_NAME}."
        )

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
        if await self.try_load_collection():
            await self.async_client.drop_collection(COLLECTION_NAME)
            await self.create_collection()
        else:
            raise ValueError("Collection does not exist.")

    async def is_connection_ready(self) -> bool:
        return await self.try_load_collection()


# Dependency inject Milvus endpoints
async def get_db():
    db = MilvusDB(settings.MILVUSDB_URI)
    await db.connect()
    return db
