import asyncio
import concurrent.futures
from itertools import chain
from typing import Iterable, Iterator, Literal, NamedTuple, TypedDict
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

# Contsants
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

        # Limit max executor to 4 as dense embedding function model is large
        # all-mpnet-base-v2: 420 MB
        self.executor_pool_size = min(multiprocessing.cpu_count() - 1, 4)
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.executor_pool_size, initializer=init_worker
        )

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

    async def disconnect(self):
        await self.async_client.close()
        connections.disconnect("default")

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

    async def encode_dense_parallel_executor(
        self, documents: list[CreateDocument], type: Literal["Document", "Query"]
    ) -> Iterable[tuple[CreateDocument, list[float]]]:
        loop = asyncio.get_running_loop()

        # Split documents into self.executor_pool_size chunks for parallel processing
        chunks = [
            documents[i :: self.executor_pool_size]
            for i in range(self.executor_pool_size)
        ]

        # Run encode_documents in parallel for each chunk
        encode_results = await asyncio.gather(
            *[
                loop.run_in_executor(self.executor, encode_dense, chunk, type)
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
        )
        sparse_req = AnnSearchRequest(
            data=[query],
            anns_field="sparse_vector",
            param={},
            limit=limit,
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

        if similarity_threshold is not None:
            # Filter results based on the similarity threshold
            results = [
                result
                for result in results
                if result["distance"] >= similarity_threshold
            ]

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
                #"metadata": document["metadata"] if "metadata" in document else {},
            }
            for (document, embedding) in encode_results
        ]

        await self.async_client.insert(COLLECTION_NAME, document_dense_embeddings)

    async def delete_document_by_id(self, doc_ids: list[int]):
        assert self.async_client is not None, "DB is not initialized."
        await self.async_client.delete(COLLECTION_NAME, ids=doc_ids)

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
                output_fields=["text"],
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
