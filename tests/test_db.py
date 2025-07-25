from typing import AsyncGenerator

import pytest
import pytest_asyncio

from rag_solution.db import CreateDocument, MilvusDB
import os

COLLECTION_NAME = "hybrid_search_collection"
DB_FILE_PATH = "./data/unittest_milvus.db"


# Single fixture for the session in session scope
@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def milvusdb() -> AsyncGenerator[MilvusDB, None]:
    """Fixture to create a MilvusDB instance for testing."""

    if os.path.exists(DB_FILE_PATH):
        os.remove(DB_FILE_PATH)

    lockfile = f"{DB_FILE_PATH}.lock"
    if os.path.exists(lockfile):
        os.remove(lockfile)

    os.makedirs(os.path.dirname(DB_FILE_PATH), exist_ok=True)

    db = MilvusDB(uri=DB_FILE_PATH)
    await db.connect()

    yield db

    await db.disconnect()

# Clean up the database file before each test, operate is session scope
@pytest_asyncio.fixture(scope="function", loop_scope="session", autouse=True)
async def clean_db(milvusdb: MilvusDB):
    """Fixture to clean up the database after tests."""
    # Clean up the collection after tests
    await milvusdb.drop_all_documents()

@pytest.mark.asyncio(loop_scope="session")
async def test_insert_search(milvusdb: MilvusDB, clean_db):
    await milvusdb.insert_documents(
        [CreateDocument(text="test doc"), CreateDocument(text="another doc")]
    )
    results = await milvusdb.hybrid_search("doc")

    assert len(results) == 2
    assert all("text" in res["entity"] for res in results)
    assert (
        results[0]["entity"]["text"] == "another doc"
        and results[1]["entity"]["text"] == "test doc"
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_insert_with_metadata_search(milvusdb: MilvusDB):
    await milvusdb.insert_documents(
        [
            CreateDocument(text="test doc", metadata={"group": "value"}),
            CreateDocument(text="another doc", metadata={"group": "value2"}),
        ]
    )
    results = await milvusdb.hybrid_search("doc")

    assert len(results) == 2
    assert all("text" in res["entity"] for res in results)
    assert all("metadata" in res["entity"] for res in results)
    assert (
        results[0]["entity"]["text"] == "another doc"
        and results[1]["entity"]["text"] == "test doc"
    )
    assert (
        results[0]["entity"]["metadata"]["group"] == "value2"
        and results[1]["entity"]["metadata"]["group"] == "value"
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_insert_with_metadata_filter_search(milvusdb: MilvusDB):
    await milvusdb.insert_documents(
        [
            CreateDocument(text="test doc", metadata={"group": "value"}),
            CreateDocument(text="another doc", metadata={"group": "value2"}),
        ]
    )
    results = await milvusdb.hybrid_search("doc", filter='metadata["group"] == "value"')

    assert len(results) == 1
    assert all("text" in res["entity"] for res in results)
    assert all("metadata" in res["entity"] for res in results)
    assert results[0]["entity"]["text"] == "test doc"
    assert results[0]["entity"]["metadata"]["group"] == "value"


@pytest.mark.asyncio(loop_scope="session")
async def test_sparse_embedding_only(milvusdb: MilvusDB):
    await milvusdb.insert_documents(
        [CreateDocument(text="sparse test"), CreateDocument(text="irrelevant")]
    )
    # Perform hybrid search with dense_weight=0 (sparse only)
    results = await milvusdb.hybrid_search(
        "sparse", dense_weight=0.0, sparse_weight=1.0, similarity_threshold=None
    )
    assert len(results) >= 1
    # The top result should be the document with the word match
    assert results[0]["entity"]["text"] == "sparse test"


@pytest.mark.asyncio(loop_scope="session")
async def test_dense_embedding_only(milvusdb: MilvusDB):
    await milvusdb.insert_documents(
        [CreateDocument(text="Iphone 16 pro"), CreateDocument(text="Toyota Corolla")]
    )
    # Perform hybrid search with sparse_weight=0 (dense only)
    results = await milvusdb.hybrid_search(
        "Phone", dense_weight=1.0, sparse_weight=0.0, similarity_threshold=None
    )
    assert len(results) >= 1
    # The top result should be the document with the closest dense embedding
    assert results[0]["entity"]["text"] == "Iphone 16 pro"


@pytest.mark.asyncio(loop_scope="session")
async def test_dense_similarity_threshold(milvusdb: MilvusDB):
    await milvusdb.insert_documents(
        [CreateDocument(text="Iphone 16 pro"), CreateDocument(text="Toyota Corolla")]
    )
    # Perform hybrid search with sparse_weight=0 (dense only)
    results = await milvusdb.hybrid_search(
        "Phone", dense_weight=1.0, sparse_weight=0.0, similarity_threshold=0.7
    )
    # There should be only one result since the threshold is high
    assert len(results) == 1
    # The top result should be the document with the closest dense embedding
    assert results[0]["entity"]["text"] == "Iphone 16 pro"


@pytest.mark.asyncio(loop_scope="session")
async def test_parallel_insert_many_documents(milvusdb: MilvusDB):
    # Insert a large number of documents in parallel
    num_docs = 100
    docs = [CreateDocument(text=f"Document {i}") for i in range(num_docs)]
    await milvusdb.insert_documents(docs)
    # Search for a common word to retrieve all documents
    results = await milvusdb.hybrid_search("Document", limit=num_docs)
    assert len(results) == num_docs
    texts = [res["entity"]["text"] for res in results]
    for i in range(num_docs):
        assert f"Document {i}" in texts


@pytest.mark.asyncio(loop_scope="session")
async def test_delete_document_by_id(milvusdb: MilvusDB):
    # Insert documents
    await milvusdb.insert_documents(
        [CreateDocument(text="delete me"), CreateDocument(text="keep me")]
    )
    # List all documents to get their IDs
    all_docs = await milvusdb.list_all_documents()
    assert len(all_docs) == 2
    # Search to get the internal IDs
    results = await milvusdb.hybrid_search("delete me", limit=2)
    # The result should contain the document to delete
    doc_id_to_delete = results[0]["id"]
    # Delete the document by ID
    await milvusdb.delete_document_by_id([doc_id_to_delete])
    # List all documents again
    remaining_docs = await milvusdb.list_all_documents()
    texts = [doc["text"] for doc in remaining_docs]
    assert "delete me" not in texts
    assert "keep me" in texts
    assert len(remaining_docs) == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_list_all_documents(milvusdb: MilvusDB):
    docs = [
        CreateDocument(text="doc1"),
        CreateDocument(text="doc2"),
        CreateDocument(text="doc3"),
    ]
    await milvusdb.insert_documents(docs)
    all_docs = await milvusdb.list_all_documents()
    texts = [doc["text"] for doc in all_docs]
    for d in docs:
        assert d["text"] in texts
    assert len(all_docs) == len(docs)


@pytest.mark.asyncio(loop_scope="session")
async def test_insert_and_list_many_documents(milvusdb: MilvusDB):
    # Insert 1015 documents
    num_docs = 1015
    docs = [CreateDocument(text=f"Bulk Document {i}") for i in range(num_docs)]
    await milvusdb.insert_documents(docs)
    # List all documents and check that all are present
    all_docs = await milvusdb.list_all_documents()
    texts = [doc["text"] for doc in all_docs]
    assert len(all_docs) == num_docs
    for i in range(num_docs):
        assert f"Bulk Document {i}" in texts


@pytest.mark.asyncio(loop_scope="session")
async def test_drop_all_documents(milvusdb: MilvusDB):
    # Insert several documents
    docs = [
        CreateDocument(text="to be deleted 1"),
        CreateDocument(text="to be deleted 2"),
        CreateDocument(text="to be deleted 3"),
    ]
    await milvusdb.insert_documents(docs)
    # Ensure documents are inserted
    all_docs = await milvusdb.list_all_documents()
    assert len(all_docs) == len(docs)
    # Drop all documents
    await milvusdb.drop_all_documents()
    # List all documents again, should be empty
    remaining_docs = await milvusdb.list_all_documents()
    assert remaining_docs == []
