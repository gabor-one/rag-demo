import os

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from rag_solution.main import app
from rag_solution.settings import settings

ASYNCIO_SCOPE = "module"

@pytest_asyncio.fixture(scope="module", loop_scope=ASYNCIO_SCOPE)
async def testclient():
    # Use a local file for testing
    milvus_path = "./data/integration_test_milvus.db"
    settings.MILVUS_URI = milvus_path
    milvus_dir = os.path.dirname(milvus_path)

    # Ensure directory exists
    os.makedirs(milvus_dir, exist_ok=True)

    # Delete file if exists
    if os.path.exists(milvus_path):
        os.remove(milvus_path)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as async_client:
        yield async_client

@pytest.mark.asyncio(loop_scope=ASYNCIO_SCOPE)
@pytest.mark.dependency()
async def test_ingest_documents_success(testclient: AsyncClient):
    data = {
        "documents": [
            {"text": "Iphone 16 pro", "metadata": {"type": "phone"}},
            {"text": "Toyota Corolla", "metadata": {"type": "car"}},
        ]
    }
    response = await testclient.post("/ingest", json=data)
    assert response.status_code == 201


@pytest.mark.asyncio(loop_scope=ASYNCIO_SCOPE)
@pytest.mark.dependency(depends=["test_ingest_documents_success"])
async def test_query_documents_success(testclient: AsyncClient):
    data = {
        "query": "phone",
        "limit": 10,
        "similarity_threshold": None,
        "filter_phrase": None,
    }
    response = await testclient.post("/query", json=data)
    assert response.status_code == 200
    assert response.json()["results"][0]["text"] == "Iphone 16 pro"

@pytest.mark.asyncio(loop_scope=ASYNCIO_SCOPE)
@pytest.mark.dependency(depends=["test_ingest_documents_success"])
async def test_list_documents_success(testclient: AsyncClient):
    response = await testclient.get("/documents")
    assert response.status_code == 200

    documents = response.json()["documents"]
    assert len(documents) == 2  # We ingested 2 documents
    ingested_texts = ["Iphone 16 pro", "Toyota Corolla"]
    assert documents[0]["text"] in ingested_texts
    assert documents[1]["text"] in ingested_texts

@pytest.mark.asyncio(loop_scope=ASYNCIO_SCOPE)
@pytest.mark.dependency(depends=["test_ingest_documents_success"])
async def test_metadata_query_documents_success(testclient: AsyncClient):
    data = {
        "query": "phone",
        "limit": 10,
        "similarity_threshold": None,
        "filter_phrase": "metadata['type'] == 'phone'",
    }
    response = await testclient.post("/query", json=data)
    assert response.status_code == 200
    assert len(response.json()["results"]) == 1
    assert response.json()["results"][0]["text"] == "Iphone 16 pro"


@pytest.mark.asyncio(loop_scope=ASYNCIO_SCOPE)
@pytest.mark.dependency(depends=[
    "test_metadata_query_documents_success",
    "test_list_documents_success",
    "test_query_documents_success"
    ])
async def test_delete_document_success(testclient: AsyncClient):
    response = await testclient.get("/documents")
    assert response.status_code == 200
    documents = response.json()["documents"]
    assert len(documents) == 2  # We ingested 2 documents

    iphone_doc_id = next(
        doc["id"] for doc in documents if doc["text"] == "Iphone 16 pro"
    )

    response = await testclient.delete(f"/documents/{iphone_doc_id}")
    assert response.status_code == 200

    response = await testclient.get("/documents")
    assert response.status_code == 200
    documents = response.json()["documents"]
    assert len(documents) == 1  # We should have only document left after deletion
    assert documents[0]["text"] == "Toyota Corolla"
