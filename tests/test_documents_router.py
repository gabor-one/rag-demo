import os

import pytest
import pytest_asyncio

from rag_solution.settings import settings
from rag_solution.main import app
from httpx import ASGITransport, AsyncClient


@pytest_asyncio.fixture(scope="session")
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

@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.dependency()
async def test_ingest_query_documents_success(testclient: AsyncClient):
    data = {"documents": [{"text": "Iphone 16 pro"}, {"text": "Toyota Corolla"}]}
    response = await testclient.post("/ingest", json=data)
    assert response.status_code == 201

    data = {
        "query": "phone",
        "limit": 10,
        "similarity_threshold": None,
        "filter_phrase": None,
    }
    response = await testclient.post("/query", json=data)
    assert response.status_code == 200
    assert response.json()["results"][0]["text"] == "Iphone 16 pro"

@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.dependency(depends=["test_ingest_query_documents_success"])
async def test_delete_list_documents_success(testclient: AsyncClient):
    response = await testclient.get("/documents")
    assert response.status_code == 200

    documents = response.json()["documents"]
    assert len(documents) == 2  # We ingested 2 documents
    ingested_texts = ["Iphone 16 pro", "Toyota Corolla"]
    assert documents[0]["text"] in ingested_texts
    assert documents[1]["text"] in ingested_texts

@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.dependency(depends=["test_delete_list_documents_success"])
async def test_delete_document_success(testclient: AsyncClient):
    response = await testclient.get("/documents")
    assert response.status_code == 200
    documents = response.json()["documents"]
    assert len(documents) == 2  # We ingested 2 documents

    iphone_doc_id = next(doc["id"] for doc in documents if doc["text"] == "Iphone 16 pro")

    response = await testclient.delete(f"/documents/{iphone_doc_id}")
    assert response.status_code == 200
    
    response = await testclient.get("/documents")
    assert response.status_code == 200
    documents = response.json()["documents"]
    assert len(documents) == 1  # We should have only document left after deletion
    assert documents[0]["text"] == "Toyota Corolla"
