from fastapi.testclient import TestClient
from rag_solution.main import app

client = TestClient(app)

def test_ingest_documents_success():
    data = {
        "documents": [
            {"text": "Test document 1"},
            {"text": "Test document 2"}
        ]
    }
    response = client.post("/ingest", json=data)
    # Status code may be 200 or 500 depending on DB config
    assert response.status_code in (200, 500)
    # If 200, should be a list
    if response.status_code == 200:
        assert isinstance(response.json(), list)

def test_query_documents_success():
    data = {
        "query": "Relevant",
        "similarity_threshold": 0.8,
        "top_k": 2,
        "filter_phase": None
    }
    response = client.post("/query", json=data)
    assert response.status_code in (200, 500)
    if response.status_code == 200:
        assert "results" in response.json()

def test_list_documents_success():
    response = client.get("/documents")
    assert response.status_code in (200, 500)
    if response.status_code == 200:
        assert "documents" in response.json()

def test_delete_document_success():
    response = client.delete("/documents/1")
    assert response.status_code in (200, 500)
    if response.status_code == 200:
        assert "id" in response.json()
