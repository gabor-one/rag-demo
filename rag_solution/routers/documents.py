import datetime

from fastapi import APIRouter, Depends, HTTPException

from rag_solution.db import MilvusDB
from rag_solution.models.documents import (
    DeleteDocumentResponse,
    Document,
    DocumentListResponse,
    DocumentsIngestRequest,
    QueryRequest,
    QueryResponse,
)
from rag_solution.settings import settings

router = APIRouter(tags=["Documents"])


# Dependency injection for MilvusDB
def get_db():
    db = MilvusDB(settings.MILVUS_URI)
    return db


@router.post("/ingest", response_model=list[Document])
async def ingest_documents(
    request: DocumentsIngestRequest, db: MilvusDB = Depends(get_db)
):
    contents = [doc.text for doc in request.documents]
    await db.insert_documents(contents)
    now = datetime.datetime.utcnow().isoformat()
    # Milvus does not return IDs directly; this is a placeholder
    return [Document(id=fn, filename=fn, processed_at=now) for fn in filenames]


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest, db: MilvusDB = Depends(get_db)):
    results = await db.hybrid_search(
        request.query,
        similarity_threshold=request.similarity_threshold,
        limit=max(request.top_k or 20, 20),
    )
    # Return the text of the top results
    return QueryResponse(
        results=[Document(id=hit["id"], text=hit["entity"]["text"]) for hit in results]
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(db: MilvusDB = Depends(get_db)):
    # Attempt to list documents if metadata is stored in Milvus
    docs = await db.list_all_documents()

    return (
        DocumentListResponse(
            documents=[Document(id=doc["pk"], text=doc["text"]) for doc in docs]
        )
        if docs
        else DocumentListResponse(documents=[])
    )


@router.delete("/documents/{id}", response_model=DeleteDocumentResponse)
async def delete_document(id: int, db: MilvusDB = Depends(get_db)):
    # Attempt to delete document by id if supported
    await db.delete_document_by_id([id])
    return DeleteDocumentResponse(id=id)
