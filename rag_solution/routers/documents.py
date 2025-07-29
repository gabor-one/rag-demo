from fastapi import APIRouter, Depends, status

from rag_solution.db import MilvusDB, get_db
from rag_solution.models.documents import (
    DeleteDocumentResponse,
    Document,
    DocumentListResponse,
    DocumentsIngestRequest,
    QueryRequest,
    QueryResponse,
)

router = APIRouter(tags=["Documents"])


@router.post(
    "/ingest",
    status_code=status.HTTP_200_OK,
    summary="Ingest documents into the vector database",
)
async def ingest_documents(
    request: DocumentsIngestRequest, db: MilvusDB = Depends(get_db)
):
    await db.insert_documents([doc.model_dump() for doc in request.documents])
    return


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query documents from the vector database using semantic search",
)
async def query_documents(request: QueryRequest, db: MilvusDB = Depends(get_db)):
    results = await db.hybrid_search(
        request.query,
        similarity_threshold=request.similarity_threshold,
        limit=min(request.limit or 20, 20),
        filter=request.filter_phrase,
    )
    # Return the text of the top results
    return QueryResponse(
        results=[Document(id=hit["pk"], text=hit["entity"]["text"], metadata=hit["entity"]["metadata"]) for hit in results]
    )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all documents in the vector database",
)
async def list_documents(db: MilvusDB = Depends(get_db)):
    # Attempt to list documents if metadata is stored in Milvus
    docs = await db.list_all_documents()

    return (
        DocumentListResponse(
            documents=[
                Document(id=doc["pk"], text=doc["text"], metadata=doc["metadata"])
                for doc in docs
            ]
        )
        if docs
        else DocumentListResponse(documents=[])
    )


@router.delete(
    "/documents/{id}",
    response_model=DeleteDocumentResponse,
    summary="Delete a document from the vector database by ID",
)
async def delete_document(id: int, db: MilvusDB = Depends(get_db)):
    # Attempt to delete document by id
    await db.delete_document_by_id([id])
    return DeleteDocumentResponse(id=id)


@router.delete(
    "/documents",
    status_code=status.HTTP_200_OK,
    summary="Delete all documents from the vector database",
)
async def delete_all_documents(db: MilvusDB = Depends(get_db)):
    await db.drop_all_documents()
