from fastapi import APIRouter, Depends, Response
from fastapi.responses import JSONResponse
from loguru import logger

from rag_solution.db import MilvusDB, SingletonWorkerPool, get_db
from rag_solution.models.health import HealthCheckResponse, StatusEnum

# Normally i'd hide this
#  For the sake of this excersie it show.
router = APIRouter(tags=["Health"], include_in_schema=True)


@router.get("/health", response_model=HealthCheckResponse, status_code=200, summary="Health check endpoint")
async def health(response: Response, db: MilvusDB = Depends(get_db)):
    db_connection_ready = False
    try:
        db_connection_ready = db.is_connection_ready()
    except Exception:
        logger.exception("Health check failed")

    worker_pool_initialized = SingletonWorkerPool.is_initialized()

    responseBody = HealthCheckResponse(
        db=StatusEnum.from_bool(db_connection_ready),
        worker_pool=StatusEnum.from_bool(worker_pool_initialized),
    )

    response.status_code = responseBody.get_response_code()
    return responseBody
