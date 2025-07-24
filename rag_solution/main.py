import uvicorn
from fastapi import FastAPI
from .routers.health import router as health_router
from .routers.documents import router as documents_router

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

app = FastAPI(
    version="0.1.0",
    title="RAG Solution Homework",
    description="RAG Solution Homework for 21.co",
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(health_router)
app.include_router(documents_router)




# Main guard


def dev():
    uvicorn.run("src.main:app", reload=True)


def serve():
    uvicorn.run("src.main:app", reload=False)


if __name__ == "__main__":
    dev()
