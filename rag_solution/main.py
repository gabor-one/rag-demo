from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI

from rag_solution.singleton_worker_pool import SingletonWorkerPool
from .routers.health import router as health_router
from .routers.documents import router as documents_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure worker pool is initialized
    # First call still be slower as initalizer runs on the first submit
    SingletonWorkerPool.get_pool() 
    yield
    SingletonWorkerPool.shutdown()

app = FastAPI(
    # Version should come from some CI automation...
    #   like semantic-release, GitVersion, and hatch-vcs building __version__.py
    version="0.1.0",
    title="RAG Solution Homework",
    description="RAG Solution Homework for 21.co",
    lifespan=lifespan
)


app.include_router(health_router)
app.include_router(documents_router)


# Main guard
def dev():
    uvicorn.run("rag_solution.main:app", reload=True, log_level="trace")

# Production server
#   Use this for production deployments, e.g. with Docker or systemd
#   It will not reload on code changes, which is more efficient for production use.
#   This accepts all traffic
def serve():
    uvicorn.run("rag_solution.main:app", reload=False, host="0.0.0.0", log_level="info")


if __name__ == "__main__":
    dev()
