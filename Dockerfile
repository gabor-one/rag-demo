# Most of this file was copied from https://github.com/astral-sh/uv-docker-example/blob/main/multistage.Dockerfile

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Disable Python downloads, because we want to use the system interpreter
# across both images. If using a managed Python version, it needs to be
# copied from the build image into the final image; see `standalone.Dockerfile`
# for an example.
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app
RUN --mount=type=cache,target=/root/.cache/uv,z \
    --mount=type=bind,source=uv.lock,target=uv.lock,z \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml,z \
    uv sync --locked --no-install-project --no-dev
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv,z \
    uv sync --locked --no-dev


## Production image    
# Use a final image without uv and other build dependencies.
FROM python:3.12-slim-bookworm

RUN useradd app
WORKDIR /app

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV SENTENCE_TRANSFORMER_CACHE_DIR="/app/models"
# This should let the container run without external DB.
# Podman uses networking on Mac to connect volumes which doens't work with MMAP.
# Probably it would work on Linux (or with native containerziation in next MacOS).
# TLDR: You can't run this container without external DB on Mac with Podman. Use `podman compose`.
ENV MILVUSDB_URI="/app/data/milvus.db"

# Copy the application from the builder
COPY --from=builder --chown=app:app /app /app

# This fetches the SentenceTransformer model and caches it in the specified directory
# Faster initialization, less external dependencies in production
# Would be desirable to run this before COPY step in a real world scenario,
#  so we don't have to copy the model on each code change + save some space in the container registry.
RUN mkdir -p /app/models \
    && chown app:app /app/models \
    && mkdir -p /app/data \
    && chown app:app /app/data \
    && /app/scripts/fetch_sentence_transformer_model.sh

# Don't try to download models form the internet in production
ENV SENTENCE_TRANSFORMER_LOCAL_FILES_ONLY="true"

USER app
EXPOSE 8000

# Run the FastAPI application by default
CMD ["fastapi", "run", "--host", "0.0.0.0", "--port", "8000", "rag_solution/main.py"]