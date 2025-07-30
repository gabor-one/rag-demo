# RAG Solution

This is a RAG API solution with IaC and a data ingestion pipeline.

# Overciew of the contents of this repository

* **rag_solution**: Python sources for the RAG Web API
* **rag_solution/data_ingestion**: Python sources for Data Ingestion script. For the sake of simplicity these two are one python project. 
  * Check `data_ingestion/README.md` for data pipeline information.
* **deploy**: Demo Pulumi IaC code for deploying the webservice to GCP. 
  * Check `deploy/README.md` for IaC specific information

# Prerequisites

To run this project the you are expected to have installed...

* **UV**: Python project and dependcy manager
* **Podman**: Container engine
  * Docker might fail due to Selinux (",Z") options. 

This project was tested on `MacOS`.

# Installation

Just use UV to install the right python version, create a `.venv` and download the dependencies.

```sh
uv sync --all-groups
```

Each sub-project has it's own dependecy group. You can install just what you are interested in: `iac`, `dev`, `pipeline`.

Don't forget to activate your new virtual environment.

```sh
source .venv/bin/activate
```

# Running the project

## Webserver (host + Embedded vector DB)

Run the webserver with FastAPI utility in development mode. First startup will take sometime as it needs to download the embedding model.

```sh
fastapi dev rag_solution/main.py
```

## Webserver (Multi container cluster)

Use compose file to bring up the two container cluster: RAG webserver and Milvus Standalone.

```sh
podman compose up --build
```

NOTE: This is for demo purposes only. Vector db (more like the underlying storage) is particulalry slow with mounted storage on non-Linux containerized environments.

## Data ignestion example

Start the webserver and run the data ingestion example with `uv`. It will download some tar-d PDFs from HuggingFace and ingest it into the RAG service.

```sh
uv run rag_solution/data_ingestion/example_usage.py 
```

Check `rag_solution/data_ingestion/example_usage.py ` for various demos.

# Running the test

Make sure your `Podman` is running.

```sh
pytest
```

# Project Documentation

## API

After bringing up the server open `http://127.0.0.1:8000/docs` to access the OpenAPI documentation page. Each endpoint and schema is documented thoroughly. Test the API.

## Dockerfile

The project uses a multistage dockerfile to build the project then copy all the artifacts into a slim, build-resource-less container. Also it prefetches embedding model and saves it into the container image, which speeds up the container startup. 

### Workarounds

* Embedder model prefetch is a hack. It should be done properly without actually running the model.

## Docker Compose

Docker compose file is better way to test the web service in close to real setup (aka integration testing). It build the container with `Dockerfile`, spins up a Milvus DB standalone instance with volume mount at `/data` and expose the API on port `8000`.

## RAG web service

Web service is built on FastAPI. It uses Pydantic to validate incoming requests and data and it uses Pydantic-settings to pick up configuration from `.env` file and environment variables. The endpoints are implemented as async functions. The web server part doesn't scale above the single python process with `uvicron`. In case of scalability is needed I'd use multiple instances (e.g.: Kubernetes Horizontal Pod Autscaler) instead of going for larger hosts with something like `gunicorn`. You can read about it how other part of the service scales below.

The resource heavy part, namely the embedding uses `sentence-transformers` library with `all-mpnet-base-v2` model. All constants has been set to accomodate this model. By default the embedding is paralellized to all CPU cores (-1) with proper resource allocation (one resource - one process) which theroetically makes it ready to work with GPUs, but this not has been tested. This way the the embedding scales to all available resources on the host, thus vertical scalability achieved for this part. Check out `rag_solution/singleton_worker_pool.py`.

Prometheus metrics (only the FastAPI default) and health endpoint has been exposed.

Rate limit has been set to 4 CPS per IP. The data ingestion piepline uses 5 threads, for the sake of testing.

Vector database is access via async client. With the async web server this external bottleneck is also eliminated.

### Missing feature

* Vector DB authentication has not been setup.
* Idempotent ingestion capability: You cannot supply the ID on ingestion. It could be desirable to be able to upsert into vector DB in case of changing documents (e.g.: Wiki pages).

## Milvus Vector Database

The project uses Milvus vector DB, which is my personal favourite due to high customizability. It calculates BM25 embeddings within the DB, stores dense and sparse embedding, the document itself and the metadata express as dictinoary. It does sparse-dense hybrid score calculation for retrival with weighting. It also gives us the ability to append queries with filtering on metadata. 

By default it spins up a file backed in-memory instance, which is perfect for testing and development. The Docker-Compose example spins a Milvus Standalone DB and connects to it. The data is stored at `/data` folder which is mounted to the container.

NOTE: Even the Docker-Compose example is not a production grade deployment of Milvus. For production grade scenarios the Helm chart should be used on Kubernetes infrastructure as Milvus relies on multiple sub-services. AWS S3 compatible services can be used as storage backend (or Minio if not available). This is not covered in the submission. For cloud-native deployment Zilliz's Milvus SaaS is recommended.

## Unit and integration test

Tests are using `pytest-asyncio` and implemented as async tests. 

Unit test are using in-memory vector db. AI was heavily used to generate some of these files.

Integration tests (API level) are using `Testcontainers` to spin up a Milvus DB instance to do the tests.

## Data ingestion piepline

Jump to `rag_solution/data_ingestion/README.md`.

## Infrastructure as code 

Jump to `deploy/README.md`.


# Topics to talk about

Here are a couple of topics that have been not covered by this demo project but are cosiderably interesting. Bring these up if you are interested?

* Retrieve accuracy: How to further increase retrieval accuracy (e.g.: contextualization, metadata extraction)? Does chunk size variation matters? What are the options in terms of open-source and SAAS models? Is there a better way to embed than traditional chunking (late-chunking, ColBERT)?

* Retrieve preprocessing: How to do retrieve better than just sending the user query in?

* Deoder-only embedding models: What are they useful for? What are the shortcoming of encoder-only models (e.g.: the one i'm using in this excercise).
