from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MILVUSDB_URI: str = "./data/milvus.db"
    MAX_EXECUTOR_POOL_SIZE: int | None = None  # Default value for executor pool size. If None: max core count - 1
    SENTENCE_TRANSFORMER_LOCAL_FILES_ONLY: bool = False  # Whether to use local files only for SentenceTransformer models
    SENTENCE_TRANSFORMER_CACHE_DIR: str = "./data"  # Directory to cache SentenceTransformer models

settings = Settings()
