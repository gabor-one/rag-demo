from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MILVUS_URI: str = "./data/milvus.db"
    MAX_EXECUTOR_POOL_SIZE: int | None = None  # Default value for executor pool size. If None: max core count - 1


settings = Settings()
