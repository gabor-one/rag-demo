from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MILVUS_URI: str = "./data/milvus.db"


settings = Settings()
