import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # API Keys
    jina_api_key: str
    
    # Database Configuration
    qdrant_url: str = "http://192.168.1.13:6333"
    collection_name: str = "finance_docs"
    
    # Model Configuration
    llm_url: str = "http://192.168.1.11:8078/v1/chat/completions"
    llm_model: str = "openai/gpt-oss-20b"
    
    # Processing Configuration
    chunk_size: int = 2048
    chunk_overlap: int = 100
    retrieval_k: int = 15
    dense_embedding_batch_size: int = 32
    upsert_batch_size: int = 256
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"


settings = Settings()
