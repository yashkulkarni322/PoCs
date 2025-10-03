import requests
from typing import List, Dict, Any
from fastembed import SparseTextEmbedding
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class EmbeddingService:
    def __init__(self):
        self.jina_endpoint = "https://api.jina.ai/v1/embeddings"
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    
    def get_jina_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get dense embeddings from Jina API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.jina_api_key}"
        }
        data = {
            "input": texts,
            "model": "jina-embeddings-v3",
            "dimensions": 1024,
            "task": "retrieval.passage"
        }
        
        try:
            response = requests.post(self.jina_endpoint, headers=headers, json=data)
            response.raise_for_status()
            return [item["embedding"] for item in response.json()["data"]]
        except Exception as e:
            logger.error(f"Error getting Jina embeddings: {str(e)}")
            raise
    
    def get_sparse_embeddings(self, texts: List[str]):
        """Get sparse embeddings"""
        try:
            return list(self.sparse_model.embed(texts))
        except Exception as e:
            logger.error(f"Error getting sparse embeddings: {str(e)}")
            raise
    
    def ensure_sparse_dict_format(self, sparse_emb) -> Dict[str, Any]:
        """Convert sparse embedding to Qdrant format"""
        if isinstance(sparse_emb, dict):
            return sparse_emb
        elif hasattr(sparse_emb, "indices") and hasattr(sparse_emb, "values"):
            return {
                "indices": list(sparse_emb.indices), 
                "values": list(sparse_emb.values)
            }
        else:
            raise ValueError(f"Unknown sparse embedding format: {type(sparse_emb)}")
    
    def process_embeddings_in_batches(self, texts: List[str]) -> tuple:
        """Process texts to get both dense and sparse embeddings"""
        dense_embeddings = []
        batch_size = settings.dense_embedding_batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.get_jina_embeddings(batch)
            dense_embeddings.extend(batch_embeddings)
        
        sparse_embeddings = self.get_sparse_embeddings(texts)
        return dense_embeddings, sparse_embeddings