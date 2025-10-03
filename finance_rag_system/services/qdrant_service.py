import time
from typing import List, Dict, Any
from qdrant_client import QdrantClient, models
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class QdrantService:
    def __init__(self):
        self.client = QdrantClient(url=settings.qdrant_url, check_compatibility=False)
    
    def setup_collection(self):
        """Create or verify Qdrant collection"""
        try:
            self.client.get_collection(settings.collection_name)
            logger.info(f"Collection '{settings.collection_name}' already exists.")
        except Exception:
            self._create_collection()
            logger.info(f"Collection '{settings.collection_name}' created.")
        
        self._create_payload_index()
    
    def _create_collection(self):
        """Create new Qdrant collection"""
        self.client.recreate_collection(
            collection_name=settings.collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=1024,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams()
            },
            shard_number=1,
        )
    
    def _create_payload_index(self):
        """Create payload index for content_type field"""
        try:
            self.client.create_payload_index(
                collection_name=settings.collection_name,
                field_name="content_type",
                field_schema="keyword"
            )
            logger.info("Payload index for field 'content_type' created.")
        except Exception as e:
            logger.info(f"Index might already exist: {e}")
    
    def get_next_id(self) -> int:
        """Get the next available ID in the collection"""
        try:
            collection_info = self.client.get_collection(settings.collection_name)
            return collection_info.points_count
        except Exception as e:
            logger.error(f"Error getting next ID: {str(e)}")
            return 0
    
    def upsert_points_in_batches(self, points: List[Dict[str, Any]]) -> int:
        """Upsert points to Qdrant in batches with error handling"""
        try:
            batch_size = settings.upsert_batch_size
            total_upserted = 0
            
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                self.client.upsert(collection_name=settings.collection_name, points=batch)
                total_upserted += len(batch)
                logger.info(f"Upserted points {i} to {i+len(batch)-1}")
            
            return total_upserted
        except Exception as e:
            logger.error(f"Error upserting points to Qdrant: {str(e)}")
            raise
    
    def create_point(self, point_id: int, dense_emb: List[float], sparse_emb: Dict, 
                     text: str, content_type: str) -> Dict[str, Any]:
        """Create a single point for Qdrant"""
        dense_emb_list = dense_emb.tolist() if hasattr(dense_emb, "tolist") else dense_emb
        
        return {
            "id": point_id,
            "vector": {
                "dense": dense_emb_list,
                "sparse": sparse_emb
            },
            "payload": {
                "text": text,
                "content_type": content_type,
                "timestamp": int(time.time())
            }
        }