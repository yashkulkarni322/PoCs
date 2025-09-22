from pydantic import BaseModel
from typing import List, Optional


class AnalysisRequest(BaseModel):
    query: str
    store_insights: bool = True


class AnalysisResponse(BaseModel):
    answer: str
    context: List[str]
    total_time: float
    retrieval_time: float
    generation_time: float
    insights_stored: int = 0


class DocumentIngestionResponse(BaseModel):
    message: str
    chunks_ingested: int
    processing_time: float
    file_names: List[str]


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None