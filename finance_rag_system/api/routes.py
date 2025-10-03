import os
import time
import tempfile
from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from api.models import (
    AnalysisRequest, 
    AnalysisResponse, 
    DocumentIngestionResponse,
    ErrorResponse
)
from services.rag_system import RAGSystem
from services.document_processor import DocumentProcessor
from utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

# Initialize services
rag_system = RAGSystem()
doc_processor = DocumentProcessor()


@router.post("/analyze-finance-docs", response_model=AnalysisResponse)
async def analyze_finance_docs(request: AnalysisRequest):
    """Generate insights from financial documents"""
    try:
        result = rag_system.analyze_finance_docs(request.query, request.store_insights)
        return AnalysisResponse(**result)
    except Exception as e:
        logger.error(f"Failed to analyze finance docs: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to analyze finance docs: {str(e)}"
        )


@router.post("/analyze-chats", response_model=AnalysisResponse)
async def analyze_chats(request: AnalysisRequest):
    """Generate chat anomaly detection and user profiling insights"""
    try:
        result = rag_system.analyze_chats(request.query, request.store_insights)
        return AnalysisResponse(**result)
    except Exception as e:
        logger.error(f"Failed to analyze chats: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to analyze chats: {str(e)}"
        )


@router.post("/query-docs", response_model=AnalysisResponse)
async def query_docs(request: AnalysisRequest):
    """Query the entire document collection with general questions"""
    try:
        result = rag_system.query_documents(request.query)
        return AnalysisResponse(**result)
    except Exception as e:
        logger.error(f"Failed to query documents: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to query documents: {str(e)}"
        )


def validate_file_type(filename: str) -> bool:
    """Validate if file type is supported"""
    supported_extensions = {'.pdf', '.txt', '.html', '.htm', '.csv', '.xlsx', '.xls'}
    return Path(filename).suffix.lower() in supported_extensions


def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file to temporary location"""
    suffix = Path(file.filename).suffix
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    
    try:
        content = file.read()
        temp_file.write(content)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        temp_file.close()
        os.unlink(temp_file.name)
        raise e


def cleanup_temp_files(file_paths: List[str]):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            os.unlink(file_path)
        except Exception as e:
            logger.warning(f"Could not delete temporary file {file_path}: {e}")


@router.post("/ingest-docs", response_model=DocumentIngestionResponse)
async def ingest_docs(
    files: List[UploadFile] = File(...),
    content_type: str = Form("document")
):
    """Ingest new documents into the system"""
    start_time = time.time()
    file_paths = []
    
    try:
        file_names = []
        
        # Validate and save files
        for file in files:
            if not validate_file_type(file.filename):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file.filename}. "
                           f"Only PDF, TXT, HTML, CSV, and Excel files are supported."
                )
            
            # Save file temporarily
            temp_path = await save_uploaded_file_async(file)
            file_paths.append(temp_path)
            file_names.append(file.filename)
        
        # Process documents
        chunks_ingested = doc_processor.ingest_documents(file_paths, content_type)
        processing_time = time.time() - start_time
        
        return DocumentIngestionResponse(
            message=f"Successfully ingested {len(files)} document(s)",
            chunks_ingested=chunks_ingested,
            processing_time=processing_time,
            file_names=file_names
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to ingest documents: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to ingest documents: {str(e)}"
        )
    finally:
        cleanup_temp_files(file_paths)


async def save_uploaded_file_async(file: UploadFile) -> str:
    """Async version of save_uploaded_file"""
    suffix = Path(file.filename).suffix
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    
    try:
        content = await file.read()
        temp_file.write(content)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        temp_file.close()
        os.unlink(temp_file.name)
        raise e


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "RAG system is running"}