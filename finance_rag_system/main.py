import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from services.document_processor import DocumentProcessor
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Finance RAG System",
    description="A comprehensive RAG system for financial document analysis and chat monitoring",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")

# Initialize system on startup
@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup"""
    try:
        doc_processor = DocumentProcessor()
        doc_processor.setup_collection()
        logger.info("Finance RAG System initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize system: {str(e)}")
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Finance RAG System API",
        "version": "1.0.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    uvicorn.run(
        app, 
        host=settings.host, 
        port=settings.port,
        log_level=settings.log_level.lower()
    )