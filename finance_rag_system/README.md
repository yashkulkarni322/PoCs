# Finance RAG System

A document analysis system that uses AI to extract insights from financial documents and chat conversations.

## Features

- Analyze financial documents (bank statements, credit card records, loan documents)
- Monitor chat conversations for suspicious activities
- Support multiple file formats (PDF, TXT, HTML, CSV, Excel)
- RESTful API with automatic documentation
- Vector search with hybrid retrieval (dense + sparse)
- Configurable through environment variables

## Requirements

- Python 3.8+
- Qdrant vector database
- LLM server (OpenAI-compatible API)
- Jina AI API key

## Quick Start 

1. **Install dependencies**
   
   pip install -r requirements.txt
   

2. **Setup environment**
   
   cp .env.example .env
   # Edit .env with your API keys and URLs


3. **Start Qdrant database**
   
   docker run -p 6333:6333 qdrant/qdrant
   

4. **Run the application**
   
   python main.py
   

5. **Access the API**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

## Project Structure

```
finance_rag_system/
├── main.py                  # Application entry point
├── config/
│   └── settings.py         # Configuration management
├── api/
│   ├── models.py          # Request/response models
│   └── routes.py          # API endpoints
├── services/
│   ├── document_processor.py  # Document processing
│   ├── rag_system.py         # RAG implementation
│   ├── embedding_service.py  # Embeddings handling
│   └── qdrant_service.py     # Vector database ops
└── utils/
    ├── file_handler.py    # File processing utilities
    └── logger.py          # Logging setup
```

## API Endpoints

- `POST /api/v1/ingest-docs` - Upload documents
- `POST /api/v1/analyze-finance-docs` - Analyze financial documents
- `POST /api/v1/analyze-chats` - Analyze chat conversations
- `POST /api/v1/query-docs` - General document queries