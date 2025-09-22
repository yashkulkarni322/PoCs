from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from services.qdrant_service import QdrantService
from services.embedding_service import EmbeddingService
from utils.file_handler import extract_text_by_type
from config.settings import settings
from utils.logger import setup_logger
import re

logger = setup_logger(__name__)


class DocumentProcessor:
    def __init__(self):
        self.qdrant_service = QdrantService()
        self.embedding_service = EmbeddingService()
    
    def setup_collection(self):
        """Setup Qdrant collection"""
        self.qdrant_service.setup_collection()
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from file with error handling"""
        try:
            return extract_text_by_type(file_path)
        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {str(e)}")
            raise
    
    def chunk_text(self, texts: List[str], content_type: str = "document") -> List[str]:
        """Split text into chunks based on content type"""
        try:
            if content_type == "chat":
                return self._chunk_chat_data(texts)
            elif content_type == "finance":
                return self._chunk_finance_data(texts)
            else:
                return self._chunk_generic_text(texts)
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            raise
    
    def _chunk_chat_data(self, texts: List[str]) -> List[str]:
        """Chunk chat data: 5 messages grouped with 1 message overlap"""
        all_chunks = []
        for text in texts:
            messages = self._parse_chat_messages(text)
            
            chunks = []
            for i in range(0, len(messages), 4):  # Step by 4 (5-1 overlap)
                chunk_messages = messages[i:i+5]
                if chunk_messages:
                    chunk_text = '\n'.join(chunk_messages)
                    chunks.append(chunk_text)
            
            all_chunks.extend(chunks)
        return all_chunks
    
    def _parse_chat_messages(self, text: str) -> List[str]:
        """Parse chat text into individual messages"""
        messages = []
        lines = text.split('\n')
        current_message = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts a new message (has timestamp or From:)
            if ('From:' in line or 'Timestamp:' in line or 
                re.match(r'\d+/\d+/\d+', line) or '----' in line):
                
                if current_message:
                    messages.append('\n'.join(current_message))
                    current_message = []
                
                if '----' not in line:  # Skip separator lines
                    current_message.append(line)
            else:
                if current_message:  # Only add if we're in a message
                    current_message.append(line)
        
        # Add last message
        if current_message:
            messages.append('\n'.join(current_message))
        
        return messages
    
    def _chunk_finance_data(self, texts: List[str]) -> List[str]:
        """Chunk finance data: 5 rows grouped with 2 rows overlap, headers attached"""
        all_chunks = []
        for text in texts:
            try:
                headers, transactions = self._parse_bank_statement(text)
                
                chunks = []
                for i in range(0, len(transactions), 3):  # Step by 3 (5-2 overlap)
                    chunk_transactions = transactions[i:i+5]
                    if chunk_transactions:
                        chunk_text = f"{headers}\n" + '\n'.join(chunk_transactions)
                        chunks.append(chunk_text)
                
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"Error parsing finance data, using generic chunking: {str(e)}")
                all_chunks.extend(self._chunk_generic_text([text]))
        
        return all_chunks
    
    def _parse_bank_statement(self, text: str) -> tuple:
        """Parse bank statement to extract headers and transaction rows"""
        lines = text.split('\n')
        
        # Extract account info as header
        account_info = []
        transaction_lines = []
        in_transactions = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for transaction header
            if ('Branch Name' in line and 'Transaction Date' in line) or in_transactions:
                in_transactions = True
                if ('Branch Name' in line or 'Debit' in line or 'Credit' in line):
                    continue  # Skip header row
                transaction_lines.append(line)
            elif not in_transactions and any(keyword in line for keyword in 
                ['Account', 'Statement From', 'IFSC', 'MICR', 'Currency']):
                account_info.append(line)
        
        # Create headers
        if account_info:
            headers = "Bank Statement Summary:\n" + '\n'.join(account_info[:5])
        else:
            headers = "Transaction Headers: Date | Type | Description | Debit | Credit | Balance"
        
        return headers, transaction_lines
    
    def _chunk_generic_text(self, texts: List[str]) -> List[str]:
        """Generic text chunking using RecursiveCharacterTextSplitter"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        
        all_chunks = []
        for text in texts:
            chunks = splitter.split_text(text)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def prepare_points_for_ingestion(self, chunks: List[str], content_type: str) -> List:
        """Prepare points for Qdrant ingestion"""
        try:
            dense_embeddings, sparse_embeddings = self.embedding_service.process_embeddings_in_batches(chunks)
            
            points = []
            start_id = self.qdrant_service.get_next_id()
            
            for idx, (text, dense_emb, sparse_emb) in enumerate(zip(chunks, dense_embeddings, sparse_embeddings)):
                sparse_emb_dict = self.embedding_service.ensure_sparse_dict_format(sparse_emb)
                point = self.qdrant_service.create_point(
                    start_id + idx, dense_emb, sparse_emb_dict, text, content_type
                )
                points.append(point)
            
            return points
        except Exception as e:
            logger.error(f"Error preparing points for ingestion: {str(e)}")
            raise
    
    def ingest_documents(self, file_paths: List[str], content_type: str) -> int:
        """Process and ingest documents from file paths"""
        try:
            texts = [self.extract_text(file_path) for file_path in file_paths]
            all_chunks = self.chunk_text(texts, content_type)
            logger.info(f"Total number of chunks: {len(all_chunks)}")
            
            if not all_chunks:
                return 0
            
            return self._ingest_text_chunks(all_chunks, content_type)
        except Exception as e:
            logger.error(f"Error ingesting documents: {str(e)}")
            raise
    
    def ingest_text_insights(self, insights_text: str, content_type: str) -> int:
        """Ingest generated insights as text chunks"""
        try:
            all_chunks = self.chunk_text([insights_text], content_type)
            logger.info(f"Total insight chunks: {len(all_chunks)}")
            
            if not all_chunks:
                return 0
            
            return self._ingest_text_chunks(all_chunks, content_type)
        except Exception as e:
            logger.error(f"Error ingesting text insights: {str(e)}")
            raise
    
    def _ingest_text_chunks(self, chunks: List[str], content_type: str) -> int:
        """Common method to ingest text chunks with embeddings"""
        try:
            points = self.prepare_points_for_ingestion(chunks, content_type)
            upserted_count = self.qdrant_service.upsert_points_in_batches(points)
            
            logger.info(f"Inserted {upserted_count} points into collection '{settings.collection_name}' with content_type '{content_type}'.")
            return upserted_count
        except Exception as e:
            logger.error(f"Error in _ingest_text_chunks: {str(e)}")
            raise