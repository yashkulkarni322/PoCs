import time
import requests
from typing import List, Dict, Any, Optional
from langchain.retrievers import EnsembleRetriever
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_community.embeddings import JinaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from services.document_processor import DocumentProcessor
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class RAGSystem:
    def __init__(self):
        self.client = QdrantClient(url=settings.qdrant_url, check_compatibility=False)
        self.embeddings = JinaEmbeddings(
            jina_api_key=settings.jina_api_key, 
            model_name="jina-embeddings-v3"
        )
        self.doc_processor = DocumentProcessor()
        
        self._setup_retrievers()
        self._setup_prompts()
    
    def _setup_retrievers(self):
        """Setup dense and sparse retrievers"""
        try:
            self.dense_retriever = self._create_dense_retriever()
            self.sparse_retriever = self._create_sparse_retriever()
            self.ensemble_retriever = self._create_ensemble_retriever()
        except Exception as e:
            logger.error(f"Error setting up retrievers: {str(e)}")
            raise
    
    def _create_dense_retriever(self):
        """Create dense vector retriever"""
        dense_vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=settings.collection_name,
            embedding=self.embeddings,
            retrieval_mode=RetrievalMode.DENSE,
            vector_name="dense",
            content_payload_key="text"
        )
        return dense_vectorstore.as_retriever(search_kwargs={"k": settings.retrieval_k})
    
    def _create_sparse_retriever(self):
        """Create sparse vector retriever"""
        sparse_model = FastEmbedSparse(model_name="qdrant/bm25")
        sparse_vectorstore = QdrantVectorStore(
            client=self.client,
            embedding=self.embeddings,
            collection_name=settings.collection_name,
            sparse_embedding=sparse_model,
            retrieval_mode=RetrievalMode.SPARSE,
            sparse_vector_name="sparse",
            content_payload_key="text"
        )
        return sparse_vectorstore.as_retriever(search_kwargs={"k": settings.retrieval_k})
    
    def _create_ensemble_retriever(self):
        """Create ensemble retriever combining dense and sparse"""
        return EnsembleRetriever(
            retrievers=[self.dense_retriever, self.sparse_retriever],
            weights=[0.5, 0.5]
        )
    
    def _setup_prompts(self):
        """Setup prompt templates"""
        self.finance_prompt = self._create_finance_prompt()
        self.chat_prompt = self._create_chat_prompt()
        self.general_prompt = self._create_general_prompt()
    
    def _create_finance_prompt(self) -> ChatPromptTemplate:
        """Create finance analysis prompt template"""
        template = """
        Using the financial documents provided, generate comprehensive insights using the schema below:

        1. Transaction-Level Insights
        - Transaction Classification: Categories: salary, transfer, cash withdrawal, card payment, EMI, investment, unknown.
        - Entity Mapping: Beneficiaries, merchants, organizations, exchanges, aliases.
        - Suspicious Activity Detection: Unusual amounts, international transfers, round-figure patterns, repeated micro-payments.

        2. Bank Account-Level Insights
        - Account Profiling: Salary inflows, regular expenses, transfers, loans.
        - Cash Flow Analysis: Monthly inflow/outflow trends, balance sustainability.
        - Cross-Account Links: Shared beneficiaries or mirrored transactions across multiple accounts.

        3. Credit Card-Level Insights
        - Spending Categorization: Groceries, luxury goods, travel, online services, utilities.
        - Merchant Risk Analysis: Transactions with flagged or high-risk merchants.
        - Geographic Usage: Locations of swipes, domestic vs. international usage.

        4. Risk Indicators & Red Flags
        - Structuring & Layering: Splitting transactions to avoid thresholds.
        - International Transactions: High-frequency foreign transfers or foreign card swipes.
        - Lifestyle vs. Declared Income: Spending much higher than reported salary.

        All insights should be in neat bullet points format only. No tabular data.

        Context: {context}
        Query: {question}
        
        Provide detailed financial insights based on the above schema.
        """
        return ChatPromptTemplate.from_template(template)
    
    def _create_chat_prompt(self) -> ChatPromptTemplate:
        """Create chat analysis prompt template"""
        template = """
        Analyze the chat conversations provided to generate user profiling and anomaly detection insights:

        Summary => A 2-3 line short readable overview of the conversation

        Who-Talks-to-Whom => Maps interactions between participants, showing central or influential members

        Insights:
        - Sentiment & Tone detection => identify emotions (anger, fear, urgency, planning, threats)
        - Message threat level => Suspicious, Non-Suspicious
        - Coded Language => detect slang, euphemisms, or shorthand

        Risk Indicators:
        - Keyword / Phrases => sensitive or red-flag words
        - Suspicious Coordination => simultaneous planning across chats
        - Urgency or Escalation Classification => aggressive/directive tone increasing suddenly

        All insights should be in neat bullet points format only. No tabular data.

        Context: {context}
        Query: {question}
        
        Provide comprehensive chat analysis and user profiling insights.
        """
        return ChatPromptTemplate.from_template(template)
    
    def _create_general_prompt(self) -> ChatPromptTemplate:
        """Create general query prompt template"""
        template = """
        Based on the provided context, answer the following question comprehensively and accurately.

        Context: {context}
        Question: {question}
        
        Answer only in bullet points or plain sentences. No tabular format.
        """
        return ChatPromptTemplate.from_template(template)
    
    def format_docs(self, docs) -> str:
        """Format documents for context"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def retrieve_documents(self, query: str, filter_content_type: Optional[str] = None):
        """Retrieve documents with optional content type filtering"""
        docs = self.ensemble_retriever.invoke(query)
        
        if filter_content_type:
            filtered_docs = [
                doc for doc in docs 
                if doc.metadata.get('content_type') == filter_content_type
            ]
            return filtered_docs if filtered_docs else docs
        
        return docs
    
    def generate_llm_response(self, formatted_prompt: str) -> str:
        """Generate response from LLM"""
        payload = {
            "model": settings.llm_model,
            "messages": [
                {"role": "user", "content": formatted_prompt}
            ]
        }
        
        try:
            response = requests.post(
                settings.llm_url,
                headers={"Content-Type": "application/json"},
                json=payload
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"LLM returned error: {response.status_code} - {response.text}")
            
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"].strip()
            else:
                return "No valid response from model."
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            raise
    
    def measure_timing(func):
        """Decorator to measure function timing"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            return result, round(end_time - start_time, 3)
        return wrapper
    
    def _generate_response(self, prompt_template: ChatPromptTemplate, query: str, 
                          filter_content_type: Optional[str] = None) -> Dict[str, Any]:
        """Generate response using specified prompt template with detailed timing"""
        total_start = time.time()
        
        # Retrieval phase
        retrieval_start = time.time()
        docs = self.retrieve_documents(query, filter_content_type)
        context_str = self.format_docs(docs)
        retrieval_time = time.time() - retrieval_start
        
        # Generation phase
        generation_start = time.time()
        formatted_prompt = prompt_template.invoke({
            "context": context_str,
            "question": query
        }).to_string()
        
        answer = self.generate_llm_response(formatted_prompt)
        generation_time = time.time() - generation_start
        
        total_time = time.time() - total_start
        
        return {
            "answer": answer,
            "context": [doc.page_content for doc in docs],
            "total_time": round(total_time, 3),
            "retrieval_time": round(retrieval_time, 3),
            "generation_time": round(generation_time, 3)
        }
    
    def analyze_finance_docs(self, query: str, store_insights: bool = True) -> Dict[str, Any]:
        """Analyze financial documents and generate insights"""
        try:
            result = self._generate_response(self.finance_prompt, query, "finance")
            
            if store_insights and result["answer"]:
                result["insights_stored"] = self._store_insights(
                    query, result["answer"], "finance_insights"
                )
            
            return result
        except Exception as e:
            logger.error(f"Error analyzing finance docs: {str(e)}")
            raise
    
    def analyze_chats(self, query: str, store_insights: bool = True) -> Dict[str, Any]:
        """Analyze chat conversations"""
        try:
            result = self._generate_response(self.chat_prompt, query, "chat")
            
            if store_insights and result["answer"]:
                result["insights_stored"] = self._store_insights(
                    query, result["answer"], "chat_insights"
                )
            
            return result
        except Exception as e:
            logger.error(f"Error analyzing chats: {str(e)}")
            raise
    
    def query_documents(self, query: str) -> Dict[str, Any]:
        """General document querying across all content types"""
        try:
            result = self._generate_response(self.general_prompt, query)
            result["insights_stored"] = 0
            return result
        except Exception as e:
            logger.error(f"Error querying documents: {str(e)}")
            raise
    
    def _store_insights(self, query: str, answer: str, content_type: str) -> int:
        """Store insights back to collection"""
        try:
            insights_text = f"Query: {query}\n\nInsights:\n{answer}"
            chunks_stored = self.doc_processor.ingest_text_insights(
                insights_text, content_type
            )
            logger.info(f"Stored {chunks_stored} insight chunks to collection")
            return chunks_stored
        except Exception as e:
            logger.error(f"Failed to store insights: {str(e)}")
            return 0
