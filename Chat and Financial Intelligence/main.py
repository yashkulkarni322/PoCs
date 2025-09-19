import fitz  # PyMuPDF
import requests
import time
import pandas as pd
from typing import List, Union
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from langchain_community.embeddings import JinaEmbeddings
from qdrant_client import QdrantClient, models
from langchain_core.prompts import ChatPromptTemplate
from fastembed import SparseTextEmbedding
import os
import tempfile
from pathlib import Path
from bs4 import BeautifulSoup

# === Configuration ===
JINA_API_KEY = "jina_749fcb059c2f422d8ea05b9a1b95f693V5KvVT-7_eUEcs4C3WsyKX-lJJ_M"
QDRANT_URL = "http://192.168.1.13:6333"
COLLECTION_NAME = "finance_docs"
K = 15
LLM_URL = "http://192.168.1.11:8078/v1/chat/completions"
LLM_MODEL = "openai/gpt-oss-20b"

app = FastAPI()

class DocumentProcessor:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL, check_compatibility=False)
        self.jina_endpoint = "https://api.jina.ai/v1/embeddings"
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF"""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    
    def extract_text_from_txt(self, txt_path):
        """Extract text from TXT file"""
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def extract_text_from_html(self, html_path):
        """Extract text from HTML file"""
        with open(html_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean it up
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def extract_text_from_csv(self, csv_path):
        """Extract text from CSV file"""
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            
            # Convert DataFrame to structured text
            text_parts = []
            
            # Add headers
            headers = " | ".join(str(col) for col in df.columns)
            text_parts.append(f"CSV Headers: {headers}")
            text_parts.append("-" * len(headers))
            
            # Add rows with proper formatting
            for idx, row in df.iterrows():
                row_text = " | ".join(str(value) if pd.notna(value) else "" for value in row)
                text_parts.append(f"Row {idx + 1}: {row_text}")
            
            # Add summary information
            text_parts.append(f"\nCSV Summary: {len(df)} rows, {len(df.columns)} columns")
            
            return "\n".join(text_parts)
            
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")
    
    def extract_text_from_excel(self, excel_path):
        """Extract text from Excel file"""
        try:
            # Read all sheets from the Excel file
            xl_file = pd.ExcelFile(excel_path)
            all_text = []
            
            for sheet_name in xl_file.sheet_names:
                # Read each sheet
                df = pd.read_excel(excel_path, sheet_name=sheet_name)
                
                # Add sheet header
                all_text.append(f"\n=== Sheet: {sheet_name} ===\n")
                
                # Convert DataFrame to text with better formatting
                if not df.empty:
                    # Create a structured text representation
                    sheet_text = []
                    
                    # Add column headers
                    headers = " | ".join(str(col) for col in df.columns)
                    sheet_text.append(f"Headers: {headers}")
                    sheet_text.append("-" * len(headers))
                    
                    # Add rows
                    for idx, row in df.iterrows():
                        row_text = " | ".join(str(value) if pd.notna(value) else "" for value in row)
                        sheet_text.append(f"Row {idx + 1}: {row_text}")
                    
                    all_text.append("\n".join(sheet_text))
                else:
                    all_text.append("Empty sheet")
            
            return "\n".join(all_text)
            
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {str(e)}")
    
    def extract_text(self, file_path):
        """Extract text from file based on extension"""
        file_extension = Path(file_path).suffix.lower()
        if file_extension == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_extension == '.txt':
            return self.extract_text_from_txt(file_path)
        elif file_extension == '.html' or file_extension == '.htm':
            return self.extract_text_from_html(file_path)
        elif file_extension == '.csv':
            return self.extract_text_from_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            return self.extract_text_from_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def chunk_text(self, texts):
        """Split text into chunks"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2048,
            chunk_overlap=100
        )
        
        all_chunks = []
        for text in texts:
            chunks = splitter.split_text(text)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def get_jina_embeddings(self, texts, dimensions=1024, task="retrieval.passage"):
        """Get dense embeddings from Jina API"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}"
        }
        data = {
            "input": texts,
            "model": "jina-embeddings-v3",
            "dimensions": dimensions,
            "task": task
        }
        response = requests.post(self.jina_endpoint, headers=headers, json=data)
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]
    
    def get_sparse_embeddings(self, texts):
        """Get sparse embeddings"""
        return list(self.sparse_model.embed(texts))
    
    def ensure_sparse_dict(self, sparse_emb):
        """Convert sparse embedding to Qdrant format"""
        if isinstance(sparse_emb, dict):
            return sparse_emb
        elif hasattr(sparse_emb, "indices") and hasattr(sparse_emb, "values"):
            return {"indices": list(sparse_emb.indices), "values": list(sparse_emb.values)}
        else:
            raise ValueError(f"Unknown sparse embedding format: {type(sparse_emb)}")
    
    def setup_collection(self):
        """Create or verify Qdrant collection"""
        try:
            self.client.get_collection(COLLECTION_NAME)
            print(f"Collection '{COLLECTION_NAME}' already exists.")
        except Exception:
            self.client.recreate_collection(
                collection_name=COLLECTION_NAME,
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
            print(f"Collection '{COLLECTION_NAME}' created.")
            
        # Create payload index
        try:
            self.client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="content_type",
                field_schema="keyword"
            )
            print(f"Payload index for field 'content_type' created.")
        except Exception as e:
            print(f"Index might already exist: {e}")
    
    def ingest_documents(self, file_paths, content_type="document"):
        """Process and ingest documents from file paths"""
        texts = [self.extract_text(file_path) for file_path in file_paths]
        
        # Chunk text
        all_chunks = self.chunk_text(texts)
        print(f"Total number of chunks: {len(all_chunks)}")
        
        if not all_chunks:
            return 0
        
        return self._ingest_text_chunks(all_chunks, content_type)
    
    def ingest_text_insights(self, insights_text, content_type):
        """Ingest generated insights as text chunks"""
        # Chunk the insights text
        all_chunks = self.chunk_text([insights_text])
        print(f"Total insight chunks: {len(all_chunks)}")
        
        if not all_chunks:
            return 0
            
        return self._ingest_text_chunks(all_chunks, content_type)
    
    def _ingest_text_chunks(self, chunks, content_type):
        """Common method to ingest text chunks with embeddings"""
        # Get embeddings in batches
        batch_size = 32
        dense_embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            dense_embeddings.extend(self.get_jina_embeddings(batch))
        
        sparse_embeddings = self.get_sparse_embeddings(chunks)
        
        # Prepare points
        points = []
        start_id = self.get_next_id()
        
        for idx, (text, dense_emb, sparse_emb) in enumerate(zip(chunks, dense_embeddings, sparse_embeddings)):
            dense_emb_list = dense_emb.tolist() if hasattr(dense_emb, "tolist") else dense_emb
            sparse_emb_dict = self.ensure_sparse_dict(sparse_emb)
            points.append({
                "id": start_id + idx,
                "vector": {
                    "dense": dense_emb_list,
                    "sparse": sparse_emb_dict
                },
                "payload": {
                    "text": text,
                    "content_type": content_type,
                    "timestamp": int(time.time())
                }
            })
        
        # Upsert points in batches
        upsert_batch_size = 256
        for i in range(0, len(points), upsert_batch_size):
            batch = points[i:i+upsert_batch_size]
            self.client.upsert(collection_name=COLLECTION_NAME, points=batch)
            print(f"Upserted points {i} to {i+len(batch)-1}")
        
        print(f"Inserted {len(points)} points into collection '{COLLECTION_NAME}' with content_type '{content_type}'.")
        return len(points)
    
    def get_next_id(self):
        """Get the next available ID in the collection"""
        try:
            collection_info = self.client.get_collection(COLLECTION_NAME)
            points_count = collection_info.points_count
            return points_count
        except:
            return 0

class RAGSystem:
    def __init__(self):
        self.client = QdrantClient(url=QDRANT_URL, check_compatibility=False)
        self.embeddings = JinaEmbeddings(jina_api_key=JINA_API_KEY, model_name="jina-embeddings-v3")
        
        self._setup_retrievers()
        self._setup_prompts()
        
        self.doc_processor = DocumentProcessor()
        
    def _setup_retrievers(self):
        dense_vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=COLLECTION_NAME,
            embedding=self.embeddings,
            retrieval_mode=RetrievalMode.DENSE,
            vector_name="dense",
            content_payload_key="text"
        )
        self.dense_retriever = dense_vectorstore.as_retriever(search_kwargs={"k": K})
        
        sparse_model = FastEmbedSparse(model_name="qdrant/bm25")
        sparse_vectorstore = QdrantVectorStore(
            client=self.client,
            embedding=self.embeddings,
            collection_name=COLLECTION_NAME,
            sparse_embedding=sparse_model,
            retrieval_mode=RetrievalMode.SPARSE,
            sparse_vector_name="sparse",
            content_payload_key="text"
        )
        self.sparse_retriever = sparse_vectorstore.as_retriever(search_kwargs={"k": K})
        
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.dense_retriever, self.sparse_retriever],
            weights=[0.5, 0.5]
        )
    
    def _setup_prompts(self):
        # Finance Analysis Prompt
        self.finance_prompt = ChatPromptTemplate.from_template("""
        Using the financial documents provided, generate comprehensive insights using the schema below:

1. Transaction-Level Insights
- Covers any single financial event (bank transfer, card swipe, EMI payment, stock trade).
- Transaction Classification: Categories: salary, transfer, cash withdrawal, card payment, EMI, investment, unknown.
- Entity Mapping: Beneficiaries, merchants, organizations, exchanges, aliases.
- Suspicious Activity Detection: Unusual amounts, international transfers, round-figure patterns, repeated micro-payments.
- Transaction Timeline: Chronological view of financial flows across accounts/cards.

2. Bank Account-Level Insights
- Account Profiling: Salary inflows, regular expenses, transfers, loans.
- Cash Flow Analysis: Monthly inflow/outflow trends, balance sustainability.
- Counterparty Networks: Graph of frequent payees/beneficiaries.
- Cross-Account Links: Shared beneficiaries or mirrored transactions across multiple accounts.

3. Credit Card-Level Insights
- Credit cards provide high-resolution lifestyle and spending behavior.
- Spending Categorization: Groceries, luxury goods, travel, online services, utilities.
- Merchant Risk Analysis: Transactions with flagged or high-risk merchants (crypto, betting, darknet).
- High-Value/Luxury Purchases: Lifestyle analysis (e.g., luxury watches, foreign travel, highend electronics).
- Debt & Repayment Patterns: EMI usage, partial vs. full repayment, cash withdrawals.
- Geographic Usage: Locations of swipes, domestic vs. international usage.

4. Loan, EMI & Credit Reports
- Loan History: Disbursed loans, EMIs, outstanding balances.
- Defaults & Delays: Missed payments or irregular repayment schedules.
- Credit Behavior: Over-leveraging, multiple concurrent loans.
- Financial Stress Indicators: Increased reliance on credit vs. income.

5. Investment & Trading Insights
- Investment Profile: Stocks, mutual funds, bonds, crypto, property.
- Risk Appetite: Conservative (FDs, bonds) vs. speculative (crypto, frequent stock trading).
- Unusual Transactions: Sudden large investments, pump-and-dump patterns.
- Cross-Entity Links: Shared trading platforms or wallets with suspects.

6. Behavioural & Temporal Financial Patterns
- Spending Patterns: Routine vs. spikes (travel season, festivals, sudden luxury purchases).
- Time-of-Day/Seasonal Trends: Late-night ATM withdrawals, clustered swipes.
- Behavioural Anomalies: Lifestyle shifts, sudden surge in credit dependence.

7. Risk Indicators & Red Flags
- Structuring & Layering: Splitting transactions to avoid thresholds.
- International Transactions: High-frequency foreign transfers or foreign card swipes.
- Known Risk Merchants: Gambling, crypto exchanges, money mules.
- Lifestyle vs. Declared Income: Spending much higher than reported salary

All the insights generated should be in neat bullet points format only in the same order as above. No tabular data should be generated.                                                               

        Context: {context}
        Query: {question}
        
        Provide detailed financial insights based on the above schema.
        """)
        
        # Chat Analysis Prompt
        self.chat_prompt = ChatPromptTemplate.from_template("""
        Analyze the chat conversations provided to generate user profiling and anomaly detection insights:

Generate insights using the following headers. 
Use clear natural text under each header, not JSON, and keep the structure consistent. 
Each section should contain explanations and findings relevant to the chat.

This is insights of the chat between "participants (include whatever participants are involved here)"

Summary => A 2-3 line short readable overview of the conversation

Who-Talks-to-Whom => Maps interactions between participants, showing central or influential members, hidden subgroups, and unusual communication patterns

Insights :
- Sentiment & Tone detection => identify emotions (anger, fear, urgency, planning, threats)
- Message threat level => Suspicious , Non-Suspicious
- Coded Language => detect slang, euphemisms, or shorthand

Dynamics and Influence :
- Conversation Clusters => group related chat threads by topics, intent, or event
- Key Topics => dominant discussion themes
- Influence Analysis => identify central participants, their role, and influence
- Hidden Subgroups => detect private conversations branching from group chats
- Cross-platform Signals => references to other apps (Telegram, Signal, SMS, etc.)
- Behavioural Patters => uncovers frequency, volume, and timing of communications, helping to detect coordination or anomalies

Behavioural and Temporal Analysis :
- Communication Frequency => spikes/drops in activity around incidents
- Time-of-day Patterns => unusual activity at odd hours
- Behavioural Anomalies => sudden changes in style or volume of messages

Risk Indicators :
- Keyword / Phrases => sensitive or red-flag words
- Suspicious Coordination => simultaneous planning across chats
- Dark-to-clear Signals => coded language shifting to explicit instructions
- Urgency or Escalation Classification => aggressive/directive tone increasing suddenly
                                                            
All the insights generated should be in neat bullet points format only in the same order as above. No tabular data should be generated.                                                               


        Context: {context}
        Query: {question}
        
        Provide comprehensive chat analysis and user profiling insights.
        """)
        
        # General Query Prompt
        self.general_prompt = ChatPromptTemplate.from_template("""
        Based on the provided context, answer the following question comprehensively and accurately.

        Context: {context}
        Question: {question}
        
        Answer only in bullet points or plain sentences. No tabular format
        """)
    
    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def _generate_response(self, prompt_template, query, filter_content_type=None):
        """Generate response using specified prompt template with detailed timing"""
        total_start = time.time()
        
        # RETRIEVAL PHASE - Track retrieval time
        retrieval_start = time.time()
        
        # Apply content type filter if specified
        if filter_content_type:
            docs = self.ensemble_retriever.invoke(query)
            # Filter docs by content_type if needed
            filtered_docs = [doc for doc in docs if doc.metadata.get('content_type') == filter_content_type]
            docs = filtered_docs if filtered_docs else docs
        else:
            docs = self.ensemble_retriever.invoke(query)
        
        context_str = self.format_docs(docs)
        retrieval_time = time.time() - retrieval_start
        
        # GENERATION PHASE - Track generation time
        generation_start = time.time()
        
        formatted_prompt = prompt_template.invoke({
            "context": context_str,
            "question": query
        }).to_string()
        
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "user", "content": formatted_prompt}
            ]
        }
        response = requests.post(
            LLM_URL,
            headers={"Content-Type": "application/json"},
            json=payload
        )
        if response.status_code != 200:
            raise RuntimeError(f"LLM returned error: {response.status_code} - {response.text}")
        
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            answer = data["choices"][0]["message"]["content"].strip()
        else:
            answer = "No valid response from model."
            
        generation_time = time.time() - generation_start
        total_time = time.time() - total_start
        
        return {
            "answer": answer,
            "context": [doc.page_content for doc in docs],
            "total_time": round(total_time, 3),
            "retrieval_time": round(retrieval_time, 3),
            "generation_time": round(generation_time, 3)
        }
    
    def analyze_finance_docs(self, query, store_insights=True):
        """Analyze financial documents and generate insights"""
        result = self._generate_response(self.finance_prompt, query, filter_content_type="finance")
        
        # Store insights back to collection if requested
        if store_insights and result["answer"]:
            try:
                insights_text = f"Query: {query}\n\nFinance Analysis Insights:\n{result['answer']}"
                chunks_stored = self.doc_processor.ingest_text_insights(
                    insights_text, 
                    "finance_insights"
                )
                result["insights_stored"] = chunks_stored
                print(f"Stored {chunks_stored} finance insight chunks to collection")
            except Exception as e:
                print(f"Failed to store finance insights: {e}")
                result["insights_stored"] = 0
        
        return result
    
    def analyze_chats(self, query, store_insights=True):
        """Analyze chat conversations for anomalies and user profiling"""
        result = self._generate_response(self.chat_prompt, query, filter_content_type="chat")
        
        # Store insights back to collection if requested
        if store_insights and result["answer"]:
            try:
                insights_text = f"Query: {query}\n\nChat Analysis Insights:\n{result['answer']}"
                chunks_stored = self.doc_processor.ingest_text_insights(
                    insights_text, 
                    "chat_insights"
                )
                result["insights_stored"] = chunks_stored
                print(f"Stored {chunks_stored} chat insight chunks to collection")
            except Exception as e:
                print(f"Failed to store chat insights: {e}")
                result["insights_stored"] = 0
        
        return result
    
    def query_documents(self, query):
        """General document querying across all content types"""
        return self._generate_response(self.general_prompt, query)

# Initialize systems
doc_processor = DocumentProcessor()
rag_system = RAGSystem()
doc_processor.setup_collection()
print("Systems ready.")

# === API Models ===
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

# === API Endpoints ===

@app.post("/analyze-finance-docs", response_model=AnalysisResponse)
async def analyze_finance_docs(request: AnalysisRequest):
    """Generate insights from financial documents"""
    try:
        result = rag_system.analyze_finance_docs(request.query, request.store_insights)
        return AnalysisResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze finance docs: {str(e)}")

@app.post("/analyze-chats", response_model=AnalysisResponse)
async def analyze_chats(request: AnalysisRequest):
    """Generate chat anomaly detection and user profiling insights"""
    try:
        result = rag_system.analyze_chats(request.query, request.store_insights)
        return AnalysisResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze chats: {str(e)}")

@app.post("/query-docs", response_model=AnalysisResponse)
async def query_docs(request: AnalysisRequest):
    """Query the entire document collection with general questions"""
    try:
        result = rag_system.query_documents(request.query)
        # Add default insights_stored value for consistency
        result["insights_stored"] = 0
        return AnalysisResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query documents: {str(e)}")

@app.post("/ingest-docs", response_model=DocumentIngestionResponse)
async def ingest_docs(
    files: List[UploadFile] = File(...),
    content_type: str = Form("document")
):
    """Ingest new documents (PDF, TXT, HTML, CSV, or Excel files) into the system"""
    start_time = time.time()
    
    try:
        file_paths = []
        file_names = []
        
        # Save uploaded files temporarily
        for file in files:
            if not file.filename.lower().endswith(('.pdf', '.txt', '.html', '.htm', '.csv', '.xlsx', '.xls')):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {file.filename}. Only PDF, TXT, HTML, CSV, and Excel (.xlsx, .xls) files are supported."
                )
            
            # Create temporary file
            suffix = Path(file.filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                content = await file.read()
                temp_file.write(content)
                file_paths.append(temp_file.name)
                file_names.append(file.filename)
        
        # Process and ingest documents
        chunks_ingested = doc_processor.ingest_documents(file_paths, content_type)
        
        # Clean up temporary files
        for file_path in file_paths:
            try:
                os.unlink(file_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {file_path}: {e}")
        
        processing_time = time.time() - start_time
        
        return DocumentIngestionResponse(
            message=f"Successfully ingested {len(files)} document(s)",
            chunks_ingested=chunks_ingested,
            processing_time=processing_time,
            file_names=file_names
        )
        
    except Exception as e:
        # Clean up temporary files in case of error
        if 'file_paths' in locals():
            for file_path in file_paths:
                try:
                    os.unlink(file_path)
                except:
                    pass
        raise HTTPException(status_code=500, detail=f"Failed to ingest documents: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
