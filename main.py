from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Optional, TypedDict
import os
from dotenv import load_dotenv 
import logging 
import pickle 
import time 
import json 
from uuid import uuid4 
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_cohere import CohereRerank
from langchain.retrievers import EnsembleRetriever
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.schema import Document
from pydantic import BaseModel as PydanticBaseModel, Field as PydanticField
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from simhash import Simhash
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langsmith import Client

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class Citation(BaseModel):
    """Citation from source document"""
    filename: str = Field(description="The filename of the source document")
    page_number: Optional[str] = Field(default=None, description="Page number if available")

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(description="The user's question", min_length=1, max_length=1000)
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation history")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is the dosage for AMTAGVI?",
                "session_id": "user_123_session"
            }
        }

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    answer: str = Field(description="The AI's response to the question")
    citations: List[Citation] = Field(description="Source documents that support the answer")
    session_id: str = Field(description="Session ID for this conversation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "AMTAGVI is provided as...",
                "citations": [
                    {"filename": "amtagvi_prescribing_info.pdf", "page_number": "3"}
                ],
                "session_id": "user_123_session"
            }
        }

class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str = Field(description="Error type")
    detail: str = Field(description="Detailed error message")
    
class StatusResponse(BaseModel):
    """Status endpoint response"""
    status: str = Field(description="Overall system status")
    components: dict = Field(description="Status of individual components")
    vector_db_docs: Optional[int] = Field(default=None, description="Number of documents in vector DB")
    bm25_docs: Optional[int] = Field(default=None, description="Number of documents in BM25 retriever")

class FeedbackRequest(BaseModel):
    """Request model for feedback submission"""
    trace_id: str = Field(description="LangSmith trace/run ID")
    feedback_type: str = Field(description="Type of feedback: 'positive' or 'negative'")
    reason: Optional[str] = Field(default=None, description="Reason for the feedback")
    details: Optional[str] = Field(default=None, description="Additional feedback details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "trace_id": "4e38649a-311e-48ae-9d7e-f64d0c9ee5c8",
                "feedback_type": "negative",
                "reason": "Not factually correct",
                "details": "The dosage information seemed outdated"
            }
        }

class FeedbackResponse(BaseModel):
    """Response model for feedback submission"""
    success: bool = Field(description="Whether feedback was submitted successfully")
    message: str = Field(description="Success or error message")
    feedback_id: Optional[str] = Field(default=None, description="LangSmith feedback ID if successful")

# Global variables to store loaded components
vector_store = None
bm25_retriever = None
ensemble_retriever = None
embedding_model = None
llm = None
cohere_reranker = None
components_loaded = False
rag_graph = None
checkpointer = None
connection_pool = None
langsmith_client = None

async def initialize_langsmith():
    """Initialize LangSmith client"""
    global langsmith_client
    try:
        langsmith_client = Client()
        logger.info("✅ LangSmith client initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize LangSmith client: {e}")
        langsmith_client = None

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await load_components()
    yield
    # Shutdown (cleanup if needed)
    global connection_pool
    if connection_pool:
        await connection_pool.close()
        logger.info("🔌 Closed PostgreSQL connection pool")
    logger.info("🛑 Shutting down...")

async def load_components():
    """Load vector database, BM25 retriever, and initialize API clients"""
    global vector_store, bm25_retriever, ensemble_retriever, embedding_model, llm, cohere_reranker, components_loaded
    
    try:
        logger.info("🚀 Starting component initialization...")
        
        # Initialize embedding model
        logger.info("📊 Initializing embedding model...")
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # Load vector database
        logger.info("🗄️ Loading vector database...")
        db_path = "./full_vector"
        collection_name = "my_documents"
        
        if os.path.exists(db_path) and os.listdir(db_path):
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embedding_model,
                persist_directory=db_path
            )
            doc_count = vector_store._collection.count()
            logger.info(f"✅ Vector database loaded: {doc_count} documents")
        else:
            raise FileNotFoundError(f"Vector database not found at {db_path}")
        
        # Load BM25 retriever
        logger.info("📂 Loading BM25 retriever...")
        bm25_path = "./full_bm25_retriever.pkl"
        
        if os.path.exists(bm25_path):
            with open(bm25_path, 'rb') as f:
                bm25_retriever = pickle.load(f)
            bm25_retriever.k = 15
            logger.info(f"✅ BM25 retriever loaded: {len(bm25_retriever.docs)} documents")
        else:
            raise FileNotFoundError(f"BM25 retriever not found at {bm25_path}")
        
        # Create ensemble retriever
        logger.info("🔗 Creating ensemble retriever...")
        vector_retriever = vector_store.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 15}
        )
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever], 
            weights=[0.5, 0.5]
        )
        
        # Initialize LLM
        logger.info("🤖 Initializing OpenAI LLM...")
        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        
        # Initialize Cohere reranker
        logger.info("🎯 Initializing Cohere reranker...")
        cohere_reranker = CohereRerank(
            model="rerank-english-v3.0",
            top_n=10,
            cohere_api_key=os.getenv("COHERE_API_KEY")
        )
        
        components_loaded = True
        logger.info("🎉 All components loaded successfully!")
        
        # Initialize RAG Graph
        logger.info("🔗 Setting up RAG Graph...")
        await setup_rag_graph()
        await initialize_langsmith()
        
    except Exception as e:
        logger.error(f"❌ Failed to load components: {e}")
        components_loaded = False
        raise e

# ============================================================================
# RAG PIPELINE COMPONENTS
# ============================================================================

# LangGraph State
class AgentState(TypedDict):
    messages: List[BaseMessage]
    documents: List[Document]
    on_topic: str
    rephrased_question: str
    proceed_to_generate: bool
    rephrase_count: int
    question: HumanMessage

# Pydantic models for LLM structured output
class GradeQuestion(PydanticBaseModel):
    # score: str = PydanticField(description="Question is about medical topics? If yes -> 'Yes' if not -> 'No'")
    score: str = PydanticField(description="Always return 'Yes'")

class GradeDocuments(PydanticBaseModel):
    relevant_document_indices: List[int] = PydanticField(description="List of indices (0-based) of documents that are relevant to the question")
    reasoning: str = PydanticField(description="Brief explanation of why these documents were selected as relevant")

def simhash_deduplication(docs, similarity_threshold=2):
    """Deduplicate documents using Simhash"""
    if len(docs) <= 10:
        return docs
    
    doc_hashes = []
    for doc in docs:
        sim_hash = Simhash(doc.page_content)
        doc_hashes.append(sim_hash)
    
    to_remove = set()
    for i in range(len(docs)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(docs)):
            if j in to_remove:
                continue
            distance = doc_hashes[i].distance(doc_hashes[j])
            if distance <= similarity_threshold:
                to_remove.add(j)
    
    final_docs = [doc for i, doc in enumerate(docs) if i not in to_remove]
    return final_docs

def retrieve_chunks(query):
    """Main retrieval function"""
    all_results = ensemble_retriever.invoke(query)
    deduplicated_docs = simhash_deduplication(all_results, similarity_threshold=2)
    
    if len(deduplicated_docs) <= 10:
        final_docs = deduplicated_docs
    else:
        final_docs = cohere_reranker.compress_documents(
            documents=deduplicated_docs[:20],
            query=query
        )
    
    return final_docs

def question_rewriter(state: AgentState):

    state["documents"] = []
    state["on_topic"] = ""
    state["rephrased_question"] = ""
    state["proceed_to_generate"] = False
    state["rephrase_count"] = 0

    if "messages" not in state or state["messages"] is None:
        state["messages"] = []

    if state["question"] not in state["messages"]:
        state["messages"].append(state["question"])

    current_question = state["question"].content
    
    # For first message or simple cases, use as-is
    if len(state["messages"]) <= 1:
        state["rephrased_question"] = current_question
    else:
        # Get last 2 exchanges for context
        recent_context = state["messages"][-3:-1] if len(state["messages"]) > 2 else state["messages"][:-1]
        
        # Simple rephrasing prompt
        context_text = " ".join([msg.content for msg in recent_context if hasattr(msg, 'content')])
        
        system_msg = SystemMessage(content="""Rephrase the user's current question to be standalone, using recent conversation context if needed. 
        If the question is already standalone or doesn't need context, return it as-is.
        Only focus on the CURRENT question - don't change the topic.""")
        
        human_msg = HumanMessage(content=f"Recent context: {context_text}\nCurrent question: {current_question}")
        
        prompt = ChatPromptTemplate.from_messages([system_msg, human_msg])
        response = llm.invoke(prompt.format())
        state["rephrased_question"] = response.content.strip()
    
    return state

def question_classifier(state: AgentState):
    #  system_message = SystemMessage(content="""You are a classifier that determines whether a user's question is about medical topics including:
    # 1. Diseases, conditions, and symptoms
    # 2. Medical treatments and procedures
    # 3. Medications and pharmaceuticals
    # 4. Human anatomy and physiology
    # 5. Medical diagnostics and testing
    # 6. Healthcare and medical advice
    # 7. Mental health and psychology
    # 8. Nutrition and dietary health
    # 9. Medical research and studies
    # 10. Any other healthcare or medical-related topics
    # 11. Medications like Amtagvi
    
    # If the question IS about any medical topics, respond with 'Yes'. Otherwise, respond with 'No'.""")

    system_message = SystemMessage(content="Just return Yes by default")

    human_message = HumanMessage(content=f"User question: {state['rephrased_question']}")
    grade_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    structured_llm = llm.with_structured_output(GradeQuestion)
    grader_llm = grade_prompt | structured_llm
    result = grader_llm.invoke({})
    state["on_topic"] = result.score.strip()
    return state

def on_topic_router(state: AgentState):
    on_topic = state.get("on_topic", "").strip().lower()
    if on_topic == "yes":
        return "retrieve"
    else:
        return "off_topic_response"

def retrieve(state: AgentState):
    documents = retrieve_chunks(state["rephrased_question"])
    state["documents"] = documents
    return state

def retrieval_grader(state: AgentState):
    if not state["documents"]:
        state["proceed_to_generate"] = False
        return state
    
    documents_text = ""
    for i, doc in enumerate(state["documents"]):
        documents_text += f"Document {i}:\n{doc.page_content}\n\n"
    
    system_message = SystemMessage(content="""You are a grader assessing the relevance of retrieved documents to a user question.
    You will be given multiple documents numbered from 0 onwards. Your task is to:
    1. Identify which documents contain information relevant to the user's question
    2. Return the indices (numbers) of the relevant documents
    3. Provide brief reasoning for your selection
    A document is relevant if it contains information that could help answer the user's question, even if partially.""")
    
    human_message = HumanMessage(content=f"User question: {state['rephrased_question']}\n\nDocuments to evaluate:\n{documents_text}") 
    grade_prompt = ChatPromptTemplate.from_messages([system_message, human_message]) 
    structured_llm = llm.with_structured_output(GradeDocuments) 
    grader_llm = grade_prompt | structured_llm 
    result = grader_llm.invoke({}) 
    
    relevant_docs = [] 
    for idx in result.relevant_document_indices: 
        if 0 <= idx < len(state["documents"]): 
            relevant_docs.append(state["documents"][idx]) 
    
    state["documents"] = relevant_docs 
    state["proceed_to_generate"] = len(relevant_docs) > 0 
    return state 

def proceed_router(state: AgentState):
    rephrase_count = state.get("rephrase_count", 0)
    if state.get("proceed_to_generate", False):
        return "generate_answer"
    elif rephrase_count >= 2:
        return "generate_answer"
    else:
        return "refine_question"

def refine_question(state: AgentState):
    rephrase_count = state.get("rephrase_count", 0)
    if rephrase_count >= 2:
        return state
    
    question_to_refine = state["rephrased_question"]
    system_message = SystemMessage(content="""You are a helpful assistant that slightly refines the user's question to improve retrieval results for medical information.
    Provide a slightly adjusted version of the question that might yield better results from a medical knowledge base.""")
    human_message = HumanMessage(content=f"Original question: {question_to_refine}\n\nProvide a slightly refined question for medical information retrieval.")
    refine_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    response = llm.invoke(refine_prompt.format())
    refined_question = response.content.strip()
    state["rephrased_question"] = refined_question
    state["rephrase_count"] = rephrase_count + 1
    return state

def generate_answer(state: AgentState):
    if "messages" not in state or state["messages"] is None:
        raise ValueError("State must include 'messages' before generating an answer.")

    history = state["messages"]
    documents = state["documents"]
    rephrased_question = state["rephrased_question"]

    # Format documents
    formatted_chunks = []
    for i, doc in enumerate(documents):
        filename = doc.metadata.get('filename', 'unknown')
        page_number = doc.metadata.get('page_number', 'unknown')
        chunk_text = f"[Source: {filename}, Page: {page_number}]\n{doc.page_content}\n"
        formatted_chunks.append(chunk_text)
    context = "\n".join(formatted_chunks)

    prompt_template = """You are a document-based assistant. Answer questions using ONLY the provided context documents. Never use external knowledge or training data beyond what's in the documents.

Conversation History: {chat_history}
Question: {input}
Context Documents: {context}

INSTRUCTIONS:
1. First, check if context documents are provided
2. If documents exist: search for direct answers to the question
3. If no direct answer exists: look for related, partially relevant, or tangentially connected information
4. If documents exist but contain no relevant info: describe what topics the documents DO cover
5. If no context documents provided: inform user that no relevant documents were found
6. NEVER supplement with knowledge from outside the provided documents

Response Format:
ANSWER: 
- If no context provided: "No relevant documents were found in the knowledge base for your question about [topic]"
- If direct answer found: [Provide complete answer]
- If partial/related info found: [Explain what related information exists and how it connects to the question]
- If documents exist but no relevant info: "The provided documents do not contain information about [specific topic]. However, the documents do cover: [list main topics from the actual documents]"

EVIDENCE: 
- If no context: "No documents retrieved from knowledge base"
- Otherwise: [Cite specific passages or "No information about [topic] found in documents"]

Guidelines:
- Be genuinely helpful by exploring connections the user might not have considered
- Acknowledge limitations clearly when information is missing or incomplete
- Focus on what IS available rather than what isn't
- Never invent or assume information not explicitly in the documents

Response:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    rag_chain = prompt | llm
    response = rag_chain.invoke({"chat_history": history, "context": context, "input": rephrased_question})
    generation = response.content.strip()
    state["messages"].append(AIMessage(content=generation))
    return state

def off_topic_response(state: AgentState):
    if "messages" not in state or state["messages"] is None:
        state["messages"] = []
    state["messages"].append(AIMessage(content="I'm sorry! I can only answer questions related to medical and healthcare topics."))
    return state

async def setup_rag_graph():
    """Setup the LangGraph workflow"""
    global rag_graph, checkpointer, connection_pool
    
    # Initialize PostgreSQL connection pool
    logger.info("🗄️ Setting up PostgreSQL connection pool...")
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        raise ValueError("POSTGRES_URL environment variable is required")

    # Create connection pool with proper settings
    connection_pool = AsyncConnectionPool(
        conninfo=postgres_url,
        max_size=3,
        kwargs={
            "autocommit": True,
            "row_factory": dict_row,
        }
    )
    
    # Create checkpointer with connection pool
    checkpointer = AsyncPostgresSaver(connection_pool)
    
    # Setup tables (only needed once - comment out after first run)
    checkpointer.setup()
    
    workflow = StateGraph(AgentState)
    workflow.add_node("question_rewriter", question_rewriter)
    workflow.add_node("question_classifier", question_classifier)
    workflow.add_node("off_topic_response", off_topic_response)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("retrieval_grader", retrieval_grader)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("refine_question", refine_question)

    workflow.add_edge("question_rewriter", "question_classifier") 
    workflow.add_conditional_edges("question_classifier", on_topic_router, {"retrieve": "retrieve", "off_topic_response": "off_topic_response"})
    workflow.add_edge("retrieve", "retrieval_grader")
    workflow.add_conditional_edges("retrieval_grader", proceed_router, {"generate_answer": "generate_answer", "refine_question": "refine_question"})
    workflow.add_edge("refine_question", "retrieve")
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("off_topic_response", END)
    workflow.set_entry_point("question_rewriter")

    rag_graph = workflow.compile(checkpointer=checkpointer)
    logger.info("✅ RAG Graph with PostgreSQL checkpointer setup complete!")

# ============================================================================
# STREAMING UTILITIES
# ============================================================================

def serialize_ai_message_chunk(chunk):
    """Extract content from AI message chunk"""
    if hasattr(chunk, 'content'):
        return chunk.content
    return str(chunk)

async def generate_stream_responses(message: str, session_id: Optional[str] = None):
    """Generate streaming responses using LangGraph's native streaming"""
    is_new_conversation = session_id is None
    
    if is_new_conversation:
        # Generate new session ID for first message
        new_session_id = str(uuid4())
        config = {"configurable": {"thread_id": new_session_id}}
        
        # Send session ID first
        yield f'data: {{"type": "session_id", "session_id": "{new_session_id}"}}\n\n'
    else:
        config = {"configurable": {"thread_id": session_id}}
    
    # Stream events from the RAG graph
    events = rag_graph.astream_events(
        {"question": HumanMessage(content=message)},
        version="v2",
        config=config
    )
    
    citations = []
    progress_sent = set()  # Track which progress messages we've sent
    trace_id = None
    
    async for event in events:
        print(event)
        event_type = event["event"]

          # Capture trace ID from the final chain end event
        if event_type == "on_chain_end" and event.get("name") == "LangGraph":
            trace_id = event.get("run_id")
            if trace_id:
                yield f'data: {{"type": "trace_id", "trace_id": "{trace_id}"}}\n\n'
        
        # Handle progress by detecting node starts
        if event_type == "on_chain_start":
            node_name = event.get("name", "")
            
            # Map node names to semantic stage names
            stage_mapping = {
                "question_rewriter": "analyzing",
                "question_classifier": "classifying",
                "retrieve": "retrieving", 
                "retrieval_grader": "filtering",
                "generate_answer": "generating",
                "off_topic_response": "generating"
            }
            
            if node_name in stage_mapping and node_name not in progress_sent:
                progress_sent.add(node_name)
                stage = stage_mapping[node_name]
                yield f'data: {{"type": "progress", "stage": "{stage}"}}\n\n'
        
        # Stream content from answer generation
        elif event_type == "on_chat_model_stream":
            metadata = event.get("metadata", {})
            if metadata.get("langgraph_node") == "generate_answer":
                chunk_content = serialize_ai_message_chunk(event["data"]["chunk"])
                if chunk_content:
                    safe_content = chunk_content.replace('"', '\\"').replace("\\", "\\\\").replace("\n", "\\n")
                    yield f'data: {{"type": "content", "content": "{safe_content}"}}\n\n'
        
        # Handle off-topic response completion
        elif event_type == "on_chain_end" and event.get("name") == "off_topic_response":
            try:
                # Get the off-topic message from the state
                final_state = event["data"]["output"]
                if "messages" in final_state and final_state["messages"]:
                    last_message = final_state["messages"][-1]
                    if hasattr(last_message, 'content'):
                        off_topic_content = last_message.content
                        safe_content = off_topic_content.replace('"', '\\"').replace("\\", "\\\\").replace("\n", "\\n")
                        yield f'data: {{"type": "content", "content": "{safe_content}"}}\n\n'
            except (KeyError, AttributeError):
                # Fallback message
                safe_content = "I'm sorry! I can only answer questions related to medical and healthcare topics."
                yield f'data: {{"type": "content", "content": "{safe_content}"}}\n\n'
        
        # Capture citations from retrieval_grader completion (filtered documents)
        elif event_type == "on_chain_end" and event.get("name") == "retrieval_grader":
            try:
                documents = event["data"]["output"]["documents"]
                seen_citations = set()
                
                for doc in documents:
                    filename = doc.metadata.get('filename', 'unknown')
                    page_number = doc.metadata.get('page_number')
                    citation_key = (filename, page_number)
                    
                    if citation_key not in seen_citations:
                        seen_citations.add(citation_key)
                        citations.append({
                            "filename": filename,
                            "page_number": page_number
                        })
                
                # Send citations immediately after grading
                if citations:
                    citations_json = json.dumps(citations)
                    yield f'data: {{"type": "citations", "citations": {citations_json}}}\n\n'
                    
            except (KeyError, AttributeError):
                pass  # No documents to extract citations from
    
    # Send final event to close stream
    yield f'data: {{"type": "final"}}\n\n'

# Create FastAPI app with lifespan handler
app = FastAPI(
    title="Medical RAG API",
    description="A medical document Q&A system using RAG pipeline",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Something went wrong", "detail": str(exc)}
    )

# Root endpoint
@app.get("/")
async def root():
    """API information and available endpoints"""
    return {
        "message": "Welcome to Medical RAG API",
        "docs": "/docs",
        "status": "/status"
    }

# Status endpoint
@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Check if all RAG components are loaded and ready"""
    if not components_loaded:
        raise HTTPException(status_code=503, detail="Components not loaded yet")
    
    status_info = {
        "status": "ready",
        "components": {
            "vector_database": vector_store is not None,
            "bm25_retriever": bm25_retriever is not None,
            "ensemble_retriever": ensemble_retriever is not None,
            "llm": llm is not None,
            "cohere_reranker": cohere_reranker is not None
        }
    }
    
    try:
        if vector_store:
            status_info["vector_db_docs"] = vector_store._collection.count()
        if bm25_retriever:
            status_info["bm25_docs"] = len(bm25_retriever.docs)
    except Exception as e:
        logger.warning(f"Could not get document counts: {e}")
    
    return status_info

# ============================================================================
# CHAT ENDPOINTS
# ============================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for medical Q&A
    
    - **message**: Your medical question
    - **session_id**: Optional session ID for conversation history
    """
    if not components_loaded:
        raise HTTPException(status_code=503, detail="System components not ready")
    
    try:
        session_id = request.session_id or f"session_{int(time.time())}"
        
        input_data = {"question": HumanMessage(content=request.message)}
        
        logger.info(f"Processing question: {request.message}")
        result = rag_graph.invoke(
            input=input_data, 
            config={"configurable": {"thread_id": session_id}}
        )
        
        # Extract answer from the last AI message
        ai_message = result['messages'][-1].content
        
        # Extract citations from documents
        citations = []
        seen_citations = set()
        
        for doc in result.get('documents', []):
            filename = doc.metadata.get('filename', 'unknown')
            page_number = doc.metadata.get('page_number')
            citation_key = (filename, page_number)
            
            if citation_key not in seen_citations:
                seen_citations.add(citation_key)
                citations.append(Citation(
                    filename=filename,
                    page_number=page_number
                ))
        
        logger.info(f"Generated response with {len(citations)} citations")
        
        return ChatResponse(
            answer=ai_message,
            citations=citations,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/chat/stream/{message}")
async def chat_stream(message: str, session_id: Optional[str] = Query(None)):
    """
    Streaming chat endpoint for real-time responses
    
    - **message**: Your medical question (URL encoded)
    - **session_id**: Optional session ID for conversation history
    
    Returns Server-Sent Events with:
    - `{"type": "session_id", "session_id": "..."}`  (for new conversations)
    - `{"type": "content", "content": "text chunk"}`
    - `{"type": "final", "citations": [...]}`
    """
    if not components_loaded:
        raise HTTPException(status_code=503, detail="System components not ready")
    
    logger.info(f"Starting streaming response for: {message}")
    
    return StreamingResponse(
        generate_stream_responses(message, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback for a specific trace to LangSmith
    
    - **trace_id**: The LangSmith trace/run ID
    - **feedback_type**: 'positive' or 'negative'
    - **reason**: Optional reason for the feedback
    - **details**: Optional additional details
    """
    if not langsmith_client:
        raise HTTPException(status_code=503, detail="LangSmith client not available")
    
    try:
        # Prepare feedback data
        feedback_data = {
            "score": 1.0 if request.feedback_type == "positive" else 0.0,
            "comment": f"User feedback: {request.feedback_type}"
        }
        
        # Add reason and details if provided
        if request.reason:
            feedback_data["comment"] += f" | Reason: {request.reason}"
        if request.details:
            feedback_data["comment"] += f" | Details: {request.details}"
        
        # Submit feedback to LangSmith
        feedback_response = langsmith_client.create_feedback(
            run_id=request.trace_id,
            key="user_feedback",
            **feedback_data
        )
        
        logger.info(f"✅ Feedback submitted to LangSmith: {request.trace_id} - {request.feedback_type}")
        
        return FeedbackResponse(
            success=True,
            message="Feedback submitted successfully",
            feedback_id=str(feedback_response.id) if hasattr(feedback_response, 'id') else None
        )
        
    except Exception as e:
        logger.error(f"❌ Error submitting feedback: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to submit feedback: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)