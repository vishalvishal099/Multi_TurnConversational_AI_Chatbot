"""
FastAPI Backend for Multi-Turn Conversational AI Chatbot
Main application file with API endpoints for chat functionality.
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from session_manager import get_session_manager, SessionManager
from rag_pipeline import get_rag_pipeline, RAGPipeline
from order_manager import get_order_manager, OrderManager


# Pydantic Models for API
class MessageRequest(BaseModel):
    """Request model for sending a message."""
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for continuing conversation")


class MessageResponse(BaseModel):
    """Response model for chat messages."""
    response: str = Field(..., description="Bot response")
    session_id: str = Field(..., description="Session ID")
    timestamp: str = Field(..., description="Response timestamp")


class SessionResponse(BaseModel):
    """Response model for session information."""
    session_id: str
    created_at: str
    message_count: int


class ConversationHistory(BaseModel):
    """Response model for conversation history."""
    session_id: str
    messages: List[dict]
    total_messages: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    rag_initialized: bool
    active_sessions: int
    timestamp: str


class OrderItem(BaseModel):
    """Model for an order item."""
    name: str
    quantity: int
    price: float


class OrderResponse(BaseModel):
    """Response model for order details."""
    order_id: str
    customer_name: str
    email: str
    status: str
    order_date: str
    items: List[OrderItem]
    shipping_address: dict
    tracking_number: Optional[str] = None
    carrier: Optional[str] = None
    estimated_delivery: Optional[str] = None
    total_amount: float


class OrderTrackingResponse(BaseModel):
    """Response model for order tracking."""
    order_id: str
    status: str
    tracking_number: Optional[str]
    carrier: Optional[str]
    estimated_delivery: Optional[str]
    shipping_address: dict


class OrderSearchResponse(BaseModel):
    """Response model for order search results."""
    orders: List[OrderResponse]
    total_count: int


# Global instances
rag_pipeline: Optional[RAGPipeline] = None
session_manager: Optional[SessionManager] = None
order_manager: Optional[OrderManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Initializes RAG pipeline on startup.
    """
    global rag_pipeline, session_manager, order_manager
    
    print("Starting up the application...")
    
    # Initialize session manager
    session_manager = get_session_manager()
    print("Session manager initialized.")
    
    # Initialize order manager
    order_manager = get_order_manager()
    print("Order manager initialized.")
    
    # Initialize RAG pipeline (this may take a while on first run)
    print("Initializing RAG pipeline (this may take a few minutes on first run)...")
    rag_pipeline = get_rag_pipeline()
    print("RAG pipeline ready!")
    
    yield
    
    # Cleanup on shutdown
    print("Shutting down the application...")


def _extract_order_context(message: str, conversation_history: List[dict]) -> str:
    """
    Extract order-related context from the message and conversation history.
    If an order ID or tracking number is mentioned, fetch the order details.
    
    Args:
        message: Current user message
        conversation_history: Recent conversation history
        
    Returns:
        Order context string to inject into RAG, empty string if no order found
    """
    import re
    
    # Pattern for TechMart order IDs: TM-YYYY-XXXXXX
    order_pattern = r'TM-\d{4}-\d{6}'
    # Pattern for tracking numbers (various carriers)
    tracking_patterns = [
        r'1Z[A-Z0-9]{16}',  # UPS
        r'\d{20,22}',  # USPS, FedEx
        r'[A-Z]{2}\d{9}[A-Z]{2}',  # International
    ]
    
    order_context_parts = []
    
    # Check current message for order ID
    order_matches = re.findall(order_pattern, message.upper())
    for order_id in order_matches:
        order_info = order_manager.format_order_for_chat(order_id)
        if order_info:
            order_context_parts.append(f"Order Information:\n{order_info}")
    
    # Check current message for tracking numbers
    for pattern in tracking_patterns:
        tracking_matches = re.findall(pattern, message.upper())
        for tracking_num in tracking_matches:
            order = order_manager.get_order_by_tracking(tracking_num)
            if order:
                order_info = order_manager.format_order_for_chat(order['order_id'])
                if order_info:
                    order_context_parts.append(f"Order Information (from tracking):\n{order_info}")
    
    # Also check recent conversation history for order context
    if not order_context_parts:
        for msg in conversation_history[-4:]:  # Check last 4 messages
            content = msg.get('content', '')
            order_matches = re.findall(order_pattern, content.upper())
            for order_id in order_matches:
                order_info = order_manager.format_order_for_chat(order_id)
                if order_info:
                    order_context_parts.append(f"Previously mentioned order:\n{order_info}")
                    break
            if order_context_parts:
                break
    
    return "\n\n".join(order_context_parts)


# Create FastAPI app
app = FastAPI(
    title="TechMart Customer Support Chatbot",
    description="Multi-turn conversational AI chatbot with RAG-based responses",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "TechMart Customer Support Chatbot API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_check": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        rag_initialized=rag_pipeline is not None and rag_pipeline.is_initialized,
        active_sessions=session_manager.get_active_session_count() if session_manager else 0,
        timestamp=datetime.now().isoformat()
    )


@app.post("/api/chat", response_model=MessageResponse, tags=["Chat"])
async def send_message(request: MessageRequest):
    """
    Send a message and receive a response.
    
    - Creates a new session if session_id is not provided
    - Maintains conversation context for follow-up questions
    - Returns bot response with session information
    """
    if not rag_pipeline or not rag_pipeline.is_initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline is not initialized yet. Please try again in a moment."
        )
    
    # Get or create session
    session = session_manager.get_or_create_session(request.session_id)
    
    # Add user message to history
    session.add_message("user", request.message)
    
    # Get conversation context
    conversation_context = session.get_context_window(window_size=6)
    
    # Check for order-related queries and inject order data
    order_context = ""
    if order_manager:
        order_context = _extract_order_context(request.message, conversation_context)
    
    # Generate response using RAG pipeline
    try:
        response = rag_pipeline.generate_response(
            query=request.message,
            conversation_history=conversation_context[:-1],  # Exclude the current message
            additional_context=order_context
        )
    except Exception as e:
        print(f"Error generating response: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate response. Please try again."
        )
    
    # Add bot response to history
    session.add_message("assistant", response)
    
    return MessageResponse(
        response=response,
        session_id=session.session_id,
        timestamp=datetime.now().isoformat()
    )


@app.post("/api/session/new", response_model=SessionResponse, tags=["Session"])
async def create_new_session():
    """Create a new chat session."""
    session = session_manager.create_session()
    return SessionResponse(
        session_id=session.session_id,
        created_at=session.created_at.isoformat(),
        message_count=0
    )


@app.get("/api/session/{session_id}", response_model=SessionResponse, tags=["Session"])
async def get_session_info(session_id: str):
    """Get information about a specific session."""
    info = session_manager.get_session_info(session_id)
    if not info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or expired"
        )
    return SessionResponse(
        session_id=info["session_id"],
        created_at=info["created_at"],
        message_count=info["message_count"]
    )


@app.get("/api/session/{session_id}/history", response_model=ConversationHistory, tags=["Session"])
async def get_conversation_history(session_id: str, limit: Optional[int] = None):
    """
    Get conversation history for a session.
    
    - limit: Optional maximum number of messages to return
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or expired"
        )
    
    messages = session.get_history(max_messages=limit)
    return ConversationHistory(
        session_id=session_id,
        messages=messages,
        total_messages=len(session.messages)
    )


@app.delete("/api/session/{session_id}", tags=["Session"])
async def delete_session(session_id: str):
    """Delete a session and its conversation history."""
    success = session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    return {"message": "Session deleted successfully", "session_id": session_id}


@app.delete("/api/session/{session_id}/clear", tags=["Session"])
async def clear_session_history(session_id: str):
    """Clear conversation history for a session while keeping the session active."""
    success = session_manager.clear_session(session_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or expired"
        )
    return {"message": "Conversation history cleared", "session_id": session_id}


@app.post("/api/admin/rebuild-index", tags=["Admin"])
async def rebuild_vector_index():
    """
    Rebuild the vector store index from knowledge base documents.
    Use this endpoint after updating the knowledge base.
    """
    if not rag_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline not available"
        )
    
    try:
        rag_pipeline.rebuild_vector_store()
        return {"message": "Vector store rebuilt successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rebuild vector store: {str(e)}"
        )


# ==================== Order Management Endpoints ====================

@app.get("/api/orders/{order_id}", response_model=OrderResponse, tags=["Orders"])
async def get_order_by_id(order_id: str):
    """
    Get order details by order ID.
    Order IDs are in format: TM-YYYY-XXXXXX (e.g., TM-2024-001234)
    """
    if not order_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Order manager not available"
        )
    
    order = order_manager.get_order(order_id)
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found"
        )
    
    # Calculate total amount
    total = sum(item["price"] * item["quantity"] for item in order["items"])
    
    return OrderResponse(
        order_id=order["order_id"],
        customer_name=order["customer_name"],
        email=order.get("customer_email", ""),
        status=order["status"],
        order_date=order["order_date"],
        items=[OrderItem(**item) for item in order["items"]],
        shipping_address=order["shipping_address"],
        tracking_number=order.get("tracking_number"),
        carrier=order.get("carrier"),
        estimated_delivery=order.get("estimated_delivery"),
        total_amount=total
    )


@app.get("/api/orders/tracking/{tracking_number}", response_model=OrderTrackingResponse, tags=["Orders"])
async def get_order_by_tracking(tracking_number: str):
    """
    Get order tracking information by tracking number.
    """
    if not order_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Order manager not available"
        )
    
    order = order_manager.get_order_by_tracking(tracking_number)
    if not order:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order with tracking number {tracking_number} not found"
        )
    
    return OrderTrackingResponse(
        order_id=order["order_id"],
        status=order["status"],
        tracking_number=order.get("tracking_number"),
        carrier=order.get("carrier"),
        estimated_delivery=order.get("estimated_delivery"),
        shipping_address=order["shipping_address"]
    )


@app.get("/api/orders/email/{email}", response_model=OrderSearchResponse, tags=["Orders"])
async def get_orders_by_email(email: str):
    """
    Get all orders for a customer by email address.
    """
    if not order_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Order manager not available"
        )
    
    orders = order_manager.get_orders_by_email(email)
    
    order_responses = []
    for order in orders:
        total = sum(item["price"] * item["quantity"] for item in order["items"])
        order_responses.append(OrderResponse(
            order_id=order["order_id"],
            customer_name=order["customer_name"],
            email=order.get("customer_email", ""),
            status=order["status"],
            order_date=order["order_date"],
            items=[OrderItem(**item) for item in order["items"]],
            shipping_address=order["shipping_address"],
            tracking_number=order.get("tracking_number"),
            carrier=order.get("carrier"),
            estimated_delivery=order.get("estimated_delivery"),
            total_amount=total
        ))
    
    return OrderSearchResponse(
        orders=order_responses,
        total_count=len(order_responses)
    )


@app.get("/api/orders/search/{query}", tags=["Orders"])
async def search_orders(query: str):
    """
    Search orders by order ID, tracking number, customer name, or email.
    Returns a list of matching orders.
    """
    if not order_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Order manager not available"
        )
    
    orders = order_manager.search_orders(query)
    
    results = []
    for order in orders:
        total = sum(item["price"] * item["quantity"] for item in order["items"])
        results.append({
            "order_id": order["order_id"],
            "customer_name": order["customer_name"],
            "email": order["email"],
            "status": order["status"],
            "order_date": order["order_date"],
            "total_amount": total
        })
    
    return {"results": results, "count": len(results)}


@app.get("/api/orders/{order_id}/status", tags=["Orders"])
async def get_order_status_summary(order_id: str):
    """
    Get a human-readable status summary for an order.
    Useful for chatbot responses.
    """
    if not order_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Order manager not available"
        )
    
    summary = order_manager.get_order_status_summary(order_id)
    if not summary:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found"
        )
    
    return summary


@app.get("/api/orders/{order_id}/chat-format", tags=["Orders"])
async def get_order_for_chat(order_id: str):
    """
    Get order details formatted for chat responses.
    Returns a nicely formatted string suitable for chatbot use.
    """
    if not order_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Order manager not available"
        )
    
    formatted = order_manager.format_order_for_chat(order_id)
    if not formatted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Order {order_id} not found"
        )
    
    return {"order_id": order_id, "formatted_response": formatted}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
