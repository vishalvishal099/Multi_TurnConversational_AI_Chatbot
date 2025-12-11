"""
Session Manager for Multi-Turn Conversations
Handles session creation, context tracking, and conversation history management.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from threading import Lock
import json


@dataclass
class Message:
    """Represents a single message in the conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        """Convert message to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        """Create message from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now()
        )


@dataclass
class Session:
    """Represents a chat session with conversation history."""
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    messages: List[Message] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    def add_message(self, role: str, content: str) -> Message:
        """Add a new message to the session."""
        message = Message(role=role, content=content)
        self.messages.append(message)
        self.last_activity = datetime.now()
        return message
    
    def get_history(self, max_messages: Optional[int] = None) -> List[dict]:
        """Get conversation history as a list of dictionaries."""
        messages = self.messages
        if max_messages:
            messages = messages[-max_messages:]
        return [msg.to_dict() for msg in messages]
    
    def get_context_window(self, window_size: int = 10) -> List[dict]:
        """Get recent messages for context."""
        recent = self.messages[-window_size:] if len(self.messages) > window_size else self.messages
        return [{"role": msg.role, "content": msg.content} for msg in recent]
    
    def clear_history(self):
        """Clear all messages from the session."""
        self.messages = []
        self.last_activity = datetime.now()
    
    def to_dict(self) -> dict:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": self.metadata
        }
    
    def is_expired(self, timeout_minutes: int = 60) -> bool:
        """Check if the session has expired."""
        expiry_time = self.last_activity + timedelta(minutes=timeout_minutes)
        return datetime.now() > expiry_time


class SessionManager:
    """
    Manages multiple chat sessions with thread-safe operations.
    Handles session creation, retrieval, and cleanup.
    """
    
    def __init__(self, session_timeout_minutes: int = 60):
        """
        Initialize the session manager.
        
        Args:
            session_timeout_minutes: Minutes of inactivity before session expires
        """
        self._sessions: Dict[str, Session] = {}
        self._lock = Lock()
        self._timeout_minutes = session_timeout_minutes
    
    def create_session(self, metadata: Optional[dict] = None) -> Session:
        """
        Create a new chat session.
        
        Args:
            metadata: Optional metadata to attach to the session
        
        Returns:
            New Session object
        """
        with self._lock:
            session_id = str(uuid.uuid4())
            session = Session(
                session_id=session_id,
                metadata=metadata or {}
            )
            self._sessions[session_id] = session
            return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """
        Get a session by ID.
        
        Args:
            session_id: The session ID to look up
        
        Returns:
            Session object or None if not found
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session and not session.is_expired(self._timeout_minutes):
                return session
            elif session and session.is_expired(self._timeout_minutes):
                # Clean up expired session
                del self._sessions[session_id]
            return None
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> Session:
        """
        Get existing session or create a new one.
        
        Args:
            session_id: Optional session ID to look up
        
        Returns:
            Session object (existing or new)
        """
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        return self.create_session()
    
    def add_message(self, session_id: str, role: str, content: str) -> Optional[Message]:
        """
        Add a message to a session.
        
        Args:
            session_id: The session ID
            role: Message role ('user' or 'assistant')
            content: Message content
        
        Returns:
            Message object or None if session not found
        """
        session = self.get_session(session_id)
        if session:
            return session.add_message(role, content)
        return None
    
    def get_conversation_history(
        self,
        session_id: str,
        max_messages: Optional[int] = None
    ) -> List[dict]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: The session ID
            max_messages: Maximum number of messages to return
        
        Returns:
            List of message dictionaries
        """
        session = self.get_session(session_id)
        if session:
            return session.get_history(max_messages)
        return []
    
    def get_context(self, session_id: str, window_size: int = 10) -> List[dict]:
        """
        Get recent context for RAG pipeline.
        
        Args:
            session_id: The session ID
            window_size: Number of recent messages to include
        
        Returns:
            List of message dictionaries for context
        """
        session = self.get_session(session_id)
        if session:
            return session.get_context_window(window_size)
        return []
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear conversation history for a session.
        
        Args:
            session_id: The session ID
        
        Returns:
            True if successful, False if session not found
        """
        session = self.get_session(session_id)
        if session:
            session.clear_history()
            return True
        return False
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session entirely.
        
        Args:
            session_id: The session ID
        
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False
    
    def cleanup_expired_sessions(self) -> int:
        """
        Remove all expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        with self._lock:
            expired = [
                sid for sid, session in self._sessions.items()
                if session.is_expired(self._timeout_minutes)
            ]
            for sid in expired:
                del self._sessions[sid]
            return len(expired)
    
    def get_active_session_count(self) -> int:
        """Get the number of active sessions."""
        with self._lock:
            return len(self._sessions)
    
    def get_session_info(self, session_id: str) -> Optional[dict]:
        """
        Get session information without full message history.
        
        Args:
            session_id: The session ID
        
        Returns:
            Session info dictionary or None
        """
        session = self.get_session(session_id)
        if session:
            return {
                "session_id": session.session_id,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "message_count": len(session.messages),
                "metadata": session.metadata
            }
        return None


# Create a singleton instance
session_manager = SessionManager(session_timeout_minutes=60)


def get_session_manager() -> SessionManager:
    """Get the session manager instance."""
    return session_manager
