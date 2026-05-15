"""
Chatbot Module for Pearson Specter Litt Legal Document RAG System
Handles conversational Q&A with multi-turn support and session management
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
from enum import Enum


class MessageRole(str, Enum):
    """Message role enum"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message:
    """Represents a single message in the conversation"""

    def __init__(self, role: str, content: str, sources: Optional[List[Dict]] = None):
        self.role = role
        self.content = content
        self.sources = sources or []
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """Convert message to dictionary"""
        return {
            "role": self.role,
            "content": self.content,
            "sources": self.sources,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Message":
        """Create message from dictionary"""
        return cls(
            role=data.get("role"),
            content=data.get("content"),
            sources=data.get("sources", [])
        )


class ChatSession:
    """Represents a chat session with conversation history"""

    def __init__(self, session_id: str, title: str = "New Chat"):
        self.session_id = session_id
        self.title = title
        self.messages: List[Message] = []
        self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
        self.metadata = {
            "topic": None,
            "question_count": 0,
            "document_count": 0
        }

    def add_message(self, message: Message) -> None:
        """Add a message to the session"""
        self.messages.append(message)
        self.updated_at = datetime.now().isoformat()

        if message.role == MessageRole.USER:
            self.metadata["question_count"] += 1

    def add_user_message(self, content: str) -> Message:
        """Add a user message to the session"""
        message = Message(role=MessageRole.USER, content=content)
        self.add_message(message)
        return message

    def add_assistant_message(self, content: str, sources: Optional[List[Dict]] = None) -> Message:
        """Add an assistant message to the session"""
        message = Message(role=MessageRole.ASSISTANT, content=content, sources=sources)
        self.add_message(message)
        return message

    def get_conversation_history(self, include_sources: bool = True) -> List[Dict]:
        """Get conversation history in a structured format"""
        history = []
        for msg in self.messages:
            msg_dict = msg.to_dict()
            if not include_sources:
                msg_dict.pop("sources", None)
            history.append(msg_dict)
        return history

    def get_last_user_message(self) -> Optional[str]:
        """Get the last user message in the conversation"""
        for msg in reversed(self.messages):
            if msg.role == MessageRole.USER:
                return msg.content
        return None

    def get_context_for_llm(self, max_messages: int = 10) -> List[Dict]:
        """
        Get conversation context formatted for LLM
        Includes recent message history for context awareness
        """
        # Get last N messages
        recent_messages = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages

        context = []
        for msg in recent_messages:
            context.append({
                "role": msg.role,
                "content": msg.content
            })

        return context

    def to_dict(self) -> Dict:
        """Convert session to dictionary"""
        return {
            "session_id": self.session_id,
            "title": self.title,
            "messages": self.get_conversation_history(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ChatSession":
        """Create session from dictionary"""
        session = cls(
            session_id=data.get("session_id"),
            title=data.get("title", "Loaded Chat")
        )
        session.created_at = data.get("created_at")
        session.updated_at = data.get("updated_at")
        session.metadata = data.get("metadata", {})

        # Restore messages
        for msg_data in data.get("messages", []):
            session.add_message(Message.from_dict(msg_data))

        return session


class QuestionAnswerPair:
    """Represents a Q&A pair from the conversation"""

    def __init__(
        self,
        question: str,
        answer: str,
        sources: List[Dict],
        confidence: float = 0.0
    ):
        self.question = question
        self.answer = answer
        self.sources = sources
        self.confidence = confidence
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        """Convert Q&A pair to dictionary"""
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
            "timestamp": self.timestamp
        }


class ChatSessionManager:
    """Manages multiple chat sessions"""

    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}

    def create_session(self, session_id: str, title: str = "New Chat") -> ChatSession:
        """Create a new chat session"""
        session = ChatSession(session_id=session_id, title=title)
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get an existing session"""
        return self.sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def list_sessions(self) -> List[Dict]:
        """List all sessions with metadata"""
        return [
            {
                "session_id": session.session_id,
                "title": session.title,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "message_count": len(session.messages),
                "question_count": session.metadata.get("question_count", 0)
            }
            for session in self.sessions.values()
        ]

    def save_session(self, session_id: str, filepath: str) -> bool:
        """Save session to file"""
        session = self.get_session(session_id)
        if not session:
            return False

        try:
            with open(filepath, 'w') as f:
                json.dump(session.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving session: {e}")
            return False

    def load_session(self, filepath: str, session_id: str) -> Optional[ChatSession]:
        """Load session from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            session = ChatSession.from_dict(data)
            self.sessions[session_id] = session
            return session
        except Exception as e:
            print(f"Error loading session: {e}")
            return None


class ChatbotConfig:
    """Configuration for the chatbot"""

    def __init__(
        self,
        system_prompt: str = None,
        max_context_messages: int = 10,
        temperature: float = 0.3,
        max_tokens: int = 500,
        top_k_results: int = 5,
        enable_source_citations: bool = True
    ):
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.max_context_messages = max_context_messages
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k_results = top_k_results
        self.enable_source_citations = enable_source_citations

    def _default_system_prompt(self) -> str:
        """Return the default system prompt"""
        return """You are a knowledgeable legal assistant specializing in Bangladesh law.
Your role is to help users understand legal documents and answer their questions about cases, statutes, and legal procedures.

Guidelines:
- Answer questions based ONLY on the provided legal document context
- If the context doesn't contain relevant information, say so clearly
- Cite specific sections, case names, or legal provisions when applicable
- Provide accurate, detailed explanations
- Be conversational but professional
- Reference source documents using the format [Source X: filename:chunk_number]
- Ask clarifying questions if the user's question is ambiguous
- Maintain context from previous messages in the conversation
- Always prioritize accuracy over providing an answer"""


class Chatbot:
    """Main chatbot class for handling conversations"""

    def __init__(self, config: ChatbotConfig = None):
        self.config = config or ChatbotConfig()
        self.session_manager = ChatSessionManager()
        self.current_session: Optional[ChatSession] = None

    def start_new_session(self, session_id: str, title: str = "New Chat") -> ChatSession:
        """Start a new chat session"""
        session = self.session_manager.create_session(session_id, title)
        self.current_session = session
        return session

    def load_session(self, session_id: str) -> Optional[ChatSession]:
        """Load an existing session"""
        session = self.session_manager.get_session(session_id)
        if session:
            self.current_session = session
        return session

    def get_current_session(self) -> Optional[ChatSession]:
        """Get the current active session"""
        return self.current_session

    def add_user_message(self, content: str) -> Optional[Message]:
        """Add a user message to the current session"""
        if not self.current_session:
            return None
        return self.current_session.add_user_message(content)

    def add_assistant_response(self, content: str, sources: Optional[List[Dict]] = None) -> Optional[Message]:
        """Add an assistant response to the current session"""
        if not self.current_session:
            return None
        return self.current_session.add_assistant_message(content, sources)

    def format_response_with_sources(self, response: str, sources: List[Dict]) -> str:
        """Format a response with source citations"""
        if not self.config.enable_source_citations or not sources:
            return response

        # Add source citations at the end
        citations = "\n\n**Sources:**\n"
        for source in sources:
            citation = f"- [{source.get('id', '?')}] {source.get('source_location', 'Unknown')}"
            if source.get('case_name'):
                citation += f" ({source['case_name']})"
            citations += citation + "\n"

        return response + citations

    def prepare_context_for_llm(self) -> List[Dict]:
        """Prepare conversation context for LLM"""
        if not self.current_session:
            return []

        return self.current_session.get_context_for_llm(
            max_messages=self.config.max_context_messages
        )

    def get_session_summary(self) -> Optional[Dict]:
        """Get a summary of the current session"""
        if not self.current_session:
            return None

        session = self.current_session
        return {
            "session_id": session.session_id,
            "title": session.title,
            "message_count": len(session.messages),
            "question_count": session.metadata.get("question_count", 0),
            "created_at": session.created_at,
            "updated_at": session.updated_at
        }

    def export_qa_pairs(self) -> List[QuestionAnswerPair]:
        """Export all Q&A pairs from current session"""
        if not self.current_session:
            return []

        qa_pairs = []
        for i, msg in enumerate(self.current_session.messages):
            if msg.role == MessageRole.USER and i + 1 < len(self.current_session.messages):
                next_msg = self.current_session.messages[i + 1]
                if next_msg.role == MessageRole.ASSISTANT:
                    qa_pair = QuestionAnswerPair(
                        question=msg.content,
                        answer=next_msg.content,
                        sources=next_msg.sources
                    )
                    qa_pairs.append(qa_pair)

        return qa_pairs
