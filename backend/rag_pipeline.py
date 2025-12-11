"""
RAG (Retrieval-Augmented Generation) Pipeline

This module implements a document-grounded conversational AI system that combines:
1. Doc2Dial patterns for multi-turn dialogue management
2. Domain-specific TechMart knowledge base for retrieval

Dataset Strategy:
- Training Patterns: Doc2Dial (open-source conversational dataset)
- Runtime Knowledge: Custom TechMart domain corpus (products, orders, FAQs, troubleshooting)
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
KNOWLEDGE_BASE_PATH = "./data/knowledge_base"


class RAGPipeline:
    """
    RAG Pipeline for document-grounded question answering.
    
    This pipeline combines:
    - Doc2Dial dialogue patterns for multi-turn conversation handling
    - Domain-specific TechMart knowledge base for accurate retrieval
    
    Key Features:
    - Reference resolution (handles "it", "that", pronouns)
    - Follow-up question handling
    - Context carryover across conversation turns
    - Document-grounded responses
    """
    
    def __init__(self):
        """Initialize the RAG pipeline components."""
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.is_initialized = False
        # Track entities mentioned in conversation for reference resolution
        self.entity_tracker = {}
        
    def initialize(self):
        """Initialize all components of the RAG pipeline."""
        print("Initializing RAG Pipeline...")
        print("Dataset: Doc2Dial patterns + TechMart Knowledge Base")
        
        # Initialize embeddings
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize LLM
        print(f"Connecting to Ollama model: {OLLAMA_MODEL}")
        self.llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.7,
        )
        
        # Load Doc2Dial few-shot examples
        self._load_doc2dial_patterns()
        
        # Load or create vector store
        self._load_or_create_vector_store()
        
        self.is_initialized = True
        print("RAG Pipeline initialized successfully!")
    
    def _load_doc2dial_patterns(self):
        """Load Doc2Dial dialogue patterns for few-shot prompting."""
        self.dialogue_patterns = self._get_doc2dial_few_shot_examples()
        self.dialogue_guidelines = self._get_dialogue_guidelines()
        print("Loaded Doc2Dial dialogue patterns for multi-turn handling")
    
    def _get_doc2dial_few_shot_examples(self) -> str:
        """
        Few-shot examples derived from Doc2Dial dataset patterns.
        These teach the model how to handle document-grounded multi-turn dialogues.
        """
        return """
## Multi-Turn Dialogue Examples (Doc2Dial Patterns)

### Pattern 1: Follow-up Question
Customer: What are your shipping options?
Agent: We offer Standard Shipping (5-7 business days, $5.99), Express Shipping (2-3 business days, $12.99), and Next-Day Delivery ($24.99, order by 2 PM).
Customer: How much is the express one?
Agent: Express Shipping costs $12.99 and delivers in 2-3 business days.

### Pattern 2: Pronoun Resolution  
Customer: Tell me about the TechMart Pro Laptop.
Agent: The TechMart Pro Laptop 15 features a 15.6" 4K display, Intel i7 processor, 16GB RAM, 512GB SSD, and 10-hour battery life. It's priced at $1,299.
Customer: Does it come with a warranty?
Agent: Yes, the TechMart Pro Laptop 15 comes with a 1-year standard warranty covering manufacturing defects.

### Pattern 3: Ellipsis Handling
Customer: What's the price of the smartphone?
Agent: The TechMart Smartphone X is priced at $899.
Customer: And the watch?
Agent: The TechMart Smart Watch Pro is priced at $349.

### Pattern 4: Topic Switch with Reference
Customer: I'm interested in the wireless headphones.
Agent: The TechMart Wireless Headphones Pro feature 30-hour battery life, active noise cancellation, and are priced at $199.
Customer: How would I return them if I don't like them?
Agent: You can return the headphones within 30 days of purchase. Items must be unused and in original packaging for a full refund.

### Pattern 5: Clarification Request
Customer: My device isn't working.
Agent: I'd be happy to help troubleshoot. Could you tell me which TechMart device you're having issues with - is it a laptop, smartphone, headphones, or smartwatch?
Customer: It's the laptop. It won't turn on.
Agent: For a laptop that won't turn on: 1) Check the charger is connected, 2) Hold power button for 15 seconds to hard reset, 3) Try a different outlet. If still not working, contact support for warranty service.
"""
    
    def _get_dialogue_guidelines(self) -> str:
        """Guidelines for multi-turn dialogue handling based on Doc2Dial patterns."""
        return """
## Multi-Turn Dialogue Guidelines (Doc2Dial Framework)

1. **Reference Resolution**: When user says "it", "that", "this", "them", refer to the most recently discussed entity (product, policy, etc.)

2. **Follow-up Questions**: If user asks "what about X?" or "and Y?", connect to previous conversation context.

3. **Ellipsis Completion**: Complete partial questions using conversation history.
   - "And the price?" → "What is the price of [last mentioned product]?"
   - "For international?" → "What about [topic] for international [shipping/orders]?"

4. **Context Carryover**: Maintain awareness of:
   - Products or services discussed
   - User's apparent intent (buying, returning, troubleshooting, inquiring)
   - Previous questions and your answers

5. **Document Grounding**: Always base responses on the knowledge base. If information isn't available, say so clearly.

6. **Clarification**: If a question is ambiguous, ask for clarification rather than guessing.

7. **Topic Transitions**: When user switches topics but references previous context (e.g., "Can I return it?"), connect the new topic to previously discussed items.
"""
        
    def _load_documents(self) -> List[Document]:
        """Load documents from the knowledge base directory."""
        documents = []
        knowledge_base_path = Path(KNOWLEDGE_BASE_PATH)
        
        if not knowledge_base_path.exists():
            print(f"Knowledge base directory not found: {knowledge_base_path}")
            return documents
        
        # Load markdown and text files using TextLoader (more reliable)
        for pattern in ["**/*.md", "**/*.txt"]:
            for file_path in knowledge_base_path.glob(pattern):
                try:
                    loader = TextLoader(str(file_path), encoding='utf-8')
                    documents.extend(loader.load())
                    print(f"Loaded: {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        print(f"Total documents loaded: {len(documents)}")
        return documents
    
    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks for better retrieval."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def _load_or_create_vector_store(self):
        """Load existing vector store or create a new one."""
        persist_path = Path(CHROMA_PERSIST_DIRECTORY)
        
        if persist_path.exists() and any(persist_path.iterdir()):
            print("Loading existing vector store...")
            self.vector_store = Chroma(
                persist_directory=str(persist_path),
                embedding_function=self.embeddings
            )
        else:
            print("Creating new vector store...")
            documents = self._load_documents()
            
            if not documents:
                print("No documents found. Creating empty vector store.")
                self.vector_store = Chroma(
                    persist_directory=str(persist_path),
                    embedding_function=self.embeddings
                )
                return
            
            chunks = self._split_documents(documents)
            
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=str(persist_path)
            )
            print("Vector store created and persisted.")
    
    def rebuild_vector_store(self):
        """Rebuild the vector store from scratch."""
        persist_path = Path(CHROMA_PERSIST_DIRECTORY)
        
        # Clear existing vector store
        if persist_path.exists():
            import shutil
            shutil.rmtree(persist_path)
            print("Cleared existing vector store.")
        
        # Reload documents and create new vector store
        self._load_or_create_vector_store()
        print("Vector store rebuilt successfully.")
    
    def retrieve_context(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve relevant documents for a query."""
        if not self.vector_store:
            return []
        
        results = self.vector_store.similarity_search(query, k=k)
        return results
    
    def _resolve_references(self, query: str, conversation_history: List[dict]) -> str:
        """
        Resolve pronouns and references in the query based on conversation history.
        This implements Doc2Dial's reference resolution pattern.
        """
        if not conversation_history:
            return query
        
        # Check for pronouns that need resolution
        pronouns = ['it', 'this', 'that', 'them', 'they', 'its', 'their']
        query_lower = query.lower()
        
        # If query contains pronouns, try to resolve them from context
        needs_resolution = any(
            f' {pronoun} ' in f' {query_lower} ' or 
            query_lower.startswith(f'{pronoun} ') or
            query_lower.endswith(f' {pronoun}')
            for pronoun in pronouns
        )
        
        if needs_resolution:
            # Extract entities from recent conversation
            recent_entities = []
            for msg in reversed(conversation_history[-4:]):
                # Look for product names, policies mentioned
                content = msg.get('content', '')
                # Simple entity extraction - look for capitalized product names
                products = re.findall(r'TechMart [A-Za-z]+ [A-Za-z0-9]+', content)
                recent_entities.extend(products)
                
            if recent_entities:
                # Add the most recent entity to context
                self.entity_tracker['last_mentioned'] = recent_entities[0]
        
        return query
    
    def _expand_ellipsis(self, query: str, conversation_history: List[dict]) -> str:
        """
        Expand elliptical queries based on conversation context.
        Implements Doc2Dial's ellipsis handling pattern.
        
        Examples:
        - "And the price?" → "What is the price of [last product]?"
        - "For returns?" → "What about [topic] for returns?"
        """
        query_lower = query.lower().strip()
        
        # Common ellipsis patterns
        ellipsis_patterns = [
            (r'^and (?:the |what about )?(.+)\??$', 'follow_up'),
            (r'^what about (.+)\??$', 'follow_up'),
            (r'^how about (.+)\??$', 'follow_up'),
            (r'^(.+)\?$', 'possible_ellipsis')  # Short questions might be elliptical
        ]
        
        if conversation_history and len(query.split()) <= 4:
            # Short query might be elliptical - augment with context
            last_assistant_msg = None
            last_user_msg = None
            
            for msg in reversed(conversation_history):
                if msg['role'] == 'assistant' and not last_assistant_msg:
                    last_assistant_msg = msg['content']
                elif msg['role'] == 'user' and not last_user_msg:
                    last_user_msg = msg['content']
                if last_assistant_msg and last_user_msg:
                    break
            
            # If this looks like a follow-up, add context
            if last_user_msg and last_assistant_msg:
                self.entity_tracker['last_topic'] = last_user_msg
                
        return query

    def generate_response(
        self,
        query: str,
        conversation_history: List[dict] = None,
        context_documents: List[Document] = None,
        additional_context: str = ""
    ) -> str:
        """
        Generate a response using RAG with Doc2Dial multi-turn patterns.
        
        This method implements document-grounded dialogue as per Doc2Dial:
        1. Resolves references (pronouns, ellipsis)
        2. Retrieves relevant context from TechMart knowledge base
        3. Generates response grounded in retrieved documents
        
        Args:
            query: The user's question
            conversation_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
            context_documents: Pre-retrieved documents (optional)
            additional_context: Additional context to include (e.g., order details)
        
        Returns:
            Generated response string
        """
        if not self.is_initialized:
            self.initialize()
        
        # Apply Doc2Dial patterns for reference resolution
        resolved_query = self._resolve_references(query, conversation_history or [])
        expanded_query = self._expand_ellipsis(resolved_query, conversation_history or [])
        
        # Build enhanced query for retrieval using conversation context
        retrieval_query = query
        if conversation_history and len(conversation_history) > 0:
            # Include recent context for better retrieval
            recent_context = " ".join([
                msg['content'] for msg in conversation_history[-2:]
            ])
            retrieval_query = f"{recent_context} {query}"
        
        # Retrieve relevant context from TechMart knowledge base
        if context_documents is None:
            context_documents = self.retrieve_context(retrieval_query)
        
        # Format context
        context_text = "\n\n".join([
            f"[Document {i+1}]\n{doc.page_content}"
            for i, doc in enumerate(context_documents)
        ]) if context_documents else "No relevant context found in knowledge base."
        
        # Add additional context (e.g., order details) if provided
        if additional_context:
            context_text = f"{context_text}\n\n[Order Information]\n{additional_context}"
        
        # Format conversation history
        history_text = ""
        if conversation_history:
            history_text = "\n".join([
                f"{'Customer' if msg['role'] == 'user' else 'Support Agent'}: {msg['content']}"
                for msg in conversation_history[-6:]  # Keep last 6 messages for context
            ])
        
        # Create prompt with Doc2Dial patterns
        system_prompt = """You are a helpful customer support agent for TechMart, an electronics retailer.
You are trained on document-grounded dialogue patterns from the Doc2Dial dataset.

{dialogue_guidelines}

{few_shot_examples}

---
## Knowledge Base Context (TechMart Domain Corpus)
The following information is retrieved from our knowledge base:

{context}

---
## Current Conversation
{history}

---
## Current Customer Query
Customer: {query}

## Instructions
1. Use ONLY information from the Knowledge Base Context above
2. Apply the multi-turn dialogue patterns learned from Doc2Dial
3. If the customer uses pronouns (it, this, that), resolve them based on conversation history
4. If the question is a follow-up, connect it to previous context
5. Be helpful, accurate, and concise

Support Agent:"""

        prompt = system_prompt.format(
            dialogue_guidelines=self.dialogue_guidelines,
            few_shot_examples=self.dialogue_patterns,
            context=context_text,
            history=history_text if history_text else "No previous conversation.",
            query=query
        )
        
        # Generate response
        try:
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again or contact our support team directly."


# Create a singleton instance
rag_pipeline = RAGPipeline()


def get_rag_pipeline() -> RAGPipeline:
    """Get the RAG pipeline instance."""
    if not rag_pipeline.is_initialized:
        rag_pipeline.initialize()
    return rag_pipeline
