# TechMart Customer Support Chatbot

A multi-turn conversational AI chatbot using RAG (Retrieval-Augmented Generation) pipeline with an open-source LLM, trained on **Doc2Dial** dialogue patterns.

## 🎯 Project Overview

This project implements a document-grounded question answering system for customer support. It features:
- **Doc2Dial Framework** - Multi-turn dialogue patterns for document-grounded conversations
- **RAG pipeline** using LangChain and ChromaDB
- **Open-source LLM** (Ollama with Mistral 7B)
- **Multi-turn conversations** with pronoun resolution, ellipsis handling, and context carryover
- **Session management** for conversation history
- **Modern React UI** with typing indicators and timestamps

## � Dataset Strategy

| Purpose | Dataset | Description |
|---------|---------|-------------|
| **Dialogue Patterns** | Doc2Dial | Open-source conversational dataset for multi-turn, document-grounded dialogues |
| **Knowledge Base** | TechMart Corpus | Custom domain-specific RAG corpus (products, orders, FAQs, troubleshooting) |

### Doc2Dial Integration

The chatbot uses dialogue patterns derived from the **Doc2Dial dataset** (Feng et al., 2020) to handle:

1. **Pronoun Resolution** - "Tell me about the laptop" → "How much does **it** cost?"
2. **Follow-up Questions** - "What about the Pro Laptop specifically?"
3. **Ellipsis Handling** - "And the warranty?" (connects to previous context)
4. **Topic Switches** - "How do I return **it** if I don't like it?"
5. **Clarification Requests** - "My device isn't working" → asks for specifics

**Reference:** Feng et al. (2020) "doc2dial: A Goal-Oriented Document-Grounded Dialogue Dataset" EMNLP 2020

## �🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| LLM | Ollama (Mistral 7B Instruct) |
| RAG Framework | LangChain |
| Vector Store | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Dialogue Patterns | Doc2Dial Framework |
| Backend | FastAPI |
| Frontend | React.js (Vite) |

## 📁 Project Structure

```
NLP_Assignment/
├── backend/
│   ├── app.py                    # FastAPI main application
│   ├── rag_pipeline.py           # RAG + Doc2Dial implementation
│   ├── doc2dial_loader.py        # Doc2Dial pattern extractor
│   ├── session_manager.py        # Session/context management
│   ├── requirements.txt          # Python dependencies
│   ├── .env                      # Environment configuration
│   └── data/
│       ├── knowledge_base/       # Domain-specific RAG corpus
│       │   └── techmart_support.md
│       └── doc2dial_cache/       # Doc2Dial dialogue patterns
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main React component
│   │   ├── App.css              # Styles
│   │   └── main.jsx             # Entry point
│   ├── index.html
│   └── package.json
├── docs/
│   └── enhancement_plan.md       # Task B documentation
├── README.md
└── SETUP.md                      # Detailed setup instructions
```

## 🚀 Getting Started

> 📖 **For detailed setup instructions, see [SETUP.md](SETUP.md)**

### Prerequisites

1. **Python 3.10+**
2. **Node.js 18+**
3. **Ollama** - Install from [ollama.ai](https://ollama.ai)

### Quick Start (3 Terminals)

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start Backend (first time setup)
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 -m uvicorn app:app --host 0.0.0.0 --port 8001

# Terminal 3: Start Frontend
cd frontend
npm install
npm run dev
```

Then open **http://localhost:5173** in your browser.

### Step 1: Start Ollama

```bash
# Start Ollama server
ollama serve

# In another terminal, pull the Mistral model
ollama pull mistral:7b-instruct
```

### Step 2: Setup Backend

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the backend server
python run_server.py
```

The backend will start at `http://localhost:8001`

### Step 3: Setup Frontend

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will start at `http://localhost:5173`

## 📖 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/api/chat` | POST | Send message and get response |
| `/api/session/new` | POST | Create new session |
| `/api/session/{id}` | GET | Get session info |
| `/api/session/{id}/history` | GET | Get conversation history |
| `/api/session/{id}/clear` | DELETE | Clear session history |
| `/api/admin/rebuild-index` | POST | Rebuild vector store |

## 💬 Usage

1. Open the frontend in your browser
2. Type your message in the input field
3. Press Enter or click Send
4. The chatbot will respond with context-aware answers

### Sample Questions

- "What products do you sell?"
- "What's the price of TechMart Pro Laptop 15?"
- "How can I return a product?"
- "What's your shipping policy?"
- "My laptop won't turn on, help!"
- "Do you offer student discounts?"

## 🔧 Configuration

Environment variables in `backend/.env`:

```env
OLLAMA_MODEL=mistral:7b-instruct
OLLAMA_BASE_URL=http://localhost:11434
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## 📊 Features

### Frontend Features
- ✅ Clean chat interface with message bubbles
- ✅ User/bot message differentiation with avatars
- ✅ Typing indicator while processing
- ✅ Timestamps for each message
- ✅ Session persistence (localStorage)
- ✅ Responsive design for mobile
- ✅ Auto-scroll to latest message

### Backend Features
- ✅ RAG pipeline with vector similarity search
- ✅ Multi-turn conversation context
- ✅ Session management with timeout
- ✅ Health check endpoint
- ✅ CORS support for frontend integration

## 📝 Dataset Source

The knowledge base is a custom-created dataset for TechMart (fictional company) containing:
- Product information (laptops, smartphones, headphones, smartwatches)
- Shipping policies and methods
- Return and refund procedures
- Warranty information
- Troubleshooting guides
- FAQs

**Source:** Custom created for this assignment (see `backend/data/knowledge_base/techmart_support.md`)

## 🎓 Academic Information

**Course:** Natural Language Processing  
**Assignment:** Document-Based Question Answering System  
**Institution:** BITS Pilani

## � Documentation

| Document | Description |
|----------|-------------|
| [SETUP.md](SETUP.md) | Detailed local setup instructions |
| [docs/DESIGN_REPORT.md](docs/DESIGN_REPORT.md) | Design choices & implementation challenges |
| [docs/enhancement_plan.md](docs/enhancement_plan.md) | Task B - Enhancement recommendations |

## �📄 License

This project is for educational purposes only.
