# 🚀 TechMart Chatbot - Local Setup Guide

Complete instructions for running the Multi-Turn Conversational AI Chatbot locally.

---

## 📋 Prerequisites

Before starting, ensure you have the following installed:

| Requirement | Version | Download Link |
|-------------|---------|---------------|
| **Python** | 3.10 or higher | [python.org](https://www.python.org/downloads/) |
| **Node.js** | 18.x or higher | [nodejs.org](https://nodejs.org/) |
| **Ollama** | Latest | [ollama.ai](https://ollama.ai/) |
| **Git** | Latest | [git-scm.com](https://git-scm.com/) |

### Verify Installation

```bash
# Check Python version
python3 --version   # Should be 3.10+

# Check Node.js version
node --version      # Should be 18+

# Check npm version
npm --version

# Check Ollama
ollama --version
```

---

## 📥 Step 1: Clone the Repository

```bash
git clone https://github.com/vishalvishal099/Multi_TurnConversational_AI_Chatbot.git
cd Multi_TurnConversational_AI_Chatbot
```

---

## 🤖 Step 2: Setup Ollama (LLM Server)

### 2.1 Start Ollama Server

Open a **new terminal** and run:

```bash
ollama serve
```

> ⚠️ **Keep this terminal running** - Ollama must stay active for the chatbot to work.

### 2.2 Pull the Mistral Model

In a **different terminal**, download the LLM:

```bash
ollama pull mistral:7b-instruct
```

> ⏳ This may take 5-10 minutes depending on your internet speed (~4GB download).

### 2.3 Verify Model Installation

```bash
ollama list
```

You should see:
```
NAME                    SIZE
mistral:7b-instruct     4.1 GB
```

---

## 🐍 Step 3: Setup Backend (Python/FastAPI)

### 3.1 Navigate to Backend Directory

```bash
cd backend
```

### 3.2 Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3.3 Install Python Dependencies

```bash
pip install -r requirements.txt
```

> ⏳ This will install FastAPI, LangChain, ChromaDB, and other dependencies.

### 3.4 Configure Environment (Optional)

Create/edit `.env` file in the `backend` directory:

```bash
# backend/.env
OLLAMA_MODEL=mistral:7b-instruct
OLLAMA_BASE_URL=http://localhost:11434
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### 3.5 Start the Backend Server

```bash
python3 -m uvicorn app:app --host 0.0.0.0 --port 8001
```

> ⏳ **First startup takes 30-60 seconds** - The server initializes embeddings and builds the vector store.

### 3.6 Verify Backend is Running

Open a new terminal and run:

```bash
curl http://localhost:8001/health
```

Expected response:
```json
{
  "status": "healthy",
  "rag_initialized": true,
  "active_sessions": 0,
  "timestamp": "2025-12-11T..."
}
```

---

## ⚛️ Step 4: Setup Frontend (React/Vite)

### 4.1 Navigate to Frontend Directory

Open a **new terminal**:

```bash
cd frontend
```

### 4.2 Install Node Dependencies

```bash
npm install
```

### 4.3 Start the Development Server

```bash
npm run dev
```

### 4.4 Access the Application

Open your browser and go to:

```
http://localhost:5173
```

---

## ✅ Step 5: Verify Everything is Working

You should now have **3 terminals** running:

| Terminal | Service | URL |
|----------|---------|-----|
| 1 | Ollama LLM Server | `http://localhost:11434` |
| 2 | Backend (FastAPI) | `http://localhost:8001` |
| 3 | Frontend (React) | `http://localhost:5173` |

### Test the Chatbot

1. Open `http://localhost:5173` in your browser
2. Type: **"What products do you sell?"**
3. Press Enter and wait for the response

---

## 🧪 Quick Test Commands

### Test Backend API Directly

```bash
# Health check
curl http://localhost:8001/health

# Send a test message
curl -X POST http://localhost:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What products do you have?", "session_id": "test-123"}'
```

### Test Order Tracking

```bash
# Get order by ID
curl http://localhost:8001/api/orders/TM-2024-001234

# Track by tracking number
curl http://localhost:8001/api/orders/tracking/1Z999AA10123456784
```

---

## 🛠️ Troubleshooting

### Issue: "Connection Refused" on Backend

**Cause:** Backend not running or wrong port

**Solution:**
```bash
# Check if port 8001 is in use
lsof -i :8001

# Kill any existing process and restart
pkill -f "uvicorn app:app"
cd backend && python3 -m uvicorn app:app --host 0.0.0.0 --port 8001
```

### Issue: "Ollama not found" Error

**Cause:** Ollama server not running

**Solution:**
```bash
# Start Ollama in a separate terminal
ollama serve
```

### Issue: Slow First Response

**Cause:** First query loads the LLM into memory

**Solution:** Wait 10-20 seconds for the first response. Subsequent responses will be faster.

### Issue: "Model not found"

**Cause:** Mistral model not downloaded

**Solution:**
```bash
ollama pull mistral:7b-instruct
```

### Issue: Frontend Shows "Connecting..."

**Cause:** Backend not running or CORS issue

**Solution:**
1. Ensure backend is running on port 8001
2. Check browser console for errors
3. Restart both backend and frontend

### Issue: ChromaDB Errors

**Cause:** Corrupted vector store

**Solution:**
```bash
# Delete and rebuild vector store
rm -rf backend/data/chroma_db
# Restart backend (it will rebuild automatically)
```

---

## 📂 Project Structure

```
Multi_TurnConversational_AI_Chatbot/
├── backend/
│   ├── app.py                 # FastAPI main application
│   ├── rag_pipeline.py        # RAG + Doc2Dial implementation
│   ├── doc2dial_loader.py     # Dialogue pattern extractor
│   ├── session_manager.py     # Session management
│   ├── order_manager.py       # Order tracking system
│   ├── requirements.txt       # Python dependencies
│   ├── .env                   # Environment config
│   └── data/
│       ├── knowledge_base/    # TechMart knowledge documents
│       ├── orders/            # Sample order data
│       └── chroma_db/         # Vector store (auto-generated)
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # Main React component
│   │   ├── App.css           # Styles
│   │   └── main.jsx          # Entry point
│   ├── package.json          # Node dependencies
│   └── vite.config.js        # Vite configuration
├── docs/
│   └── enhancement_plan.md   # Task B documentation
├── README.md                 # Project overview
└── SETUP.md                  # This file
```

---

## 🔄 Starting/Stopping Services

### Start All Services (Quick Reference)

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Start Backend
cd backend
source venv/bin/activate  # if using venv
python3 -m uvicorn app:app --host 0.0.0.0 --port 8001

# Terminal 3: Start Frontend
cd frontend
npm run dev
```

### Stop All Services

```bash
# Stop Backend
pkill -f "uvicorn app:app"

# Stop Frontend
# Press Ctrl+C in the frontend terminal

# Stop Ollama
# Press Ctrl+C in the Ollama terminal
```

---

## 📊 API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/api/chat` | POST | Send chat message |
| `/api/session/new` | POST | Create new session |
| `/api/session/{id}` | GET | Get session info |
| `/api/session/{id}/history` | GET | Get conversation history |
| `/api/orders/{order_id}` | GET | Get order by ID |
| `/api/orders/tracking/{tracking}` | GET | Track by tracking number |

---

