#!/usr/bin/env python3
"""
Startup script for the TechMart Chatbot Backend
"""
import os
import sys

# Change to the backend directory
backend_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(backend_dir)

# Add backend to path
sys.path.insert(0, backend_dir)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
