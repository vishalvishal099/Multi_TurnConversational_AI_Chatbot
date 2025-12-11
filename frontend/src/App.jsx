import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import './App.css'

const API_BASE_URL = 'http://localhost:8001'

function App() {
  const [messages, setMessages] = useState([])
  const [inputMessage, setInputMessage] = useState('')
  const [sessionId, setSessionId] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Check API health on mount
  useEffect(() => {
    checkHealth()
    // Load session from localStorage
    const savedSession = localStorage.getItem('chatSessionId')
    if (savedSession) {
      setSessionId(savedSession)
      loadConversationHistory(savedSession)
    }
  }, [])

  const checkHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`)
      setIsConnected(response.data.status === 'healthy' && response.data.rag_initialized)
    } catch (error) {
      console.error('Health check failed:', error)
      setIsConnected(false)
    }
  }

  const loadConversationHistory = async (sid) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/session/${sid}/history`)
      if (response.data.messages) {
        const formattedMessages = response.data.messages.map(msg => ({
          role: msg.role,
          content: msg.content,
          timestamp: new Date(msg.timestamp)
        }))
        setMessages(formattedMessages)
      }
    } catch (error) {
      console.error('Failed to load conversation history:', error)
      // Session might be expired, clear it
      localStorage.removeItem('chatSessionId')
      setSessionId(null)
    }
  }

  const sendMessage = async (e) => {
    e.preventDefault()
    if (!inputMessage.trim() || isLoading) return

    const userMessage = {
      role: 'user',
      content: inputMessage.trim(),
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setIsLoading(true)

    try {
      const response = await axios.post(`${API_BASE_URL}/api/chat`, {
        message: userMessage.content,
        session_id: sessionId
      })

      const botMessage = {
        role: 'assistant',
        content: response.data.response,
        timestamp: new Date(response.data.timestamp)
      }

      setMessages(prev => [...prev, botMessage])
      
      // Save session ID
      if (!sessionId && response.data.session_id) {
        setSessionId(response.data.session_id)
        localStorage.setItem('chatSessionId', response.data.session_id)
      }
    } catch (error) {
      console.error('Failed to send message:', error)
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date(),
        isError: true
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
      inputRef.current?.focus()
    }
  }

  const startNewChat = async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/session/new`)
      setSessionId(response.data.session_id)
      localStorage.setItem('chatSessionId', response.data.session_id)
      setMessages([])
    } catch (error) {
      console.error('Failed to create new session:', error)
    }
  }

  const formatTime = (date) => {
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit',
      hour12: true 
    })
  }

  return (
    <div className="app-container">
      {/* Header */}
      <header className="chat-header">
        <div className="header-content">
          <div className="logo-section">
            <div className="logo">🛒</div>
            <div className="title-section">
              <h1>TechMart Support</h1>
              <span className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
                {isConnected ? '● Online' : '○ Connecting...'}
              </span>
            </div>
          </div>
          <button className="new-chat-btn" onClick={startNewChat}>
            + New Chat
          </button>
        </div>
      </header>

      {/* Chat Container */}
      <main className="chat-container">
        {/* Messages Area */}
        <div className="messages-area">
          {messages.length === 0 ? (
            <div className="welcome-message">
              <div className="welcome-icon">🤖</div>
              <h2>Welcome to TechMart Support!</h2>
              <p>I'm your AI assistant. I can help you with:</p>
              <ul className="help-topics">
                <li>📦 Product information & specifications</li>
                <li>🚚 Order status & shipping inquiries</li>
                <li>↩️ Returns & refund policies</li>
                <li>🔧 Technical troubleshooting</li>
                <li>💳 Payment & account questions</li>
              </ul>
              <p className="start-prompt">How can I help you today?</p>
            </div>
          ) : (
            messages.map((message, index) => (
              <div 
                key={index} 
                className={`message ${message.role === 'user' ? 'user-message' : 'bot-message'} ${message.isError ? 'error-message' : ''}`}
              >
                <div className="message-avatar">
                  {message.role === 'user' ? '👤' : '🤖'}
                </div>
                <div className="message-content">
                  <div className="message-bubble">
                    {message.content}
                  </div>
                  <div className="message-timestamp">
                    {formatTime(message.timestamp)}
                  </div>
                </div>
              </div>
            ))
          )}
          
          {/* Typing Indicator */}
          {isLoading && (
            <div className="message bot-message">
              <div className="message-avatar">🤖</div>
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <form className="input-area" onSubmit={sendMessage}>
          <div className="input-container">
            <input
              ref={inputRef}
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Type your message..."
              disabled={isLoading || !isConnected}
              maxLength={4000}
            />
            <button 
              type="submit" 
              disabled={!inputMessage.trim() || isLoading || !isConnected}
              className="send-btn"
            >
              {isLoading ? (
                <span className="loading-spinner"></span>
              ) : (
                '➤'
              )}
            </button>
          </div>
          <div className="input-hint">
            Press Enter to send • {inputMessage.length}/4000
          </div>
        </form>
      </main>
    </div>
  )
}

export default App
