# CBOTTY - Customer Service Chatbot

A simple RAG (Retrieval-Augmented Generation) chatbot for ShopUNow, a fictional retail company selling clothing, DIY products, books, and toys. This chatbot provides intelligent customer support by routing inquiries to appropriate departments and leveraging a vector database of company knowledge.

## Scope

CBOTTY is designed as an internal employee support system for ShopUNow that:

- **Multi-Department Support**: Routes queries to HR, IT, Sales, or General departments based on content analysis
- **Sentiment Analysis**: Detects user sentiment and escalates negative interactions to human support
- **RAG Architecture**: Uses ChromaDB vector store with synthetic FAQ data for context-aware responses
- **Intelligent Routing**: Employs LangGraph workflow for sophisticated query processing
- **Web Interface**: Simple NiceGUI-based chat interface for easy interaction
- **Conversation Memory**: Maintains session-based conversation history

### Key Features:
- Automated department classification (HR, IT, Sales, General)
- Sentiment-driven escalation to customer service
- Vector similarity search for relevant context retrieval
- Web search integration for queries outside the knowledge base
- Streaming responses with real-time updates
- Docker containerization for easy deployment

## Installation

### Prerequisites
- Python 3.12+
- OpenAI API key
- Tavily API key (for web search functionality)

1. **Clone the repository**:
```bash
git clone https://github.com/heinerlehr/cbotty.git
cd cbotty
```

2. **Create environment file**:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. **Build and run with Docker Compose**:
```bash
docker-compose up --build
```

4. **Access the application**:
   - Open your browser to `http://localhost:8080`

## Usage

### Basic Chat Interface

1. **Start a conversation**: Type your question in the input field and press Enter or click Send
2. **Department routing**: The system automatically determines which department should handle your query
3. **Streaming responses**: Watch as the bot processes your request and provides real-time updates
4. **Follow-up questions**: Continue the conversation with context-aware responses

### Example Queries

**HR Questions**:
- "How do I apply for annual leave?"
- "What is the procedure for reporting sick leave?"
- "Where can I find the employee handbook?"

**IT Questions**:
- "How do I reset my email password?"
- "I can't connect to the office Wi-Fi"
- "How do I request new software?"

**Sales Questions**:
- "What is the current sales target for the quarter?"
- "How do I access customer purchase history?"
- "What are the current promotional offers?"

### System Behavior

- **Positive/Neutral sentiment**: Routes to appropriate department specialist
- **Negative sentiment**: Escalates to customer service agent with empathetic response
- **Unknown queries**: Falls back to general agent with web search capability
- **Unresolvable issues**: Redirects to human support with contact information

## Limitations

- **Memory Management**: Sessions stored in memory only (lost on restart), no persistence layer
- **Scalability**: Single-threaded processing, limited concurrent users, no horizontal scaling
- **Security**: No authentication, API keys in environment variables, minimal input validation
- **Data**: Uses synthetic FAQ data only, limited knowledge base scope
- **Production Readiness**: No health checks, limited error handling, no monitoring capabilities
- **Classification**: May misclassify queries, no learning mechanism for improvement

---

**Note**: This is a demonstration project for educational purposes. For production use, address the security and scalability limitations listed above.