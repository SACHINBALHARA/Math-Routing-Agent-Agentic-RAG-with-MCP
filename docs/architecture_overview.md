# Code Architecture Overview

## System Architecture

The Math-Routing-Agent-Agentic-RAG-with-MCP system follows a modular, agentic architecture designed for intelligent math problem-solving. Here's a comprehensive overview of the key components and their interactions:

## Core Components

### 1. User Interface Layer
- **Web Interface** (`src/ui/app.py`): Flask-based web application providing user interaction
- **CLI Interface** (`src/main.py`): Command-line tool for direct problem solving
- **Templates** (`src/ui/templates/`): HTML templates for web interface
- **Static Assets** (`src/ui/static/`): CSS, JS, and other frontend resources

### 2. AI Gateway (`src/gateway/ai_gateway.py`)
- **Input Validation**: Sanitizes user queries, detects PII, validates math content
- **Output Validation**: Ensures LaTeX formatting, confidence thresholds, citation validation
- **MCP Envelope Creation**: Structures data for Model Context Protocol tracking

### 3. Pipeline Orchestrator (`src/pipeline/enhanced_pipeline.py`)
- **Enhanced Pipeline**: Coordinates the entire RAG workflow
- **Data Flow Management**: Manages sequence of retrieval → reranking → generation → verification
- **Error Handling**: Propagates errors and manages fallback mechanisms

### 4. Retrieval System (`src/retriever/`)
- **FAISS Retriever** (`retriever.py`): Vector similarity search using FAISS
- **Index Management**: Loads pre-built FAISS indexes and document collections
- **Batch Processing**: Supports multiple query handling

### 5. Reranker (`src/reranker/reranker.py`)
- **Cross-Attention Reranker**: Uses transformer models to improve context relevance
- **Scoring**: Provides confidence scores for retrieved documents
- **Top-K Selection**: Filters most relevant contexts

### 6. Generator (`src/generator/`)
- **Enhanced Generator** (`enhanced_generator.py`): Produces step-by-step solutions
- **LaTeX Formatting**: Ensures mathematical notation quality
- **Symbolic Solving**: Attempts direct SymPy solutions for simple equations

### 7. Symbolic Verifier (`src/verification/symbolic_verifier.py`)
- **Equation Verification**: Uses SymPy to validate mathematical correctness
- **Step Validation**: Checks intermediate solution steps
- **Error Detection**: Identifies incorrect operations

### 8. Web Search Integration (`src/web/web_search.py`)
- **DuckDuckGo Search**: Lightweight web search for additional context
- **Fallback Mechanism**: Provides web results when KB similarity is low

### 9. MCP Persistence (`src/gateway/mcp_persistence.py`)
- **Envelope Storage**: Saves MCP tracking data to JSON files
- **Audit Logging**: Maintains provenance records

## Data Flow

1. **User Input** → AI Gateway (validation/sanitization)
2. **Sanitized Query** → Pipeline Orchestrator
3. **Retrieval Phase** → FAISS Retriever (KB search) + Web Search (if needed)
4. **Reranking Phase** → Cross-Attention Reranker (context refinement)
5. **Generation Phase** → Enhanced Generator (solution creation)
6. **Verification Phase** → Symbolic Verifier (correctness check)
7. **MCP Tracking** → Envelope creation and persistence
8. **Output** → User Interface (formatted results)

## Key Technologies

- **Vector Search**: FAISS for efficient similarity search
- **ML Models**: Transformers (FLAN-T5, Sentence Transformers, Cross-Encoders)
- **Symbolic Math**: SymPy for equation solving and verification
- **Web Framework**: Flask for UI and API
- **Data Processing**: JSON for configuration and data storage

## Configuration and Utilities

- **Config Loader** (`src/utils/config_loader.py`): Environment variable management
- **Data Processing** (`src/data_processing/`): Dataset loading and preprocessing
- **Embeddings** (`src/embeddings/`): Vector generation and indexing

## Testing Framework

- **Unit Tests**: Component-level testing (`tests/`)
- **Integration Tests**: Pipeline testing
- **Demo Scripts**: MCP envelope demonstration

## Architecture Benefits

This architecture ensures:
- **Modularity**: Each component has clear responsibilities and interfaces
- **Scalability**: Stateless design allows horizontal scaling
- **Maintainability**: Clear separation of concerns and comprehensive documentation
- **Reliability**: Robust error handling and fallback mechanisms
- **Auditability**: Full MCP-based provenance tracking

## Component Interactions

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Interface│    │   AI Gateway    │    │   MCP Layer     │
│   (Web/CLI)     │◄──►│   (Validation)  │◄──►│   (Tracking)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Pipeline      │    │   Retriever     │    │   Reranker      │
│   Orchestrator  │◄──►│   (FAISS)       │◄──►│   (Cross-Attn)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Generator     │    │   Verifier      │    │   Web Search    │
│   (LLM)         │◄──►│   (SymPy)       │◄──►│   (DuckDuckGo)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

For detailed technical specifications, see the [High-Level Design Document](https://docs.google.com/document/d/1EXAMPLE_LINK_HERE/edit?usp=sharing).
