# Math Agent with RAG and MCP

An intelligent math problem-solving agent that uses Retrieval-Augmented Generation (RAG) and Model Context Protocol (MCP) to provide step-by-step solutions with verification.

## Features

- 🔍 **Smart Retrieval**: FAISS-based semantic search for relevant mathematical context
- 🔄 **Cross-Attention Reranking**: Improved context selection using cross-encoder
- ✍️ **LaTeX Formatting**: Beautiful mathematical expressions in solutions
- ✅ **Symbolic Verification**: Uses SymPy to verify mathematical correctness
- 🔒 **AI Gateway**: Input/output validation and safety checks
- 📝 **MCP Integration**: Full provenance tracking and chain-of-thought recording
- 🎯 **Confidence Scoring**: Automatic flagging for human review when needed

## Installation

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Solve equations using the CLI:

```bash
python -m src.main solve "2x + 3 = 11"
```

Enable debug mode for more detailed logs:

```bash
python -m src.main solve "x/2 + 4 = 7" --debug
```

### Python API

```python
from src.pipeline.enhanced_pipeline import EnhancedPipeline

# Initialize the pipeline
pipeline = EnhancedPipeline()

# Solve a math problem
result = pipeline.process_question("Solve: 3x - 6 = 9")

# Access the solution
steps = result["solution"]["steps"]
final_answer = result["solution"]["final_answer"]
confidence = result["solution"]["confidence"]
```

## Pre-warm models (optional)

If you want to pre-download model weights and initialize models before starting the server (recommended for stable first requests), run the warmup script:

```powershell
python scripts\warmup_models.py
```

Or start the UI with a background warmup thread:

```powershell
$env:PRELOAD_MODELS='1'; python src\ui\app.py
```

## Architecture

### Components

1. **AI Gateway** (`src/gateway/ai_gateway.py`)
   - Input sanitization
   - PII detection
   - Output validation
   - LaTeX formatting enforcement

2. **Retriever** (`src/retriever/retriever.py`)
   - FAISS vector store integration
   - Semantic similarity search
   - Batch retrieval support

3. **Reranker** (`src/reranker/reranker.py`)
   - Cross-attention scoring
   - Context relevance improvement
   - Top-k selection

4. **Generator** (`src/generator/enhanced_generator.py`)
   - Step-by-step solution generation
   - LaTeX formatting
   - Confidence scoring

5. **Symbolic Verifier** (`src/verification/symbolic_verifier.py`)
   - SymPy-based equation verification
   - Step validation
   - Solution correctness checking

### MCP Implementation

The Model Context Protocol (MCP) provides:
- Message tracking between components
- Provenance recording
- Chain-of-thought fragments
- Evidence flags for human review

## Testing

Run the test suite:

```bash
python -m unittest discover tests
```

For coverage report:

```bash
pytest --cov=src tests/
```

## Limitations

1. Currently supports:
   - Linear equations
   - Basic algebraic operations
   - Single-variable equations

2. Known limitations:
   - No support for systems of equations yet
   - Limited handling of complex numbers
   - No graphical output for equations

## Future Improvements

1. Web interface for interactive solving
2. Support for systems of equations
3. Integration with computer algebra systems
4. Enhanced explanation generation
5. Student skill level adaptation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - See LICENSE file for details