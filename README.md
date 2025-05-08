# Code Explainer

An AI-powered code explanation system using Retrieval-Augmented Generation (RAG) with open-source LLMs.

## Overview

Code Explainer is an agentic system that leverages state-of-the-art language models and retrieval techniques to provide high-quality explanations for code snippets. The system consists of several key components:

1. **Code-Specialized LLM**: Uses models like CodeLlama, StarCoder, or WizardCoder for generating explanations
2. **Embedding Model**: SentenceTransformers for vectorizing code and finding similar examples
3. **Vector Database**: ChromaDB for efficient storage and retrieval of code snippets
4. **Web Search**: SearxNG integration for finding context when the local database is insufficient
5. **API & Frontend**: FastAPI and a simple HTML/JS interface for easy interaction

## Architecture

The system follows a hybrid retrieval workflow:
1. Vectorize input code and find similar examples in the vector database
2. If insufficient context is found, search the web via SearxNG
3. Rerank combined results using a cross-encoder
4. Use the retrieved context to enhance the LLM's code explanation

## Installation

### Using Docker (Recommended)

The easiest way to get started is using Docker Compose:

```bash
# Clone the repository
git clone <repository-url>
cd code-explainer

# Start the services
docker-compose up -d
```

This will start:
- Code Explainer API server
- Local LLM server (llama.cpp) for inference
- SearxNG for web search capabilities

### Manual Setup

If you prefer a manual setup:

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python code_explain.py --server
```

## Model Requirements

To use the system with local LLM inference, download one of these models:
- [CodeLlama-7B-GGUF](https://huggingface.co/TheBloke/CodeLlama-7B-GGUF) (recommended)
- [StarCoder2-7B-GGUF](https://huggingface.co/TheBloke/starcoder2-7b-GGUF)
- [WizardCoder-Python-7B-GGUF](https://huggingface.co/TheBloke/WizardCoder-Python-7B-GGUF)

Place the model file in the `models/` directory and update the docker-compose.yml file if the filename differs.

## Usage

### Web Interface

Once the server is running, open a browser and navigate to:
```
http://localhost:8000/
```

### API

The system exposes a REST API endpoint:

```bash
curl -X POST "http://localhost:8000/explain" \
  -H "Content-Type: application/json" \
  -d '{"code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)", "context": "This is a recursive function."}'
```

### CLI

The tool can also be used from the command line:

```bash
# Explain code from a file
python code_explain.py -f your_code_file.py

# Directly explain code snippet
python code_explain.py -c "def hello():\n    print('Hello, world!')"
```

## Customization

### Environment Variables

The system can be customized using environment variables:

- `EMBEDDING_MODEL`: SentenceTransformer model to use for embeddings
- `LLM_API_URL`: Endpoint for the LLM API
- `SEARCH_URL`: SearxNG instance URL
- `LLM_TEMPERATURE`: Temperature setting for text generation (0.0-1.0)

### Adding to Knowledge Base

To improve the system with your own code examples:

```python
from code_explain import CodeExplainer

explainer = CodeExplainer()
code_examples = [
    "def example1(): ...",
    "class Example2: ..."
]
metadata = [
    {"language": "python", "topic": "functions"},
    {"language": "python", "topic": "classes"}
]
explainer.add_to_knowledge_base(code_examples, metadata)
```

## Project Structure

```
code_explain/
├── code_explain.py        # Main application code
├── docker-compose.yml     # Docker services configuration  
├── Dockerfile             # Container definition
├── requirements.txt       # Python dependencies
├── static/                # Frontend files
│   └── index.html         # Web interface
├── data/                  # Data storage (created at runtime)
│   └── ...
└── models/                # LLM model storage
    └── ...
```

## License

This project is available under the MIT License.

## Acknowledgments

This system integrates several open-source projects:
- [Code Llama](https://github.com/facebookresearch/codellama)
- [SentenceTransformers](https://www.sbert.net/)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [SearxNG](https://github.com/searxng/searxng)
- [FastAPI](https://fastapi.tiangolo.com/)