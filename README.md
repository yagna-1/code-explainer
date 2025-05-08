# Code Explainer

An AI-powered code explanation system using Retrieval-Augmented Generation (RAG) and open-source LLMs. This project provides API, web, and CLI interfaces for easy code understanding, leveraging local models and web search for context.

**Repository:** [https://github.com/yagna-1/code-explainer](https://github.com/yagna-1/code-explainer)

---

## Features

- **Explain code snippets** using an LLM, optionally with context from similar code and web search.
- **Suggest improvements** to code only when relevant.
- **Conversational chat**: Handles follow-up questions naturally, including general or social interactions.
- **Retrieval-Augmented Generation**: Combines local codebase and web resources (via SearxNG) for enhanced explanations.
- **FastAPI web server** with a simple frontend.
- **CLI and REST API** for automation and integration.

---

## Architecture Overview

- **Code Explainer API**: Python FastAPI app (see `code_explain.py`)
- **Local LLM Server**: `llama.cpp` container serving GGUF models (model used: `phi-2-dpo.Q4_K_M.gguf`)
- **Web Search**: SearxNG container for open web queries
- **Vector Store**: ChromaDB via `sentence-transformers` for code similarity search

All services can be run together using Docker Compose.

---

## Getting Started

### Prerequisites

- Docker & Docker Compose (recommended)
- (For manual use) Python 3.9+ and pip

### Quick Start (Docker Compose)

```bash
git clone https://github.com/yagna-1/code-explainer.git
cd code-explainer

# Download the phi-2-dpo.Q4_K_M.gguf model into ./models/
# e.g. wget -O models/phi-2-dpo.Q4_K_M.gguf <your-model-url>

docker-compose up --build -d
```

This launches:
- The Code Explainer API (`localhost:8000`)
- The LLM server (`localhost:8080`)
- SearxNG for web search (`localhost:8888`)

### Manual Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start the server:
python code_explain.py --server
```

---

## Model Setup

You **must download the GGUF LLM model** (not included in this repo due to size). Place it in `models/`:

- [phi-2-dpo.Q4_K_M.gguf](https://huggingface.co/TheBloke/phi-2-dpo-GGUF) (used by default in this project)

Update `docker-compose.yml` if you use a different filename.

---

## Usage

### Web UI

Go to [http://localhost:8000/](http://localhost:8000/) in your browser.

### API

**POST `/explain`**

```bash
curl -X POST "http://localhost:8000/explain" \
  -H "Content-Type: application/json" \
  -d '{"code": "def hello():\n    print(\"Hello, world!\")"}'
```

**POST `/agent_explain`**  
For agentic, step-by-step explanations (with reasoning trace).

### CLI

```bash
python code_explain.py -f myscript.py
python code_explain.py -c "def add(a, b): return a + b"
```

---

## Customization

### Environment Variables

- `EMBEDDING_MODEL`: SentenceTransformer model for embeddings
- `LLM_API_URL`: LLM server endpoint (default: http://llm-server:8080/v1)
- `SEARCH_URL`: SearxNG endpoint (default: http://searxng:8080)
- `LLM_TEMPERATURE`: LLM temperature (0.0-1.0)

### Add Your Own Examples

```python
from code_explain import CodeExplainer

explainer = CodeExplainer()
explainer.add_to_knowledge_base(
    ["def foo(): pass", "class Bar: pass"],
    [{"language": "python", "topic": "functions"}, {"language": "python", "topic": "classes"}]
)
```

---

## Project Structure

```
code_explain/
├── code_explain.py           # Main FastAPI app, retrieval, agent, and CLI
├── docker-compose.yml        # Multi-service orchestration
├── Dockerfile                # API server container
├── requirements.txt          # Python dependencies
├── static/                   # Web frontend (index.html)
├── searxng/                  # SearxNG config
├── data/                     # ChromaDB and runtime data
└── models/                   # LLM models (not included)
```

---

## Notes

- **Model files are not in this repo** (see `.gitignore`). Download them manually.
- **No large files should be committed**; see `.gitignore` for exclusions.

---

## License

MIT License

---

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [SentenceTransformers](https://www.sbert.net/)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [SearxNG](https://github.com/searxng/searxng)
- [FastAPI](https://fastapi.tiangolo.com/)
