# Mini RAG Chatbot

## Environment Setup

### Create a conda environment:

```bash
conda create -n mini-rag python=3.10
conda activate mini-rag
conda install -c conda-forge faiss-cpu sentence-transformers transformers
pip install langchain rank-bm25 fastapi uvicorn requests tqdm scikit-learn pypdf
```

### Install Ollama for local LLM:

Download Ollama from: https://ollama.com.

Then pull a model:
```bash
ollama pull llama3
```
## How to test
### 1. Add a document

Create a new folder `/data`.

Place your PDF file at: `data/documents.pdf`.

### 2. Build the Vector Database

Run `python src/ingest.py`.

### 3. Start the API Server

Run `python -m uvicorn src.api:app --reload --port 8000`.

Open http://127.0.0.1:8000/docs in your browser.

### 4. Run test

Run `python test.py`.
