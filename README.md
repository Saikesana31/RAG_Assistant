# RAG Agent - Intelligent Document Q&A System

A production-ready Retrieval-Augmented Generation (RAG) system built with FastAPI, Inngest, and Qdrant. This project enables semantic search and question-answering over your PDF documents using state-of-the-art embeddings and LLMs.

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128+-green.svg)](https://fastapi.tiangolo.com/)
[![Inngest](https://img.shields.io/badge/Inngest-0.5.13-purple.svg)](https://www.inngest.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 📋 Table of Contents

- [Introduction](#introduction)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Setup](#environment-setup)
- [Usage](#usage)
  - [Running the Application](#running-the-application)
  - [Ingesting Documents](#ingesting-documents)
  - [Querying Documents](#querying-documents)
- [Project Structure](#project-structure)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Introduction

This RAG Agent transforms static PDF documents into an intelligent, queryable knowledge base. Users can:

- **Upload PDF documents** for automatic processing and indexing
- **Ask natural language questions** and receive contextual answers
- **Track processing status** in real-time via Inngest workflows
- **Access source attribution** for every answer generated

Perfect for building internal knowledge bases, customer support systems, research assistants, or any application requiring document understanding.

---

## 🛠️ Tech Stack

### Core Framework
- **[FastAPI](https://fastapi.tiangolo.com/)** `0.128+` - Modern, high-performance web framework
- **[Uvicorn](https://www.uvicorn.org/)** `0.40+` - ASGI server for production deployment

### Workflow Orchestration
- **[Inngest](https://www.inngest.com/)** `0.5.13+` - Event-driven workflow engine with built-in retry logic and observability

### Vector Database
- **[Qdrant](https://qdrant.tech/)** - High-performance vector similarity search engine
  - Cosine similarity for semantic matching
  - 3072-dimensional vector space
  - Persistent storage

### AI/ML Stack
- **[OpenAI API](https://openai.com/)**
  - `text-embedding-3-large` - 3072-dimensional embeddings
  - `gpt-4o-mini` - Efficient LLM for answer generation
- **[llama-index](https://www.llamaindex.ai/)** - Document processing and chunking
  - `PDFReader` for PDF extraction
  - `SentenceSplitter` for intelligent chunking

### Frontend
- **[Streamlit](https://streamlit.io/)** - Interactive web UI for rapid prototyping

### Type Safety & Validation
- **[Pydantic](https://pydantic.dev/)** `2.12+` - Data validation and settings management

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface Layer                    │
│                    (Streamlit Frontend)                      │
└────────────────────┬────────────────────────────────────────┘
                     │ HTTP Requests
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   API Layer (FastAPI)                        │
│  • /events endpoint (receives Inngest events)               │
│  • /health endpoint (health checks)                          │
└────────────────────┬────────────────────────────────────────┘
                     │ Event Triggers
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              Workflow Orchestration (Inngest)                │
│                                                              │
│  ┌────────────────────┐      ┌─────────────────────┐       │
│  │ rag/ingest_pdf     │      │  rag/query_pdf      │       │
│  │                    │      │                     │       │
│  │ 1. Load PDF        │      │  1. Embed query     │       │
│  │ 2. Chunk text      │      │  2. Search vectors  │       │
│  │ 3. Generate        │      │  3. Retrieve context│       │
│  │    embeddings      │      │  4. Generate answer │       │
│  │ 4. Upsert to       │      │                     │       │
│  │    Qdrant          │      │                     │       │
│  └────────┬───────────┘      └──────────┬──────────┘       │
│           │                             │                   │
└───────────┼─────────────────────────────┼───────────────────┘
            │                             │
            ▼                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Processing Layer                     │
│                                                              │
│  ┌─────────────────┐         ┌──────────────────┐          │
│  │  data_loader.py │         │   OpenAI API     │          │
│  │                 │         │                  │          │
│  │ • PDFReader     │────────▶│ • Embeddings     │          │
│  │ • Sentence      │         │ • Chat           │          │
│  │   Splitter      │         │   Completions    │          │
│  └─────────────────┘         └──────────────────┘          │
└───────────────────────────────┬─────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                 Vector Storage Layer (Qdrant)                │
│                                                              │
│  • Collection: rag_collection                                │
│  • Vectors: 3072 dimensions (text-embedding-3-large)        │
│  • Distance: Cosine similarity                               │
│  • Payloads: {source, text}                                  │
│  • Persistence: Local or cloud-hosted                        │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

#### Ingestion Pipeline
```
PDF Upload → Streamlit UI → Inngest Event (rag/ingest_pdf)
    ↓
Load PDF (PDFReader) → Extract Text
    ↓
Chunk Text (SentenceSplitter: 1000 chars, 200 overlap)
    ↓
Generate Embeddings (OpenAI text-embedding-3-large, 3072-dim)
    ↓
Upsert to Qdrant (Collection: rag_collection)
    ↓
Return Ingestion Status
```

#### Query Pipeline
```
User Question → Streamlit UI → Inngest Event (rag/query_pdf)
    ↓
Embed Question (OpenAI text-embedding-3-large)
    ↓
Vector Search (Qdrant, top_k=5, cosine similarity)
    ↓
Retrieve Context Chunks + Sources
    ↓
Build Prompt (Context + Question)
    ↓
Generate Answer (OpenAI gpt-4o-mini)
    ↓
Return {answer, sources, num_contexts}
```

### Key Components

#### 1. **custom_types.py** - Type Definitions
- `RAGChunkAndSrc`: Chunked document data
- `RAGUpsertResult`: Ingestion confirmation
- `RAGSearchResult`: Retrieved context and sources
- `RAGQueryResult`: Final answer with metadata

#### 2. **data_loader.py** - Document Processing
- PDF loading with llama-index PDFReader
- Semantic chunking with SentenceSplitter
- Embedding generation via OpenAI API

#### 3. **vector_DB.py** - Qdrant Interface
- Collection management
- Vector upsert with payloads
- Similarity search with metadata retrieval

#### 4. **main.py** - FastAPI + Inngest Integration
- Two Inngest functions: `rag_ingest_pdf`, `rag_query_pdf`
- Event-driven step execution
- Type-safe serialization with Pydantic

#### 5. **streamlit.py** - User Interface
- PDF upload interface
- Question/answer form
- Real-time status polling

---

## ✨ Features

### Core Capabilities
- ✅ **Semantic Search** - Find relevant information across documents
- ✅ **Context-Aware Answers** - AI-generated responses based on your data
- ✅ **Source Attribution** - Know which documents contributed to each answer
- ✅ **Async Processing** - Non-blocking workflows with Inngest
- ✅ **Real-Time Status** - Poll workflow status via Inngest API

### Technical Highlights
- ✅ **Event-Driven Architecture** - Scalable, resilient workflows
- ✅ **Type Safety** - Pydantic models throughout
- ✅ **Chunk Overlap** - Better context preservation (200 char overlap)
- ✅ **Deterministic IDs** - UUID5 based on source + index
- ✅ **Configurable Top-K** - Adjust retrieval count per query

---

## 🚀 Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.12+** - [Download](https://www.python.org/downloads/)
- **Docker** (for Qdrant) - [Download](https://www.docker.com/get-started)
- **OpenAI API Key** - [Get yours](https://platform.openai.com/api-keys)
- **Inngest Dev Server** - [Install](https://www.inngest.com/docs/local-development)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/rag-agent.git
   cd rag-agent
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # Using pip
   pip install -r requirements.txt

   # Or using uv (faster)
   uv pip install -r requirements.txt
   ```

### Environment Setup

1. **Create a `.env` file** in the project root:
   ```env
   # OpenAI Configuration
   OPENAI_API_KEY=sk-your-openai-api-key-here

   # Inngest Configuration
   INNGEST_API_BASE=http://127.0.0.1:8288  # Local dev server
   INNGEST_SIGNING_KEY=your-signing-key    # Optional for local dev
   INNGEST_EVENT_KEY=your-event-key        # Optional for local dev

   # Qdrant Configuration (optional, defaults to localhost)
   QDRANT_URL=http://localhost:6333
   QDRANT_COLLECTION=rag_collection
   ```

2. **Start Qdrant** (using Docker)
   ```bash
   docker run -p 6333:6333 -p 6334:6334 \
       -v $(pwd)/qdrant_storage:/qdrant/storage \
       qdrant/qdrant
   ```

   Or use Qdrant Cloud:
   ```bash
   # Update .env with your cloud URL
   QDRANT_URL=https://your-cluster.qdrant.io
   QDRANT_API_KEY=your-api-key
   ```

3. **Install and start Inngest Dev Server**
   ```bash
   # Install Inngest CLI
   npm install -g inngest-cli

   # Start dev server
   inngest dev
   ```

---

## 📖 Usage

### Running the Application

1. **Start the FastAPI server**
   ```bash
   uvicorn main:app --reload --port 8000
   ```

2. **In a new terminal, start Streamlit**
   ```bash
   streamlit run streamlit.py
   ```

3. **Access the application**
   - Streamlit UI: http://localhost:8501
   - FastAPI Docs: http://localhost:8000/docs
   - Inngest Dev UI: http://localhost:8288

### Ingesting Documents

1. Navigate to the Streamlit UI
2. Click **"Choose a PDF"** and select your document
3. The system will:
   - Upload the file to `uploads/` directory
   - Trigger the `rag/ingest_pdf` event
   - Process chunks in the background
   - Display success message when complete

**Behind the scenes:**
```python
# Event triggered
{
  "name": "rag/ingest_pdf",
  "data": {
    "pdf_path": "/path/to/document.pdf",
    "source_id": "document.pdf"
  }
}

# Workflow steps
Step 1: load_chunks → Extract and chunk text
Step 2: embeddings_upsert → Embed and store in Qdrant
```

### Querying Documents

1. Enter your question in the text input
2. Adjust **"How many chunks to retrieve"** (default: 5)
3. Click **"Ask"**
4. View the answer and source documents

**Behind the scenes:**
```python
# Event triggered
{
  "name": "rag/query_pdf",
  "data": {
    "question": "What is the main topic?",
    "top_k": 5
  }
}

# Workflow steps
Step 1: embed_search → Vector similarity search
Step 2: call-openai → Generate contextual answer

# Response
{
  "answer": "The main topic is...",
  "sources": ["doc1.pdf", "doc2.pdf"],
  "num_contexts": 5
}
```

---

## 📁 Project Structure

```
rag-agent/
│
├── main.py                  # FastAPI app + Inngest functions
├── data_loader.py           # PDF processing and embedding
├── vector_DB.py             # Qdrant client wrapper
├── custom_types.py          # Pydantic models
├── streamlit.py             # Streamlit UI
│
├── uploads/                 # Uploaded PDF storage
├── qdrant_storage/          # Qdrant persistent storage (if local)
│
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Project metadata (uv/poetry)
├── uv.lock                 # Locked dependencies
│
├── .env                    # Environment variables (create this)
├── .gitignore
└── README.md               # This file
```

---

## 🚢 Deployment

### Docker Deployment

**Assumptions:**
- Production Qdrant instance (cloud or self-hosted)
- Inngest Cloud account for production workflows
- Environment variables configured

1. **Create `Dockerfile`**
   ```dockerfile
   FROM python:3.12-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   EXPOSE 8000

   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
   ```

2. **Build and run**
   ```bash
   docker build -t rag-agent .
   docker run -p 8000:8000 --env-file .env rag-agent
   ```

### Cloud Deployment Options

#### Option 1: Railway / Render
- Push to GitHub
- Connect repository to platform
- Set environment variables
- Deploy FastAPI app
- Deploy Streamlit separately (or combine)

#### Option 2: AWS / GCP / Azure
- **Compute:** ECS, Cloud Run, or App Service
- **Vector DB:** Qdrant Cloud or self-hosted on EC2/Compute Engine
- **Inngest:** Inngest Cloud (recommended for production)

#### Option 3: Kubernetes
- Deploy FastAPI as a Deployment
- Use Qdrant Helm chart or managed service
- Configure Inngest Cloud webhook

### Production Checklist

- [ ] Set up Qdrant Cloud or production instance
- [ ] Configure Inngest Cloud account
- [ ] Store secrets in vault (AWS Secrets Manager, etc.)
- [ ] Enable CORS if needed for Streamlit
- [ ] Set up logging and monitoring
- [ ] Configure rate limiting for OpenAI API
- [ ] Add authentication to FastAPI endpoints
- [ ] Set up CI/CD pipeline
- [ ] Configure backup for Qdrant data

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
   - Follow PEP 8 style guidelines
   - Add type hints
   - Update tests if applicable
4. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
5. **Push to your branch**
   ```bash
   git push origin feature/amazing-feature
   ```
6. **Open a Pull Request**

### Development Guidelines

- Use Pydantic models for all data structures
- Add type hints to all functions
- Write docstrings for public methods
- Test with multiple PDF formats
- Update README for new features

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- [Inngest](https://www.inngest.com/) for workflow orchestration
- [Qdrant](https://qdrant.tech/) for vector search capabilities
- [OpenAI](https://openai.com/) for embeddings and LLM
- [llama-index](https://www.llamaindex.ai/) for document processing

---

## 📬 Contact

**Questions or suggestions?**
- Open an issue on GitHub
- Connect on LinkedIn:www.linkedin.com/in/saikesana
- Email: kesana.class2024@gmail.com

---

**Built with ❤️ using Python and modern AI/ML tools**
