# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local RAG (Retrieval Augmented Generation) Chat Application that enables document ingestion from local filesystem, vector storage, and AI-powered chat with documents as knowledge base.

## Commands

### Running the Application
```bash
streamlit run app.py
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

## Architecture

### Core Components

1. **app.py**: Streamlit web interface with two tabs
   - Tab 1: Document ingestion (directory selection, file processing)
   - Tab 2: Chat interface with RAG capabilities

2. **processing/**: Document processing pipeline
   - `loaders.py`: File loaders for txt/md/pdf/docx using Langchain
   - `chunking.py`: Document chunking (1000 chars, 200 overlap)
   - `vector_store.py`: ChromaDB integration with Japanese embeddings (pkshatech/GLuCoSE-base-ja)

3. **llm/**: LLM provider abstraction
   - `llm_interface.py`: Unified interface supporting Ollama/OpenAI/Anthropic/Google via LiteLLM

### Key Design Patterns

- **Vector Storage**: ChromaDB persisted in `./chroma_db` directory
- **Document Processing Flow**: Load → Chunk → Embed → Store with metadata
- **RAG Flow**: Query → Vector search → Context retrieval → LLM prompt → Response with sources
- **Session State**: API keys and LLM config stored in Streamlit session state

### Dependencies

- UI: streamlit
- Document Processing: langchain, langchain-community, pypdf, unstructured
- Vector DB: chromadb, sentence-transformers
- LLM: litellm (provides unified API for multiple providers)
- Package Management: Uses pyproject.toml with Rye/Hatch build system

### Important Notes

- Japanese language optimized with GLuCoSE embedding model
- Supports offline operation with Ollama
- No test suite currently implemented
- No linting configuration found

タスクが終わったら、以下のコマンドで「現在のセッションのコスト」を表示してください。
npx ccusage@latest session --json | jq --arg sid "<現在のセッションID>" '.[] | select(.sessionId == $sid) | .totalCost'