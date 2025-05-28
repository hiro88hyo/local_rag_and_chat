# Local File RAG Chat Application

This application allows you to ingest documents from a local directory, process them into a vector store, and then chat with an AI model that uses these documents as a knowledge base (Retrieval Augmented Generation - RAG).

## Features

*   **Local Document Ingestion:** Supports `.txt`, `.md`, `.pdf`, and `.docx` files.
*   **Vector Store Creation:** Chunks documents and stores them in a local ChromaDB vector database.
*   **Configurable LLM Integration:**
    *   Supports multiple LLM providers via LiteLLM: Ollama (for local models), OpenAI, Anthropic, Google.
    *   UI for selecting provider, model name, and entering API keys/Ollama URL.
*   **Chat Interface:** Ask questions and get answers based on the content of your ingested documents.
*   **Source Display:** Shows snippets from source documents used to generate answers.
*   **Data Persistence:** Vector store is saved locally and reloaded. Option to clear the database.

## Setup Instructions

### Prerequisites

*   **Python:** Version 3.9 or higher recommended.
*   **Ollama (for local LLMs):** If you plan to use local LLMs with Ollama:
    *   Install Ollama from [https://ollama.com/](https://ollama.com/).
    *   Ensure the Ollama server is running (`ollama serve`).
    *   Pull desired models, e.g., `ollama pull llama2` or `ollama pull mistral`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
    Activate it:
    *   On macOS/Linux: `source venv/bin/activate`
    *   On Windows: `venv\Scripts\activate`
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

```bash
streamlit run app.py
```
The application should open in your web browser.

### Configuring LLMs

*   The application provides a UI to configure LLM settings in the "Chat with Documents" tab sidebar.
*   **Provider Selection:** Choose from Ollama, OpenAI, Anthropic, or Google.
*   **Model Name:** Enter the specific model name for the selected provider (e.g., `llama2`, `gpt-3.5-turbo`, `claude-3-haiku-20240307`, `gemini-pro`).
    *   For Ollama, this is the name of the model you pulled (e.g., `llama2`).
    *   For cloud providers, refer to their documentation for available model names.
*   **Ollama Base URL:** If using Ollama, ensure this points to your running Ollama server (default: `http://localhost:11434`).
*   **API Keys:** For OpenAI, Anthropic, and Google, you must provide your API key. These are handled client-side in this local application; see "API Key Management" for security notes.

## API Key Management

*   **Local Application Handling:** In this Streamlit application, API keys entered into the UI are stored in the browser's session state. This is generally acceptable for local, personal use.
*   **Security Note:** **DO NOT** deploy this application as-is to a shared or public environment with API keys managed this way.
*   **Production/Shared Environments:** For more secure deployments, API keys should be managed through environment variables or dedicated secret management services. LiteLLM automatically reads API keys from standard environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`). You would typically set these in your server environment before running the application, and the UI for key input could be removed or disabled.

## Local LLM Availability (Offline Usage)

*   If you select "Ollama" as the LLM provider and have correctly configured the Ollama Base URL and a downloaded model (e.g., `llama2`), the chat functionality can operate **without an internet connection**.
*   The document ingestion process (specifically downloading the embedding model `pkshatech/GLuCoSE-base-ja` for the first time) and installation of Python packages will require an internet connection initially. Once the embedding model is cached locally by HuggingFace Transformers and packages are installed, subsequent runs for ingestion and chat (with Ollama) can be offline.

## Troubleshooting

*   **Ollama Connection Issues:**
    *   Ensure the Ollama server is running. You can test this by opening `http://localhost:11434` (or your configured URL) in a browser or using `curl`.
    *   Verify the model name specified in the UI has been pulled via `ollama pull <model_name>`.
    *   Check the Ollama server logs for any errors.
*   **Embedding Model Download:** If you see errors related to `sentence-transformers` or `HuggingFaceEmbeddings` on first run, ensure you have an internet connection for the initial download of the `pkshatech/GLuCoSE-base-ja` model.
*   **PDF/DOCX Processing Issues:** `unstructured` (used for DOCX) and `pypdf` can sometimes have system dependencies. If you encounter issues loading these file types, check the `unstructured` documentation for any necessary system packages (e.g., `libreoffice` for `.doc`, `poppler-utils` for some PDFs, though `PyPDFLoader` is primary for PDFs).
*   **Incorrect API Key:** If using cloud LLMs, ensure your API key is correct and has the necessary permissions/credits.

## System Architecture

This application follows a component-based architecture:

*   **Frontend:**
    *   **Streamlit (`app.py`):** Provides the user interface for document ingestion, LLM configuration, and chat.
*   **Backend Logic (Python Modules):**
    *   **`processing/` directory:** Contains modules for document handling.
        *   `loaders.py`: Functions to load various document types (`.txt`, `.md`, `.pdf`, `.docx`).
        *   `chunking.py`: Logic to split documents into smaller, manageable chunks.
        *   `vector_store.py`: Manages the ChromaDB vector store, including adding documents and initializing the embedding model.
    *   **`llm/` directory:**
        *   `llm_interface.py`: Provides a unified interface to various LLMs (Ollama, OpenAI, Anthropic, Google) using LiteLLM.
*   **Key Components & Data Flow:**
    *   **Embedding Model:** `HuggingFaceEmbeddings` (model: `pkshatech/GLuCoSE-base-ja`) is used to generate vector embeddings for text chunks. This model is downloaded and cached locally.
    *   **Vector Store:** `ChromaDB` is used as the local vector database to store text chunks and their corresponding embeddings. It's persisted in the `./chroma_db` directory.
    *   **RAG Orchestration:** The core Retrieval Augmented Generation logic is primarily managed within `app.py` in the chat tab.
        1.  **Ingestion Data Flow:**
            *   User selects a directory containing documents via the Streamlit UI.
            *   `app.py` iterates through files:
                *   `processing/loaders.py` loads supported files.
                *   `processing/chunking.py` splits loaded documents into chunks.
                *   `processing/vector_store.py` generates embeddings for chunks and stores them in ChromaDB along with metadata.
        2.  **Chat Data Flow (RAG):**
            *   User asks a question in the Streamlit chat interface.
            *   `app.py` takes the user's question.
            *   The question is used to query `ChromaDB` (via `vector_store.similarity_search_with_score`) for relevant document chunks.
            *   The retrieved chunks (context) and the original question are formatted into a prompt.
            *   The prompt is sent to the configured LLM (via `llm/llm_interface.py`).
            *   The LLM's response is displayed in the chat UI, along with references to the source document chunks.

## Data Model

*   **Input Document Types:** The system supports ingestion of the following file types:
    *   `.txt` (Plain text)
    *   `.md` (Markdown)
    *   `.pdf` (Portable Document Format)
    *   `.docx` (Microsoft Word Document)

*   **Vector Store (ChromaDB):**
    *   Each document chunk processed by the system is stored as an entry in ChromaDB.
    *   Key information stored per chunk includes:
        *   **`page_content`**: The actual text content of the chunk.
        *   **Vector Embedding**: A numerical representation of the `page_content`, generated by the HuggingFace embedding model. This is used for similarity searches.
        *   **Metadata**: A collection of fields providing information about the chunk's origin and characteristics. This includes:
            *   `source`: The full file path of the original document.
            *   `file_name`: The name of the original document file (e.g., `my_document.pdf`).
            *   `file_type`: The extension of the original document file (e.g., `pdf`, `txt`).
            *   `page`: The page number from which the chunk was extracted (primarily for PDF documents).
            *   `last_modified`: The ISO format timestamp of when the original file was last modified.
    *   This metadata is added in `processing/vector_store.py` during the `add_documents_to_store` function and is used for display and potential filtering.

## Technical Stack

*   **Programming Language:** Python 3.9+
*   **UI Framework:** Streamlit (`streamlit`)
*   **Core Orchestration & Components:** Langchain (`langchain`, `langchain-community`)
    *   **Document Loaders:**
        *   `PyPDFLoader` (via `pypdf`) for PDF files.
        *   `UnstructuredWordDocumentLoader` (via `unstructured` and `python-docx`) for DOCX files.
        *   `TextLoader` for `.txt` and `.md` files.
    *   **Text Splitter:** `RecursiveCharacterTextSplitter` from Langchain.
*   **Embedding Model:**
    *   HuggingFace Embeddings via `sentence-transformers`.
    *   Model: `pkshatech/GLuCoSE-base-ja` (Japanese language model, effective for general text).
*   **Vector Database:** ChromaDB (`chromadb`) for local vector storage and retrieval.
*   **LLM Interface:** LiteLLM (`litellm`) for standardized access to various LLM providers.
*   **Supported LLM Providers (via LiteLLM):**
    *   Ollama (for local models like Llama 2, Mistral)
    *   OpenAI (e.g., GPT-3.5-turbo, GPT-4)
    *   Anthropic (e.g., Claude series)
    *   Google (e.g., Gemini series)
