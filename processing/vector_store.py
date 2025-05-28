import os
import datetime
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Define the path for the ChromaDB persistent storage
CHROMA_DB_PATH = "./chroma_db"
# Define the embedding model
EMBEDDING_MODEL = "pkshatech/GLuCoSE-base-ja"

import shutil # For deleting the directory

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def clear_vector_store(db_path=CHROMA_DB_PATH):
    """
    Deletes the ChromaDB persistent storage directory.
    """
    if os.path.exists(db_path):
        try:
            shutil.rmtree(db_path)
            return True, f"Successfully deleted vector store at {db_path}"
        except Exception as e:
            return False, f"Error deleting vector store at {db_path}: {e}"
    return True, f"Vector store at {db_path} does not exist. No action taken."

def get_vector_store(collection_name="default_collection"):
    """
    Initializes and returns a Chroma vector store.

    Args:
        collection_name (str): Name of the collection within ChromaDB.

    Returns:
        Chroma: An instance of the Chroma vector store.
    """
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    return vector_store

def add_documents_to_store(documents: list[Document], vector_store: Chroma, file_path: str):
    """
    Adds chunked documents to the Chroma vector store with metadata.

    Args:
        documents (list[Document]): A list of chunked Document objects.
        vector_store (Chroma): The Chroma vector store instance.
        file_path (str): The full path to the original source file.
    """
    if not documents:
        return 0

    metadatas = []
    ids = [] # Optional: Chroma can auto-generate IDs, or you can provide them.
    
    file_name = os.path.basename(file_path)
    file_type = os.path.splitext(file_name)[1].lower().strip('.')
    
    try:
        last_modified_timestamp = os.path.getmtime(file_path)
        last_modified_datetime = datetime.datetime.fromtimestamp(last_modified_timestamp).isoformat()
    except FileNotFoundError:
        last_modified_datetime = datetime.datetime.now().isoformat() # Fallback if file is deleted during processing

    for i, doc_chunk in enumerate(documents):
        # Prepare metadata for each chunk
        chunk_metadata = {
            "source": file_path,
            "file_name": file_name,
            "file_type": file_type,
            "last_modified": last_modified_datetime,
            # Add page number if available (common for PDFs)
            # The 'page' key might come from PyPDFLoader's output directly in doc_chunk.metadata
            "page": doc_chunk.metadata.get("page", None) 
        }
        metadatas.append(chunk_metadata)
        # Creating a unique ID for each chunk. Useful for updates/deletions later.
        ids.append(f"{file_path}_chunk_{i}")

    # Add documents to the vector store
    # Note: Chroma's `add_documents` handles embedding generation internally if an embedding_function is set.
    # We pass the `Document` objects directly.
    if documents: # Ensure there are documents to add
        vector_store.add_documents(documents=documents, metadatas=metadatas, ids=ids)
        vector_store.persist() # Persist changes to disk
        return len(documents)
    return 0
