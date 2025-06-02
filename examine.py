import os
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document # Though not directly returned, it's the type within lists
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import shutil

directory_path = "C:\\Users\\hiroyuki.eto\\Documents\\Obsidian Vault\\Business"



if not os.path.exists(directory_path):
    print(f"エラー: パス '{directory_path}' が存在しません。")
    exit()

files_to_process = []
for root, dirs, files in os.walk(directory_path):
    for file_name in files:
        # os.path.join() を使ってフルパスを作成
        full_path = os.path.join(root, file_name)
        if file_name.endswith(".md"):
            files_to_process.append(os.path.abspath(full_path)) # 絶対パスに変換して追加

def load_md_file(filepath: str) -> list[Document]:
    """Loads text from a .md file into a list containing a single Document."""
    # TextLoader can handle markdown files effectively.
    loader = TextLoader(filepath, encoding='utf-8')
    return loader.load()


def chunk_document(document_input: Document | list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """
    Splits a Langchain Document or a list of Documents into smaller chunks.

    Loaders like PyPDFLoader return a list of Documents (one per page).
    Other loaders like TextLoader or UnstructuredWordDocumentLoader typically return a list containing a single Document.
    This function handles both cases.

    Args:
        document_input: A single Document object or a list of Document objects to be chunked.
        chunk_size: The maximum size of each chunk (in characters).
        chunk_overlap: The number of characters to overlap between chunks.

    Returns:
        A list of Document objects, where each Document is a chunk. Returns an empty list if no valid documents can be processed.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len, # Use Python's built-in len() to count characters
        is_separator_regex=False, # Treat separators as literal strings, not regex
    )
    
    all_chunks = []

    # Ensure document_input is a list to iterate over
    if isinstance(document_input, Document):
        documents_to_process = [document_input]
    elif isinstance(document_input, list):
        documents_to_process = document_input
    else:
        # If input is neither a Document nor a list, it's an unsupported type for this function's design.
        # For robustness, one might log a warning or raise a TypeError.
        # print(f"Warning: chunk_document received an unexpected type: {type(document_input)}. Skipping.")
        return [] # Return empty list if input type is not processable

    for doc in documents_to_process:
        if not isinstance(doc, Document):
            # This handles cases where a list might contain non-Document items,
            # though loaders should ideally return List[Document].
            # print(f"Warning: Item in document list is not a Document type: {type(doc)}. Skipping this item.")
            continue

        # The split_documents method takes a list of Documents and processes each one.
        # It splits the `page_content` of each Document and creates new Document objects for the chunks.
        # Metadata from the original Document is typically carried over to the chunks.
        chunks = text_splitter.split_documents([doc]) # Pass as a list even if it's a single doc from the iteration
        all_chunks.extend(chunks)
        
    return all_chunks

CHROMA_DB_PATH = ".\\chroma_db"

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

loader = load_md_file(files_to_process[14])
#print(loader)
all_chunks = chunk_document(loader)
#print(all_chunks)
#print(f"{len(all_chunks)} chunks")

EMBEDDING_MODEL = "pkshatech/GLuCoSE-base-ja"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

clear_vector_store(CHROMA_DB_PATH)

current_vector_store = get_vector_store(collection_name="main_document_collection")