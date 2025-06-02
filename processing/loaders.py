from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_core.documents import Document # Though not directly returned, it's the type within lists

# Note: All loaders here are expected to return a list of Document objects.
# For TextLoader and UnstructuredWordDocumentLoader, this list usually contains a single Document.
# For PyPDFLoader, it contains one Document per page.

def load_txt_file(filepath: str) -> list[Document]:
    """Loads text from a .txt file into a list containing a single Document."""
    loader = TextLoader(filepath, encoding='utf-8')
    return loader.load()

def load_md_file(filepath: str) -> list[Document]:
    """Loads text from a .md file into a list containing a single Document."""
    # TextLoader can handle markdown files effectively.
    loader = TextLoader(filepath, encoding='utf-8')
    return loader.load()

def load_pdf_file(filepath: str) -> list[Document]:
    """Loads text from a .pdf file, creating one Document per page."""
    loader = PyPDFLoader(filepath)
    # load_and_split() is suitable for PDFs, creating a document per page.
    return loader.load_and_split() 

def load_docx_file(filepath: str) -> list[Document]:
    """Loads text from a .docx file into a list containing a single Document."""
    loader = UnstructuredWordDocumentLoader(filepath)
    return loader.load()
