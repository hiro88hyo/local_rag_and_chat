from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Used for type hinting and creating Document objects

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

if __name__ == '__main__':
    # Example Usage (for testing this module directly)
    
    # Simulating a single Document object (like from TextLoader for a small file)
    sample_doc_content = "This is a sample document. It has several sentences. We want to split this into smaller chunks for processing."
    single_doc = Document(page_content=sample_doc_content, metadata={"source": "single_doc_test.txt"})
    
    print("--- Testing with a single Document ---")
    chunks_from_single = chunk_document(single_doc, chunk_size=50, chunk_overlap=10)
    if chunks_from_single:
        for i, chunk in enumerate(chunks_from_single):
            print(f"Chunk {i+1}: '{chunk.page_content}' (Metadata: {chunk.metadata})")
    else:
        print("No chunks created from single document.")

    # Simulating a list of Document objects (like from PyPDFLoader)
    sample_docs_list = [
        Document(page_content="Page one content. This is the first page and has some text.", metadata={"source": "multipage_doc.pdf", "page": 1}),
        Document(page_content="Page two content. This is the second page, continuing the document.", metadata={"source": "multipage_doc.pdf", "page": 2}),
        Document(page_content="A very short third page.", metadata={"source": "multipage_doc.pdf", "page": 3})
    ]
    
    print("\n--- Testing with a list of Documents ---")
    chunks_from_list = chunk_document(sample_docs_list, chunk_size=60, chunk_overlap=15)
    if chunks_from_list:
        for i, chunk in enumerate(chunks_from_list):
            print(f"Chunk {i+1}: '{chunk.page_content}' (Metadata: {chunk.metadata})")
    else:
        print("No chunks created from list of documents.")

    # Simulating an empty list
    print("\n--- Testing with an empty list ---")
    chunks_from_empty_list = chunk_document([], chunk_size=50, chunk_overlap=10)
    if not chunks_from_empty_list:
        print("Correctly returned an empty list for empty input.")
    else:
        print(f"Error: Expected empty list, got {len(chunks_from_empty_list)} chunks.")

    # Simulating invalid input (though type hinting should prevent this in typed code)
    # print("\n--- Testing with invalid input type (string instead of Document) ---")
    # chunks_from_string = chunk_document("This is just a raw string.", chunk_size=50, chunk_overlap=10)
    # if not chunks_from_string: # Based on current implementation, this path might not be hit if it tries to convert
    #     print("Correctly handled or skipped invalid input type (string).")
    # else:
    #     print(f"Handled string input, created {len(chunks_from_string)} chunks.")

    # The previous version had a path for raw strings, the current one is stricter on Document types.
    # The current version will return an empty list if a raw string is passed as `document_input`.
