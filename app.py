import streamlit as st
import os
import gc
from dotenv import load_dotenv
from processing.loaders import load_txt_file, load_md_file, load_pdf_file, load_docx_file
from processing.chunking import chunk_document
from processing.vector_store import get_vector_store, get_vector_store_context, add_documents_to_store, clear_vector_store, CHROMA_DB_PATH
from langchain_core.prompts import PromptTemplate # For creating dynamic prompts
from llm.llm_interface import get_llm # Interface to LLM models

import traceback

# .envファイルから環境変数を読み込み
load_dotenv()

# Initialize session state for conversation history if it doesn't exist
# This helps maintain chat history across Streamlit reruns.
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Function to load or initialize the vector store
# This function will be called when needed, e.g., before a search
def load_or_initialize_vector_store(collection_name="main_document_collection"):
    if not os.path.exists(CHROMA_DB_PATH):
        # st.sidebar.warning("Vector store not found. Please ingest documents first.")
        return None # Indicate that the store is not ready
    try:
        vector_store = get_vector_store(collection_name=collection_name)
        # st.sidebar.success("ChromaDB vector store loaded.")
        return vector_store
    except Exception as e:
        # st.sidebar.error(f"Failed to load vector store: {e}")
        return None


# Main application function
def main():
    st.set_page_config(layout="wide")
    st.title("Document-Aware AI Assistant")

    # --- Tabs for Ingestion and Chat ---
    tab1, tab2 = st.tabs(["📚 Document Ingestion", "💬 Chat with Documents"])

    # --- Tab 1: Document Ingestion ---
    with tab1:
        st.header("Manage Document Corpus")
        st.info(f"Vector database is stored locally in: ./{CHROMA_DB_PATH}")
        
        # --- Configuration for Ingestion ---
        st.sidebar.header("Ingestion Configuration")
        directory_path = st.sidebar.text_input("Enter directory path to ingest documents:", key="dir_path_input_ingest")
        clear_db_option = st.sidebar.checkbox("Clear existing data before ingest", value=False, key="clear_db_checkbox_ingest")
        ingest_button = st.sidebar.button("Start Ingestion Process", key="ingest_button_ingest")

        ingest_status_area = st.empty()
        ingest_summary_area = st.sidebar.empty() # For ingestion summary

        if ingest_button and directory_path:
            if not os.path.isdir(directory_path): # NFQ3.2 - Improved error message
                ingest_status_area.error(f"Invalid directory path: '{directory_path}'. Please enter a valid path to a directory containing documents.")
            else:
                ingest_status_area.info(f"Starting ingestion from directory: {directory_path}")

                if clear_db_option:
                    if os.path.exists(CHROMA_DB_PATH):
                        ingest_status_area.info(f"Clear existing data option selected. Attempting to delete current vector store at ./{CHROMA_DB_PATH}...")
                        success, message = clear_vector_store()
                        if success:
                            ingest_status_area.success(message)
                        else: # NFQ3.2 - Improved error message
                            ingest_status_area.error(f"Failed to delete vector store: {message}. Please check permissions or delete manually if necessary. Ingestion halted.")
                            st.stop()
                    else:
                        ingest_status_area.info(f"Clear existing data option selected, but no existing store found at ./{CHROMA_DB_PATH}.")
                
                ingest_status_area.info("Ready to start ingestion process...")

                files_processed_count = 0
                total_chunks_added = 0
                errors_encountered = []

                try:
#                    files_to_process = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

                    files_to_process = []
                    for root, dirs, files in os.walk(directory_path):
                        for file_name in files:
                            # os.path.join() を使ってフルパスを作成
                            full_path = os.path.join(root, file_name)
                            files_to_process.append(os.path.abspath(full_path)) # 絶対パスに変換して追加

                    if not files_to_process:
                        ingest_status_area.write("No files found in this directory.")
                    else:
                        ingest_status_area.write(f"Found {len(files_to_process)} file(s) to process.")
                        
                        progress_bar = ingest_status_area.progress(0)
                        progress_text_area = st.empty()

                        for i, filename in enumerate(files_to_process):
                            filepath = os.path.join(directory_path, filename)
                            progress_text_area.text(f"Processing file: {filename} ({i+1}/{len(files_to_process)})")
                            
                            loaded_data = None
                            try: # Individual file processing block NFQ3.2
                                # Determine file type and load content
                                if filename.endswith(".txt"): loaded_data = load_txt_file(filepath)
                                elif filename.endswith(".md"): loaded_data = load_md_file(filepath)
                                elif filename.endswith(".pdf"): loaded_data = load_pdf_file(filepath) 
                                elif filename.endswith(".docx"): loaded_data = load_docx_file(filepath)
                                else:
                                    st.warning(f"File '{filename}': Unsupported file type. Skipping.")
                                    errors_encountered.append(f"{filename}: Unsupported file type")
                                    continue # Move to the next file

                                if not loaded_data: # NFQ3.2 - Improved error message
                                    st.warning(f"File '{filename}': No data loaded (file might be empty or corrupted). Skipping.")
                                    errors_encountered.append(f"{filename}: No data loaded")
                                    continue
                                
                                progress_text_area.text(f"Chunking {filename}...")
                                all_chunks = chunk_document(loaded_data) # Chunk the loaded document

                                if not all_chunks: # NFQ3.2 - Improved error message
                                    st.warning(f"File '{filename}': No chunks created (document might be empty or too small). Skipping.")
                                    errors_encountered.append(f"{filename}: No chunks created")
                                    continue
                                
                                progress_text_area.text(f"Adding {len(all_chunks)} chunks from {filename} to vector store...")
                                # Use context manager to ensure proper cleanup
                                try:
                                    with get_vector_store_context(collection_name="main_document_collection") as current_vector_store:
                                        # Add processed chunks to the vector store
                                        num_added = add_documents_to_store(all_chunks, current_vector_store, filepath)
                                except Exception as e_vector_store:
                                    st.error(f"Error initializing vector store for file '{filename}': {e_vector_store}. Skipping this file.")
                                    errors_encountered.append(f"{filename}: Vector store error - {e_vector_store}")
                                    continue
                                
                                if num_added > 0:
                                    total_chunks_added += num_added
                                    files_processed_count += 1
                                else: # Should ideally not happen if all_chunks is not empty NFQ3.2
                                    st.warning(f"File '{filename}': Chunks were created but not added to the store. This might indicate an issue with the vector store itself.")
                                    errors_encountered.append(f"{filename}: Chunks created but not added to store")

                            except Exception as e_file_proc: # NFQ3.2 - Catch errors during individual file load/chunk/add
                                print(traceback.format_exc())
                                st.error(f"Error processing file '{filename}': {e_file_proc}. Skipping this file.")
                                errors_encountered.append(f"{filename}: {e_file_proc}")
                            
                            # Update progress bar after each file
                            progress_bar.progress((i + 1) / len(files_to_process))
                        
                        progress_text_area.empty() # Clear the "Processing file..." message
                        # Display overall ingestion results
                        ingest_status_area.success(f"Ingestion complete! Processed {files_processed_count}/{len(files_to_process)} files. Added {total_chunks_added} new chunks to the database.")

                        # Display summary of any issues in the sidebar
                        if errors_encountered: # NFQ3.2 - Clearer summary of issues
                            ingest_summary_area.subheader("Ingestion Issues Summary")
                            ingest_summary_area.warning(f"{len(errors_encountered)} file(s) encountered issues:")
                            for error_msg in errors_encountered:
                                ingest_summary_area.write(f"- {error_msg}")
                        else:
                            ingest_summary_area.subheader("Ingestion Summary")
                            ingest_summary_area.success("All files processed successfully.")
                        
                        ingest_summary_area.write(f"Files successfully processed: {files_processed_count}")
                        ingest_summary_area.write(f"Total new chunks added to vector store: {total_chunks_added}")


                except Exception as e_ingest_critical: # NFQ3.2 - Catch critical errors in the overall ingestion loop
                    ingest_status_area.error(f"A critical error occurred during the ingestion process: {e_ingest_critical}. Check console for details. Some files may not have been processed.")
        elif ingest_button and not directory_path: # NFQ3.2 - Improved error message
            ingest_status_area.warning("Directory path is empty. Please enter a valid directory path for ingestion.")

    # --- Tab 2: Chat with Documents ---
    with tab2:
        st.header("Ask Questions About Your Documents")
        # Placeholder for chat-specific status messages (e.g., LLM loading)
        chat_status_area = st.empty() 

        # --- LLM Configuration UI in Sidebar ---
        # NFQ5.1: Tooltips and improved layout for LLM settings
        with st.sidebar.expander("LLM Settings", expanded=True):
            # Initialize session state for LLM settings if they don't exist (idempotent)
            # 環境変数から値を取得し、存在しない場合はデフォルト値を使用
            if 'llm_provider' not in st.session_state: st.session_state.llm_provider = "OpenAI"
            if 'llm_model_name' not in st.session_state: st.session_state.llm_model_name = "gpt-3.5-turbo" # Default for OpenAI
            if 'openai_api_key' not in st.session_state: st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")
            if 'anthropic_api_key' not in st.session_state: st.session_state.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
            if 'google_api_key' not in st.session_state: st.session_state.google_api_key = os.getenv("GOOGLE_API_KEY", "")
            # Azure OpenAI settings
            if 'azure_api_key' not in st.session_state: st.session_state.azure_api_key = os.getenv("AZURE_API_KEY", "")
            if 'azure_api_base' not in st.session_state: st.session_state.azure_api_base = os.getenv("AZURE_API_BASE", "")
            if 'azure_api_version' not in st.session_state: st.session_state.azure_api_version = os.getenv("AZURE_API_VERSION", "2024-02-01")
            # AWS Bedrock settings
            if 'aws_access_key_id' not in st.session_state: st.session_state.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "")
            if 'aws_secret_access_key' not in st.session_state: st.session_state.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
            if 'aws_region_name' not in st.session_state: st.session_state.aws_region_name = os.getenv("AWS_REGION_NAME", "us-east-1")
            # Google VertexAI settings
            if 'google_application_credentials' not in st.session_state: st.session_state.google_application_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

            # LLM Provider Selection
            st.session_state.llm_provider = st.selectbox(
                "Select LLM Provider",
                ["OpenAI", "Anthropic", "Google", "Azure", "Bedrock", "VertexAI"],
                key="llm_provider_select" # Unique key for the widget
            )

            # Model Name Input - Default changes based on provider
            # 環境変数からモデル名を取得し、存在しない場合はデフォルト値を使用
            def get_model_name_from_env(env_key, default_value):
                """環境変数からモデル名を取得し、プレフィックスを除去"""
                env_value = os.getenv(env_key, default_value)
                # プレフィックス（azure/, bedrock/, vertex_ai/など）を除去
                if "/" in env_value:
                    return env_value.split("/", 1)[1]
                return env_value
            
            env_model_names = {
                "OpenAI": os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo"),
                "Anthropic": os.getenv("ANTHROPIC_MODEL_NAME", "claude-3-haiku-20240307"),
                "Google": os.getenv("GOOGLE_MODEL_NAME", "gemini-pro"),
                "Azure": get_model_name_from_env("AZURE_MODEL_NAME", "gpt-35-turbo"),
                "Bedrock": get_model_name_from_env("AWS_BEDROCK_MODEL_NAME", "anthropic.claude-3-haiku-20240307-v1:0"),
                "VertexAI": get_model_name_from_env("GEMINI_MODEL_NAME", "gemini-pro")
            }
            default_models = env_model_names
            # Get the model name stored for the current provider, or use the default if none stored
            current_model_for_provider = st.session_state.get(f"{st.session_state.llm_provider}_model_name", default_models[st.session_state.llm_provider])
            
            st.session_state.llm_model_name = st.text_input(
                "Model Name", 
                value=current_model_for_provider, 
                key=f"llm_model_name_input_{st.session_state.llm_provider}", # Unique key for re-rendering on provider change
                help="Enter the exact model name (e.g., 'llama2', 'gpt-3.5-turbo', 'claude-3-haiku-20240307', 'gemini-pro'). Refer to provider documentation for available models."
            )
            # Store this model name specific to the provider to remember choices across provider switches
            st.session_state[f"{st.session_state.llm_provider}_model_name"] = st.session_state.llm_model_name

            # Conditional Inputs based on Provider
            if st.session_state.llm_provider == "OpenAI":
                st.session_state.openai_api_key = st.text_input(
                    "OpenAI API Key", type="password", 
                    value=st.session_state.openai_api_key, key="openai_api_key_input", help="Your OpenAI API key."
                )
            elif st.session_state.llm_provider == "Anthropic":
                st.session_state.anthropic_api_key = st.text_input(
                    "Anthropic API Key", type="password",
                    value=st.session_state.anthropic_api_key, key="anthropic_api_key_input", help="Your Anthropic API key."
                )
            elif st.session_state.llm_provider == "Google":
                st.session_state.google_api_key = st.text_input(
                    "Google API Key", type="password",
                    value=st.session_state.google_api_key, key="google_api_key_input", help="Your Google API key for Gemini models."
                )
            elif st.session_state.llm_provider == "Azure":
                st.session_state.azure_api_key = st.text_input(
                    "Azure API Key", type="password",
                    value=st.session_state.azure_api_key, key="azure_api_key_input", help="Your Azure OpenAI API key."
                )
                st.session_state.azure_api_base = st.text_input(
                    "Azure API Base URL",
                    value=st.session_state.azure_api_base, key="azure_api_base_input",
                    help="Your Azure OpenAI endpoint URL (e.g., https://your-resource.openai.azure.com/)."
                )
                st.session_state.azure_api_version = st.text_input(
                    "Azure API Version",
                    value=st.session_state.azure_api_version, key="azure_api_version_input",
                    help="Azure OpenAI API version (e.g., 2024-02-01)."
                )
            elif st.session_state.llm_provider == "Bedrock":
                st.session_state.aws_access_key_id = st.text_input(
                    "AWS Access Key ID", type="password",
                    value=st.session_state.aws_access_key_id, key="aws_access_key_id_input", help="Your AWS access key ID."
                )
                st.session_state.aws_secret_access_key = st.text_input(
                    "AWS Secret Access Key", type="password",
                    value=st.session_state.aws_secret_access_key, key="aws_secret_access_key_input", help="Your AWS secret access key."
                )
                st.session_state.aws_region_name = st.text_input(
                    "AWS Region",
                    value=st.session_state.aws_region_name, key="aws_region_name_input",
                    help="AWS region for Bedrock (e.g., us-east-1)."
                )
            elif st.session_state.llm_provider == "VertexAI":
                st.session_state.google_application_credentials = st.text_input(
                    "Google Application Credentials Path",
                    value=st.session_state.google_application_credentials, key="google_application_credentials_input",
                    help="Path to your Google service account JSON file for VertexAI authentication."
                )
            
            # NFQ4.1 - Enhanced API Key Warning
            st.warning(
                "API keys are stored in session state for this local application instance. "
                "For production or shared environments, it is strongly recommended to use "
                "environment variables or other secure secret management practices. "
                "LiteLLM can automatically pick up API keys from environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`)."
            )


        # Check if vector store exists without keeping it loaded
        if not os.path.exists(CHROMA_DB_PATH):
            st.warning("Vector store is not available. Please ingest documents in the 'Document Ingestion' tab first.")
        else:
            st.success("Vector store is available. Ready for chat.")

        # Display existing conversation history
        if st.session_state.conversation_history:
            for i, message in enumerate(st.session_state.conversation_history):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    # Display retrieved documents if they exist for that message
                    if "retrieved_docs" in message and message["retrieved_docs"]:
                        with st.expander("Retrieved Context", expanded=False):
                            for doc_idx, doc in enumerate(message["retrieved_docs"]):
                                st.markdown(f"**Document Chunk {doc_idx+1} (Source: {doc.metadata.get('file_name', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')})**")
                                st.caption(doc.page_content[:300] + "...") # Display a snippet

        # Input field for user's question
        user_question = st.chat_input("Ask your question:")

        if user_question:
            if not os.path.exists(CHROMA_DB_PATH): # NFQ3.2 - Improved error message
                st.error("Cannot process question: Vector store is not available. Please ingest documents or check vector store path and permissions.")
                # Add user question to history to show it was received, even if not processed
                st.session_state.conversation_history.append({"role": "user", "content": user_question, "retrieved_docs": []})
                st.rerun() # Rerun to display the user message and error immediately
            else:
                # Add user's question to history
                st.session_state.conversation_history.append({"role": "user", "content": user_question, "retrieved_docs": []})
                
                llm_response_content = "An unexpected error occurred. Could not generate LLM response." # Default error
                retrieved_docs_for_history = []

                try:
                    # 1. Use context manager for vector store retrieval
                    chat_status_area.text("Loading vector store and retrieving relevant documents...") # Temporary status
                    with get_vector_store_context(collection_name="main_document_collection") as chat_vector_store:
                        retrieved_docs_with_scores = chat_vector_store.similarity_search_with_score(user_question, k=3)
                        retrieved_docs_for_history = [doc for doc, score in retrieved_docs_with_scores]
                    
                    chat_status_area.empty() # Clear "Retrieving..." message

                    # 2. Construct context string for the LLM prompt
                    context_str = "\n\n".join([doc.page_content for doc in retrieved_docs_for_history])
                    if not retrieved_docs_for_history: # Check if the list is empty
                        context_str = "No relevant context found in the documents for your query."
                        # Inform user if no specific context is found
                        st.info("No specific document chunks found to answer your question. The LLM will answer based on its general knowledge.")


                    # 3. Define the prompt template (NFQ3.2 - Improved prompt for no context)
                    prompt_template = PromptTemplate.from_template(
                        """Based on the following context, answer the question. 
                        If the context does not provide sufficient information or is empty, try to answer based on general knowledge but clearly state that the answer is not derived from the provided documents.
                        Context:
                        {context}
                        Question: {question}"""
                    )

                    # 4. Initialize LLM based on session state settings
                    llm = None # Ensure llm is defined in this scope
                    chat_status_area.text("Initializing LLM...") # Indicate LLM loading
                    try:
                        llm = get_llm(
                            provider=st.session_state.llm_provider.lower(), # Ensure provider is lowercase for get_llm
                            model_name=st.session_state.llm_model_name,
                            openai_api_key=st.session_state.get("openai_api_key"),
                            anthropic_api_key=st.session_state.get("anthropic_api_key"),
                            google_api_key=st.session_state.get("google_api_key"),
                            # Azure OpenAI parameters
                            azure_api_key=st.session_state.get("azure_api_key"),
                            azure_api_base=st.session_state.get("azure_api_base"),
                            azure_api_version=st.session_state.get("azure_api_version"),
                            # AWS Bedrock parameters
                            aws_access_key_id=st.session_state.get("aws_access_key_id"),
                            aws_secret_access_key=st.session_state.get("aws_secret_access_key"),
                            aws_region_name=st.session_state.get("aws_region_name"),
                            # Google VertexAI parameters
                            google_application_credentials=st.session_state.get("google_application_credentials")
                        )
                        chat_status_area.text("LLM Initialized. Generating response...") # Update status
                    except ValueError as ve: # NFQ3.2 - Improved error message for config issues
                        llm_response_content = f"LLM Configuration Error: {ve}. Please check settings in the sidebar."
                        st.error(llm_response_content)
                        chat_status_area.empty()
                    except ConnectionError as ce: # NFQ3.2 - Improved error message for connection issues
                        llm_response_content = f"LLM Connection Error: {ce}. Ensure the LLM server/API is reachable and configured correctly."
                        st.error(llm_response_content)
                        chat_status_area.empty()
                    except Exception as e_init: # NFQ3.2 - Catch any other init errors
                        llm_response_content = f"Unexpected error initializing LLM: {e_init}. Check console for details."
                        st.error(llm_response_content)
                        chat_status_area.empty()


                    if llm:
                        # 5. Generate response if LLM initialized successfully
                        try:
                            final_prompt = prompt_template.format(context=context_str, question=user_question)
                            llm_response = llm.invoke(final_prompt)
                            llm_response_content = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
                            chat_status_area.empty() # Clear "Generating response..."
                        except Exception as e_invoke: # NFQ3.2 - Improved error message for invocation issues
                            llm_response_content = f"Error during LLM response generation: {e_invoke}. The LLM may be unavailable or the request timed out."
                            st.error(llm_response_content)
                            chat_status_area.empty()
                    # If LLM initialization failed, and no specific error message was set for llm_response_content yet
                    elif not ('llm_response_content' in locals() and llm_response_content.startswith("LLM")): # NFQ3.2
                        llm_response_content = "LLM could not be initialized due to previous errors. Please check settings and error messages above."
                        chat_status_area.empty() # Clear status
                
                except Exception as e_outer: # NFQ3.2 - Catch errors in retrieval or prompt formatting
                    llm_response_content = f"Error in RAG process (e.g., document retrieval): {e_outer}. Check console for details."
                    st.error(llm_response_content)
                    chat_status_area.empty() # Clear progress on error
                
                # Add assistant's response (or error message) to history
                st.session_state.conversation_history.append({
                    "role": "assistant", 
                    "content": llm_response_content, 
                    "retrieved_docs": retrieved_docs_for_history 
                })
                st.rerun() # Rerun to display the new messages

if __name__ == "__main__":
    main()

