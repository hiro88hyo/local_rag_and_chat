from langchain_community.chat_models import ChatLiteLLM

def get_llm(provider: str, 
            model_name: str, 
            ollama_base_url: str = None, 
            openai_api_key: str = None,
            anthropic_api_key: str = None,
            google_api_key: str = None,
            temperature: float = 0.1,
            verbose: bool = True):
    """
    Initializes and returns a ChatLiteLLM instance for the specified provider and model.

    Args:
        provider (str): The LLM provider ("ollama", "openai", "anthropic", "google").
        model_name (str): The name of the model to use.
        ollama_base_url (str, optional): Base URL for Ollama. Required if provider is "ollama".
        openai_api_key (str, optional): API key for OpenAI. Required if provider is "openai".
        anthropic_api_key (str, optional): API key for Anthropic. Required if provider is "anthropic".
        google_api_key (str, optional): API key for Google (Gemini). Required if provider is "google".
        temperature (float): Temperature for LLM generation.
        verbose (bool): Whether to print verbose output from LiteLLM.

    Returns:
        ChatLiteLLM: An instance of the ChatLiteLLM model, or None if initialization fails.
    
    Raises:
        ValueError: If required parameters for a provider are missing.
    """
    llm_model_string = ""  # The final model string to be passed to LiteLLM
    init_params = {"temperature": temperature, "verbose": verbose} # Common parameters

    # Provider-specific logic for setting model string and API keys/URLs
    if provider == "ollama":
        if not ollama_base_url:
            raise ValueError("Ollama Base URL is required for 'ollama' provider. Please enter it in the LLM settings.")
        if not model_name:
            raise ValueError("Model name is required for 'ollama' provider. Please enter it in the LLM settings.")
        # LiteLLM expects "ollama/" prefix for Ollama models
        llm_model_string = f"ollama/{model_name}"
        init_params["api_base"] = ollama_base_url
    
    elif provider == "openai":
        if not openai_api_key:
            raise ValueError("OpenAI API Key is required for 'openai' provider. Please enter it in the LLM settings.")
        if not model_name:
            raise ValueError("Model name is required for 'openai' provider (e.g., 'gpt-3.5-turbo'). Please enter it in the LLM settings.")
        # For OpenAI, LiteLLM can often infer the provider from the model name if it's standard (e.g., "gpt-3.5-turbo")
        # or you can explicitly use "openai/gpt-3.5-turbo".
        llm_model_string = model_name 
        init_params["api_key"] = openai_api_key
        # LiteLLM can also pick up OPENAI_API_KEY from environment variables if not passed explicitly.
    
    elif provider == "anthropic":
        if not anthropic_api_key:
            raise ValueError("Anthropic API Key is required for 'anthropic' provider. Please enter it in the LLM settings.")
        if not model_name:
            raise ValueError("Model name is required for 'anthropic' provider (e.g., 'claude-3-haiku-20240307'). Please enter it in the LLM settings.")
        # LiteLLM expects "anthropic/" prefix for Claude models
        llm_model_string = f"anthropic/{model_name}"
        init_params["api_key"] = anthropic_api_key

    elif provider == "google": # For Google Gemini models via LiteLLM
        if not google_api_key:
            raise ValueError("Google API Key is required for 'google' (Gemini) provider. Please enter it in the LLM settings.")
        if not model_name: 
            raise ValueError("Model name is required for 'google' (Gemini) provider (e.g., 'gemini-pro'). Please enter it in the LLM settings.")
        # LiteLLM uses "gemini/" prefix for Google models
        llm_model_string = f"gemini/{model_name}" if not model_name.startswith("gemini/") else model_name
        init_params["api_key"] = google_api_key
        
    else:
        # Should not be reached if UI selectbox is synced with this logic
        raise ValueError(f"Unsupported LLM provider: {provider}. Supported providers are 'ollama', 'openai', 'anthropic', 'google'.")

    # Attempt to initialize the ChatLiteLLM instance
    try:
        llm = ChatLiteLLM(model=llm_model_string, **init_params)
        print(f"Successfully initialized LLM: {llm_model_string} for provider {provider} with temp {temperature}")
        return llm
    except Exception as e:
        # NFQ3.2 - More specific error for connection/initialization failure
        error_message = (
            f"Failed to connect or initialize LLM '{llm_model_string}' for provider '{provider}'. "
            f"Details: {e}. "
            f"Please ensure the model name is correct, the API key (if applicable) is valid, "
            f"and the service (e.g., Ollama server at '{ollama_base_url}' if using Ollama) is running and accessible."
        )
        print(error_message)
        # Re-raise a ConnectionError to be caught by the UI for user feedback
        raise ConnectionError(error_message)


if __name__ == '__main__':
    # Example usage and test (requires environment variables or direct key passing for cloud models)
    
    # Ollama Test
    print("Testing Ollama LLM initialization...")
    try:
        # Ensure Ollama server is running and model 'llama2' (or your test model) is pulled.
        ollama_llm = get_llm(provider="ollama", model_name="llama2", ollama_base_url="http://localhost:11434")
        if ollama_llm:
            print("Ollama LLM initialized. Testing invocation...")
            response = ollama_llm.invoke("Why is the sky blue?")
            print(f"Ollama LLM response: {response.content}")
    except Exception as e:
        print(f"Ollama test failed: {e}")

    # OpenAI Test (Requires OPENAI_API_KEY environment variable or pass directly)
    # print("\nTesting OpenAI LLM initialization...")
    # try:
    #     openai_api_key_env = os.getenv("OPENAI_API_KEY")
    #     if openai_api_key_env:
    #         openai_llm = get_llm(provider="openai", model_name="gpt-3.5-turbo", openai_api_key=openai_api_key_env)
    #         if openai_llm:
    #             print("OpenAI LLM initialized. Testing invocation...")
    #             response = openai_llm.invoke("Why is the sky blue?")
    #             print(f"OpenAI LLM response: {response.content}")
    #     else:
    #         print("OPENAI_API_KEY not found in environment. Skipping OpenAI test.")
    # except Exception as e:
    #     print(f"OpenAI test failed: {e}")

    # Add similar test blocks for Anthropic and Google if you have API keys and want to test them.
    # Ensure to handle API key retrieval securely (e.g., from env variables).
