from langchain_litellm import ChatLiteLLM
import os
from dotenv import load_dotenv

# .envファイルを読み込み
load_dotenv()

def get_llm(provider: str,
            model_name: str = None,
            openai_api_key: str = None,
            anthropic_api_key: str = None,
            google_api_key: str = None,
            # Azure OpenAI parameters
            azure_api_key: str = None,
            azure_api_base: str = None,
            azure_api_version: str = None,
            # AWS Bedrock parameters
            aws_access_key_id: str = None,
            aws_secret_access_key: str = None,
            aws_region_name: str = None,
            # Google VertexAI parameters
            google_application_credentials: str = None,
            temperature: float = 0.1,
            verbose: bool = True):
    """
    Initializes and returns a ChatLiteLLM instance for the specified provider and model.

    Args:
        provider (str): The LLM provider ("openai", "anthropic", "google", "azure", "bedrock", "vertexai").
        model_name (str): The name of the model to use.
        openai_api_key (str, optional): API key for OpenAI. Required if provider is "openai".
        anthropic_api_key (str, optional): API key for Anthropic. Required if provider is "anthropic".
        google_api_key (str, optional): API key for Google (Gemini). Required if provider is "google".
        azure_api_key (str, optional): API key for Azure OpenAI. Required if provider is "azure".
        azure_api_base (str, optional): Base URL for Azure OpenAI. Required if provider is "azure".
        azure_api_version (str, optional): API version for Azure OpenAI. Required if provider is "azure".
        aws_access_key_id (str, optional): AWS access key ID for Bedrock. Required if provider is "bedrock".
        aws_secret_access_key (str, optional): AWS secret access key for Bedrock. Required if provider is "bedrock".
        aws_region_name (str, optional): AWS region for Bedrock. Required if provider is "bedrock".
        google_application_credentials (str, optional): Path to Google service account JSON for VertexAI. Required if provider is "vertexai".
        temperature (float): Temperature for LLM generation.
        verbose (bool): Whether to print verbose output from LiteLLM.

    Returns:
        ChatLiteLLM: An instance of the ChatLiteLLM model, or None if initialization fails.
    
    Raises:
        ValueError: If required parameters for a provider are missing.
    """
    # 環境変数から設定値を自動取得（引数で指定されていない場合）
    if azure_api_key is None:
        azure_api_key = os.getenv("AZURE_API_KEY")
    if azure_api_base is None:
        azure_api_base = os.getenv("AZURE_API_BASE")
    if azure_api_version is None:
        azure_api_version = os.getenv("AZURE_API_VERSION")
    if model_name is None and provider == "azure":
        model_name = os.getenv("AZURE_MODEL_NAME")
        
    if aws_access_key_id is None:
        aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    if aws_secret_access_key is None:
        aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if aws_region_name is None:
        aws_region_name = os.getenv("AWS_REGION_NAME")
    if model_name is None and provider == "bedrock":
        model_name = os.getenv("AWS_BEDROCK_MODEL_NAME")
        
    if google_application_credentials is None:
        google_application_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if model_name is None and provider == "vertexai":
        model_name = os.getenv("GEMINI_MODEL_NAME")

    llm_model_string = ""  # The final model string to be passed to LiteLLM
    init_params = {"temperature": temperature, "verbose": verbose} # Common parameters

    # Provider-specific logic for setting model string and API keys/URLs
    if provider == "openai":
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

    elif provider == "azure": # For Azure OpenAI models via LiteLLM
        if not azure_api_key:
            raise ValueError("Azure API Key is required for 'azure' provider. Please enter it in the LLM settings.")
        if not azure_api_base:
            raise ValueError("Azure API Base URL is required for 'azure' provider. Please enter it in the LLM settings.")
        if not azure_api_version:
            raise ValueError("Azure API Version is required for 'azure' provider. Please enter it in the LLM settings.")
        if not model_name:
            raise ValueError("Model name is required for 'azure' provider (e.g., 'gpt-35-turbo'). Please enter it in the LLM settings.")
        # LiteLLM expects "azure/" prefix for Azure OpenAI models
        # 環境変数から取得した場合は既にプレフィックスが含まれている可能性がある
        if model_name.startswith("azure/"):
            llm_model_string = model_name
        else:
            llm_model_string = f"azure/{model_name}"
        init_params["api_key"] = azure_api_key
        init_params["api_base"] = azure_api_base
        init_params["api_version"] = azure_api_version

    elif provider == "bedrock": # For AWS Bedrock models via LiteLLM
        if not aws_access_key_id:
            raise ValueError("AWS Access Key ID is required for 'bedrock' provider. Please enter it in the LLM settings.")
        if not aws_secret_access_key:
            raise ValueError("AWS Secret Access Key is required for 'bedrock' provider. Please enter it in the LLM settings.")
        if not aws_region_name:
            raise ValueError("AWS Region is required for 'bedrock' provider. Please enter it in the LLM settings.")
        if not model_name:
            raise ValueError("Model name is required for 'bedrock' provider (e.g., 'anthropic.claude-3-haiku-20240307-v1:0'). Please enter it in the LLM settings.")
        # LiteLLM expects "bedrock/" prefix for Bedrock models
        # 環境変数から取得した場合は既にプレフィックスが含まれている可能性がある
        if model_name.startswith("bedrock/"):
            llm_model_string = model_name
        else:
            llm_model_string = f"bedrock/{model_name}"
        init_params["aws_access_key_id"] = aws_access_key_id
        init_params["aws_secret_access_key"] = aws_secret_access_key
        init_params["aws_region_name"] = aws_region_name

    elif provider == "vertexai": # For Google VertexAI models via LiteLLM
        if not google_application_credentials:
            raise ValueError("Google Application Credentials path is required for 'vertexai' provider. Please enter it in the LLM settings.")
        if not model_name:
            raise ValueError("Model name is required for 'vertexai' provider (e.g., 'gemini-pro'). Please enter it in the LLM settings.")
        # LiteLLM expects "vertex_ai/" prefix for VertexAI models
        # 環境変数から取得した場合は既にプレフィックスが含まれている可能性がある
        if model_name.startswith("vertex_ai/"):
            llm_model_string = model_name
        else:
            llm_model_string = f"vertex_ai/{model_name}"
        # Set environment variable for Google Application Credentials
        if google_application_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_application_credentials
        
    else:
        # Should not be reached if UI selectbox is synced with this logic
        raise ValueError(f"Unsupported LLM provider: {provider}. Supported providers are 'openai', 'anthropic', 'google', 'azure', 'bedrock', 'vertexai'.")

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
            f"and the service is running and accessible."
        )
        print(error_message)
        # Re-raise a ConnectionError to be caught by the UI for user feedback
        raise ConnectionError(error_message)


def get_llm_from_env(provider: str, temperature: float = 0.1, verbose: bool = True):
    """
    環境変数から設定を読み込んでLLMを初期化する便利な関数
    
    Args:
        provider (str): LLMプロバイダー ("azure", "bedrock", "vertexai")
        temperature (float): 生成温度
        verbose (bool): 詳細出力の有無
    
    Returns:
        ChatLiteLLM: 初期化されたLLMインスタンス
    """
    return get_llm(provider=provider, temperature=temperature, verbose=verbose)


if __name__ == '__main__':
    # .env ファイルからの読み込みテスト
    print("環境変数の確認:")
    print(f"AZURE_API_KEY: {'設定済み' if os.getenv('AZURE_API_KEY') else '未設定'}")
    print(f"AZURE_MODEL_NAME: {os.getenv('AZURE_MODEL_NAME', '未設定')}")
    print(f"AWS_BEDROCK_MODEL_NAME: {os.getenv('AWS_BEDROCK_MODEL_NAME', '未設定')}")
    print(f"GEMINI_MODEL_NAME: {os.getenv('GEMINI_MODEL_NAME', '未設定')}")
    
    # Azure OpenAI テスト
    print("\nAzure OpenAI LLM初期化テスト...")
    try:
        if os.getenv("AZURE_API_KEY"):
            azure_llm = get_llm_from_env(provider="azure")
            if azure_llm:
                print("Azure OpenAI LLM初期化成功")
                # response = azure_llm.invoke("こんにちは")
                # print(f"Azure LLM応答: {response.content}")
        else:
            print("AZURE_API_KEY が環境変数に設定されていません")
    except Exception as e:
        print(f"Azure テスト失敗: {e}")
    
    # AWS Bedrock テスト
    print("\nAWS Bedrock LLM初期化テスト...")
    try:
        if os.getenv("AWS_ACCESS_KEY_ID"):
            bedrock_llm = get_llm_from_env(provider="bedrock")
            if bedrock_llm:
                print("AWS Bedrock LLM初期化成功")
        else:
            print("AWS_ACCESS_KEY_ID が環境変数に設定されていません")
    except Exception as e:
        print(f"Bedrock テスト失敗: {e}")
    
    # Google VertexAI テスト
    print("\nGoogle VertexAI LLM初期化テスト...")
    try:
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            vertexai_llm = get_llm_from_env(provider="vertexai")
            if vertexai_llm:
                print("Google VertexAI LLM初期化成功")
        else:
            print("GOOGLE_APPLICATION_CREDENTIALS が環境変数に設定されていません")
    except Exception as e:
        print(f"VertexAI テスト失敗: {e}")
