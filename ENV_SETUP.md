# 環境変数設定ガイド

このプロジェクトでは、`.env`ファイルを使用してLLMプロバイダーの認証情報を管理できます。

## .envファイルの設定

プロジェクトルートに`.env`ファイルを作成し、以下の環境変数を設定してください：

### Azure OpenAI
```
AZURE_API_KEY=<your_api_key>
AZURE_API_BASE=https://<your_endpoint>.openai.azure.com/
AZURE_API_VERSION=<your_api_version>
AZURE_MODEL_NAME=azure/<your_model>
```

### AWS Bedrock
```
AWS_ACCESS_KEY_ID=<your_access_key>
AWS_SECRET_ACCESS_KEY=<your_secret_key>
AWS_REGION_NAME=ap-northeast-1
AWS_BEDROCK_MODEL_NAME=bedrock/apac.anthropic.claude-sonnet-4-20250514-v1:0
```

### Google Vertex AI
```
GOOGLE_APPLICATION_CREDENTIALS=".\\key\\<credential_file.json>"
GEMINI_MODEL_NAME=vertex_ai/gemini-2.5-pro-preview-05-06
```

## 使用方法

### 1. 環境変数から自動読み込み
```python
from llm.llm_interface import get_llm_from_env

# Azure OpenAI
llm = get_llm_from_env("azure")

# AWS Bedrock
llm = get_llm_from_env("bedrock")

# Google Vertex AI
llm = get_llm_from_env("vertexai")
```

### 2. 従来の方法（引数で直接指定）
```python
from llm.llm_interface import get_llm

# Azure OpenAI
llm = get_llm(
    provider="azure",
    model_name="azure/gpt-4",
    azure_api_key="your_key",
    azure_api_base="https://your-endpoint.openai.azure.com/",
    azure_api_version="2024-12-01-preview"
)
```

## 動作確認

環境変数が正しく設定されているかテストするには：

```bash
python llm/llm_interface.py
```

このコマンドで各プロバイダーの初期化テストが実行されます。

## セキュリティ注意事項

- `.env`ファイルは`.gitignore`に追加して、バージョン管理システムにコミットしないようにしてください
- APIキーや認証情報は適切に管理し、第三者と共有しないでください
- 本番環境では環境変数を直接システムに設定することを推奨します

## 依存関係

この機能を使用するには、以下のパッケージが必要です：

```bash
pip install python-dotenv
```

または、requirements.txtから一括インストール：

```bash
pip install -r requirements.txt