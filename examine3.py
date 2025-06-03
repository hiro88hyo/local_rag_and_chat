import os
import asyncio
from langchain_litellm import ChatLiteLLM
from dotenv import load_dotenv

# .env ファイルから環境変数を読み込む
load_dotenv()

# for Azure
os.environ["AZURE_API_KEY"]
os.environ["AZURE_API_BASE"] 
os.environ["AZURE_API_VERSION"]
os.environ["AZURE_MODEL_NAME"]

# for Bedrock
os.environ["AWS_ACCESS_KEY_ID"]
os.environ["AWS_SECRET_ACCESS_KEY"]
os.environ["AWS_REGION_NAME"]
os.environ["AWS_BEDROCK_MODEL_NAME"]

# for VertexAI
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
os.environ["GEMINI_MODEL_NAME"]

async def call_llm(llm, prompt):
    response = await llm.ainvoke(prompt)
    return response

async def main():
    # azure call
    llm_azure = ChatLiteLLM(model=os.environ["AZURE_MODEL_NAME"])
    # bedrock call
    llm_bedrock = ChatLiteLLM(model=os.environ["AWS_BEDROCK_MODEL_NAME"])
    # vertexai call
    llm_vertexai = ChatLiteLLM(model=os.environ["GEMINI_MODEL_NAME"])
    response = await call_llm(llm_bedrock, "こんにちは、あなたは何ができますか？")
    print(response)

asyncio.run(main())