#Note: The openai-python library support for Azure OpenAI is in preview.

import os
import openai
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv())
_



openai.api_type = "azure"
openai.api_base = "https://convaip-sbx-openai.openai.azure.com/"
openai.api_version = "2023-03-15-preview"
#openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("AZURE_OPENAI_KEY")
openai.proxy = os.getenv("HTTP_PROXY")



response = openai.ChatCompletion.create(
  engine="chat-gpt-0301",
  messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}],
  temperature=0.7,
  max_tokens=800,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)

print(response)