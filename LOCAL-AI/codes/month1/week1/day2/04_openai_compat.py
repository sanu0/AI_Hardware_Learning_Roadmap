"""
Day 2 — Use the official OpenAI client library, pointed at LOCAL Ollama.
Any tool/example written for OpenAI (LangChain, LlamaIndex, instructor, ...) works with
your local model just by changing base_url. api_key is required by the SDK but ignored.
Needs:  uv pip install openai
Run (venv active):  python 04_openai_compat.py
"""
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # required by the SDK but ignored by Ollama
)

response = client.chat.completions.create(
    model='llama3.1:8b',
    messages=[
        {'role': 'user', 'content': 'Suggest 3 names for a Python library that simplifies local LLMs.'}
    ],
)
print(response.choices[0].message.content)
