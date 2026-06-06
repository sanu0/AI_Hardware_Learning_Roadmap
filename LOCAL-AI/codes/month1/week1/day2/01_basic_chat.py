"""
Day 2 — Basic Ollama chat (the simplest possible programmatic call).
Sends one user message and prints the full reply. Your first "programming an LLM" moment.
Run (venv active):  python 01_basic_chat.py
"""
import ollama

response = ollama.chat(
    model='llama3.1:8b',
    messages=[
        {'role': 'user', 'content': 'Why is the sky blue? Answer in 2 sentences.'}
    ]
)
print(response['message']['content'])
