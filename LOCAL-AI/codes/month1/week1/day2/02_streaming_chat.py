"""
Day 2 — Streaming Ollama chat: tokens print as they're generated (like the ChatGPT UI).
The only change vs basic chat is stream=True + printing each chunk with flush=True.
Run (venv active):  python 02_streaming_chat.py
"""
import ollama

stream = ollama.chat(
    model='llama3.1:8b',
    messages=[
        {'role': 'user', 'content': 'Write a haiku about local AI.'}
    ],
    stream=True,
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
print()  # final newline
