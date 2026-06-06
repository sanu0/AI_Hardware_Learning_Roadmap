"""
Day 3 — Streaming with the ollama library (the simplest way to print tokens live).
The single magic flag is stream=True; flush=True forces each token to the screen instantly.
Run (venv active):  python 06_stream_lib.py
"""
import ollama

stream = ollama.chat(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "Write a haiku about local AI, then explain it."}],
    stream=True,
)

for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)
print()
