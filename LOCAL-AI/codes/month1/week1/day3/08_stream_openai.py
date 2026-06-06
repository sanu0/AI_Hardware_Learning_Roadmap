"""
Day 3 — Streaming through the OpenAI-compatible endpoint (what most 3rd-party tools use).
The openai library handles the SSE protocol for you; delta.content is None on the final chunk.
Needs:  uv pip install openai
Run (venv active):  python 08_stream_openai.py
"""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # api_key ignored by Ollama

stream = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "Give me 3 tips for fast local inference."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:                               # delta is None on the final chunk
        print(delta, end="", flush=True)
print()
