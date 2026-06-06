"""
Day 2 — Hit Ollama's REST API directly with `requests` (no ollama library).
Shows that the ollama lib is just a thin wrapper over HTTP POST to localhost:11434.
Useful for debugging and for languages with no ollama SDK (Go, Rust, Java...).
Needs:  uv pip install requests
Run (venv active):  python 03_raw_api.py
"""
import requests
import json

response = requests.post(
    'http://localhost:11434/api/chat',
    json={
        'model': 'llama3.1:8b',
        'messages': [{'role': 'user', 'content': 'List 3 benefits of local LLMs.'}],
        'stream': False,
    },
)
data = response.json()
print(data['message']['content'])
print(f"\n-- {data['eval_count']} tokens, {data['eval_count'] / (data['eval_duration']/1e9):.1f} tok/s")
