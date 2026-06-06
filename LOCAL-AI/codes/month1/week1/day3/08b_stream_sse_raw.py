"""
Day 3 — See the RAW SSE wire format the OpenAI-compatible endpoint sends.
Lines are prefixed with 'data: ' and the stream ends with a literal 'data: [DONE]'.
(Contrast with Ollama-native NDJSON in 07_stream_requests.py, which has neither.)
Needs:  uv pip install requests
Run (venv active):  python 08b_stream_sse_raw.py
"""
import requests, json

resp = requests.post(
    "http://localhost:11434/v1/chat/completions",
    json={"model": "qwen2.5:7b",
          "messages": [{"role": "user", "content": "Say hello in 3 languages."}],
          "stream": True},
    stream=True,
)
for line in resp.iter_lines():
    if not line:
        continue
    s = line.decode("utf-8")
    if s.startswith("data: "):
        payload = s[len("data: "):]
        if payload == "[DONE]":
            break
        delta = json.loads(payload)["choices"][0]["delta"].get("content", "")
        print(delta, end="", flush=True)
print()
