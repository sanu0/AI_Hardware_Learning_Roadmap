"""
Day 3 — Streaming via raw HTTP, parsing Ollama's NDJSON (newline-delimited JSON) by hand.
Two stream flags are needed: "stream": True in the JSON (tell Ollama) AND stream=True in
requests.post (tell requests not to buffer the whole body). Each line is one JSON object.
Needs:  uv pip install requests
Run (venv active):  python 07_stream_requests.py
"""
import requests
import json

resp = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "qwen2.5:7b",
        "messages": [{"role": "user", "content": "Count from 1 to 10 with a word each."}],
        "stream": True,
    },
    stream=True,
)

for line in resp.iter_lines():        # each line is one complete JSON object
    if not line:
        continue
    obj = json.loads(line)
    print(obj["message"]["content"], end="", flush=True)
    if obj.get("done"):               # final object carries timing stats
        print(f"\n\n-- {obj.get('eval_count')} tokens, "
              f"{obj['eval_count'] / (obj['eval_duration']/1e9):.1f} tok/s")
        break
