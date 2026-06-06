"""
Day 3 — Prove the API is STATELESS: the model has no memory across separate requests,
then show that YOU create "memory" by sending the whole conversation history each turn.
This is the single most important concept for building chat apps.
Run (venv active):  python 10_stateless_proof.py
"""
import ollama

MODEL = "qwen2.5:7b"

print("=== Two SEPARATE requests (no shared history) ===")
ollama.chat(model=MODEL, messages=[{"role": "user", "content": "My name is Alex."}])
r = ollama.chat(model=MODEL, messages=[{"role": "user", "content": "What is my name?"}])
print("Q: What is my name? →", r["message"]["content"].strip(), "\n")   # it won't know

print("=== ONE request WITH history (memory you built) ===")
messages = [
    {"role": "user",      "content": "My name is Alex."},
    {"role": "assistant", "content": "Nice to meet you, Alex!"},
    {"role": "user",      "content": "What is my name?"},
]
r = ollama.chat(model=MODEL, messages=messages)
print("Q: What is my name? →", r["message"]["content"].strip())          # now it knows
