"""
Day 4 — Control output length and halt early.
  num_predict -> hard cap on number of tokens generated.
  stop         -> list of strings; generation halts the moment one appears (structured output).
Run (venv active):  python 17_length_and_stop.py
"""
import ollama

MODEL = "qwen2.5:7b"

print("=== num_predict=30 (hard cap) ===")
r = ollama.chat(model=MODEL,
                messages=[{"role": "user", "content": "Explain photosynthesis."}],
                options={"num_predict": 30})
print(r["message"]["content"].strip())

print("\n=== stop at first blank line (structured) ===")
r = ollama.chat(model=MODEL,
                messages=[{"role": "user", "content": "List 5 fruits, one per line, then explain each."}],
                options={"stop": ["\n\n"]})   # halt before the explanations
print(r["message"]["content"].strip())
