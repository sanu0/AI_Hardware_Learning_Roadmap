"""
Day 4 — Temperature sweep. Same prompt across temperatures (3 runs each) to see
output go from deterministic (temp=0, identical every run) to chaotic (temp=1.6).
Run (venv active):  python 14_temperature_sweep.py
"""
import ollama

MODEL = "qwen2.5:7b"
PROMPT = "Give one creative name for a coffee shop on Mars. Just the name."

for temp in [0.0, 0.4, 0.8, 1.2, 1.6]:
    print(f"\n=== temperature={temp} ===")
    for run in range(3):
        r = ollama.chat(model=MODEL,
                        messages=[{"role": "user", "content": PROMPT}],
                        options={"temperature": temp})
        print(f"  run {run+1}: {r['message']['content'].strip()}")
