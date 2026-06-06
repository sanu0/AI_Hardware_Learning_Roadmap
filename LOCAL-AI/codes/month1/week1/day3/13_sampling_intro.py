"""
Day 3 — Feel `temperature`: low = deterministic & safe, high = creative & chaotic.
Runs the same prompt 3x at each temperature. At 0.0 the outputs are ~identical;
at 1.3 they diverge wildly. (Full parameter deep-dive is Day 4.)
Run (venv active):  python 13_sampling_intro.py
"""
import ollama

MODEL = "qwen2.5:7b"
PROMPT = "Invent a name for a coffee shop on Mars. Just the name."

for temp in [0.0, 0.7, 1.3]:
    print(f"\n--- temperature={temp} ---")
    for _ in range(3):
        r = ollama.chat(model=MODEL,
                        messages=[{"role": "user", "content": PROMPT}],
                        options={"temperature": temp, "seed": None})
        print(" ", r["message"]["content"].strip())
