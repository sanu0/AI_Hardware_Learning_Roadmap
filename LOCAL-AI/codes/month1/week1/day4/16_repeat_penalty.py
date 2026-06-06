"""
Day 4 — Repetition penalty. Small models loop without it. Compare repeat_penalty
1.0 (loops likely) -> 1.1 (clean) -> 1.3 (strict, can feel choppy).
Run (venv active):  python 16_repeat_penalty.py
"""
import ollama

MODEL = "llama3.2:3b"   # smaller model loops more readily
PROMPT = "Write a motivational paragraph about persistence. Keep going for a while."

for rp in [1.0, 1.1, 1.3]:
    print(f"\n=== repeat_penalty={rp} ===")
    r = ollama.chat(model=MODEL,
                    messages=[{"role": "user", "content": PROMPT}],
                    options={"repeat_penalty": rp, "num_predict": 200, "temperature": 0.8, "seed": 7})
    print(r["message"]["content"].strip())
