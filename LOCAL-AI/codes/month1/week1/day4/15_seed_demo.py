"""
Day 4 — Seed = reproducibility. With temperature>0, a fixed seed reproduces the exact
output; with temperature=0 the output is deterministic regardless of seed.
Run (venv active):  python 15_seed_demo.py
"""
import ollama

MODEL = "qwen2.5:7b"
PROMPT = "Invent a fantasy character name."


def gen(temperature, seed):
    r = ollama.chat(model=MODEL, messages=[{"role": "user", "content": PROMPT}],
                    options={"temperature": temperature, "seed": seed})
    return r["message"]["content"].strip()


print("temp=1.0, seed=42 (run twice) — should MATCH:")
print(" ", gen(1.0, 42)); print(" ", gen(1.0, 42))

print("\ntemp=1.0, different seeds — should DIFFER:")
print(" ", gen(1.0, 1)); print(" ", gen(1.0, 2))

print("\ntemp=0 (seed irrelevant) — always the same:")
print(" ", gen(0, 1)); print(" ", gen(0, 999))
