"""
Day 4 — Task presets. Run the same prompt under factual / balanced / creative presets
to feel how the parameter bundles change tone and reliability. Reuse these dicts
everywhere (and bake your favorite into a Modelfile on Day 5).
Run (venv active):  python presets.py
"""
import ollama

MODEL = "qwen2.5:7b"
PROMPT = "Describe a sunset over the ocean."

PRESETS = {
    "factual":  {"temperature": 0.1, "top_p": 0.9, "repeat_penalty": 1.1},
    "balanced": {"temperature": 0.8, "top_p": 0.9, "repeat_penalty": 1.1},
    "creative": {"temperature": 1.2, "top_p": 0.95, "repeat_penalty": 1.15},
}

for name, opts in PRESETS.items():
    print(f"\n{'='*60}\n{name.upper()}  {opts}\n{'='*60}")
    r = ollama.chat(model=MODEL, messages=[{"role": "user", "content": PROMPT}], options=opts)
    print(r["message"]["content"].strip())
