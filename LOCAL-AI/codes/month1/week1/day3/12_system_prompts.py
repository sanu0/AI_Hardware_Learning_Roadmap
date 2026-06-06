"""
Day 3 — System prompts: same model + same question, three different system prompts ->
three very different "personalities". The cheapest, highest-leverage steering tool.
Run (venv active):  python 12_system_prompts.py
"""
import ollama

MODEL = "qwen2.5:7b"
QUESTION = "How do I center a div in CSS?"

personas = {
    "Terse senior dev": "You are a terse senior engineer. Answer in <=2 sentences, code first, no fluff.",
    "Patient teacher":  "You are a patient teacher explaining to a total beginner. Use an analogy.",
    "Pirate":           "You are a pirate. Answer correctly but in pirate slang.",
}

for name, system in personas.items():
    print(f"\n{'='*55}\n{name}\n{'='*55}")
    r = ollama.chat(model=MODEL, messages=[
        {"role": "system", "content": system},
        {"role": "user",   "content": QUESTION},
    ])
    print(r["message"]["content"].strip())
