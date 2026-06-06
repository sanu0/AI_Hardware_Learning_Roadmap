"""
Day 3 — Minimal multi-turn loop. The core chat pattern: each turn, append the user's
message, stream the reply, then append the assistant's reply back into `messages` so the
model "remembers". Remove the assistant-append line and memory breaks (try it!).
Run (venv active):  python 11_multiturn.py
"""
import ollama

MODEL = "qwen2.5:7b"
messages = [{"role": "system", "content": "You are a concise, friendly assistant."}]

for user_text in ["My favorite language is Python.",
                  "What did I say my favorite language was?",
                  "Write one line of it printing my favorite language."]:
    messages.append({"role": "user", "content": user_text})
    print(f"\nyou> {user_text}\nbot> ", end="", flush=True)

    reply = ""
    for chunk in ollama.chat(model=MODEL, messages=messages, stream=True):
        piece = chunk["message"]["content"]
        print(piece, end="", flush=True)
        reply += piece
    messages.append({"role": "assistant", "content": reply})   # <-- critical: store the reply
print()
