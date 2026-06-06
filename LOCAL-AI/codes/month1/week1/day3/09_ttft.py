"""
Day 3 — Measure time-to-first-token (TTFT) vs total time.
Proves streaming's UX win: the answer STARTS in ~100-300 ms even if it finishes seconds later,
so it feels instant. Streaming doesn't change total time — only perceived latency.
Run (venv active):  python 09_ttft.py
"""
import ollama, time

start = time.time()
ttft = None
tokens = 0

for chunk in ollama.chat(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "Explain how a transformer works in ~150 words."}],
    stream=True,
):
    if ttft is None:                       # first token arrived
        ttft = time.time() - start
    print(chunk["message"]["content"], end="", flush=True)
    tokens += 1

total = time.time() - start
print(f"\n\nTTFT: {ttft*1000:.0f} ms   |   Total: {total:.1f} s   |   ~{tokens/total:.1f} tok/s")
