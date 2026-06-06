"""
Day 2 — Mini-project: run the SAME prompt through multiple local models and report
each one's answer + decode speed (tok/s). Unloads each model (keep_alive=0) before the
next so they don't fight over 6 GB of VRAM. Reuse this whenever evaluating a new model.
Run (venv active):  python 05_compare_models.py
"""
import ollama
import time

# All five models on disk — a full size + family + quant sweep:
#   3B tier:   llama3.2:3b (Meta), qwen2.5:3b (Alibaba)
#   7-8B tier: qwen2.5:7b, llama3.1:8b (Q4_K_M), llama3.1:8b-instruct-q4_K_S (Q4_K_S)
MODELS = [
    'llama3.2:3b',
    'qwen2.5:3b',
    'qwen2.5:7b',
    'llama3.1:8b',
    'llama3.1:8b-instruct-q4_K_S',
]
PROMPT = "In 2 sentences, what makes a good unit test?"

for model in MODELS:
    print(f"\n{'='*60}")
    print(f"MODEL: {model}")
    print('='*60)

    start = time.time()
    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': PROMPT}],
        options={'temperature': 0.3},  # consistent across models
    )
    elapsed = time.time() - start

    answer = response['message']['content']
    eval_count = response.get('eval_count', 0)
    eval_dur_ns = response.get('eval_duration', 1)
    tps = eval_count / (eval_dur_ns / 1e9) if eval_dur_ns else 0

    print(answer)
    print(f"\n-- {eval_count} tokens, {elapsed:.1f}s wall, {tps:.1f} tok/s decode")

    # Free VRAM before loading the next model (only one fits at a time on 6 GB):
    ollama.generate(model=model, prompt='', keep_alive=0)
