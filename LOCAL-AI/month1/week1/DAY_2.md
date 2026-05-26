# Week 1, Day 2 — VRAM Reality Check + Driving Ollama from Python

> **Goal:** Push your 6 GB VRAM by trying the 8B tier (Llama 3.1 8B, Qwen 2.5 7B), observe what happens when models barely fit (and when they don't), measure tokens/sec rigorously across model sizes, then graduate from CLI chat to **driving Ollama programmatically from Python**.
>
> **Time:** ~2-3 hours.
>
> **Why this matters:** By the end of today you'll (a) understand the VRAM math intuitively so you can predict whether ANY new model will fit, and (b) be able to call Ollama from Python — the foundation for all the apps, agents, RAG systems, and pipelines we'll build for the next 11 months.

---

## 📋 Today's Checklist

- [ ] Pull Llama 3.1 8B Instruct Q4_K_M (~4.7 GB)
- [ ] Measure peak VRAM with `nvidia-smi` while loaded
- [ ] Benchmark TTFT (time-to-first-token) and TPS (tokens-per-second) on Llama 3.1 8B
- [ ] Pull Qwen 2.5 7B Instruct + Mistral 7B Instruct v0.3 — compare quality and speed
- [ ] Try a model that doesn't quite fit (e.g., Phi-3 Medium 14B Q4_K_M) — observe partial CPU offload
- [ ] Try same model at TWO different quantizations (Q4_K_M vs Q8_0) — compare VRAM + speed + quality
- [ ] Compute the VRAM math by hand for one model and validate against measured VRAM
- [ ] Install Python `ollama` library in your venv
- [ ] Write your first Python script that calls Ollama
- [ ] Write a streaming Python script (tokens print as generated)
- [ ] Hit the REST API directly with `curl` (no library)
- [ ] Test the OpenAI-compatible endpoint from Python
- [ ] Build a small **"compare 3 models on the same prompt"** Python utility
- [ ] Save all scripts to `~/local-ai/scripts/` so you can re-use them

---

## 🧠 Concepts I'm Learning Today

### The VRAM math equation (internalize this)

```
Total VRAM ≈ Parameters × Bytes-per-param + KV-cache + Activations + Overhead
```

Component breakdown:
- **Weights:** `params × bits/param ÷ 8` (e.g., 8B params × 4 bits ÷ 8 = 4 GB)
- **KV-cache:** scales with `context_length × num_layers × num_heads × head_dim × 2 × bytes`
  - For a 7-8B model at 4K context: ~0.5-1 GB
  - For 32K context: 3-5 GB (this is what kills you on small GPUs)
- **Activations:** ephemeral tensors during forward pass; ~0.2-0.5 GB typical
- **Overhead:** Ollama runtime, CUDA libraries, fragmentation; ~0.3-0.5 GB

Practical rules of thumb for 6 GB VRAM:
| Model size at Q4_K_M | Disk size | Approx VRAM @ 4K ctx | Fits? |
|---|---|---|---|
| 1-3B (Phi-3 Mini, Llama 3.2 3B) | 1.5-2.5 GB | 2-3 GB | ✅ comfortably |
| 7-8B (Llama 3.1 8B, Qwen 2.5 7B, Mistral 7B) | 4-5 GB | 5-6 GB | ✅ tight but fits |
| 9-14B (Phi-3 Medium, Yi 1.5 9B) | 5-8 GB | 7-10 GB | ⚠ partial CPU offload |
| 14B+ | >8 GB | >10 GB | ❌ mostly CPU, very slow |

### Partial CPU offload (why some models are dog-slow)

When a model doesn't fit entirely in VRAM, Ollama (via llama.cpp) splits it: some layers on GPU, rest on CPU.

- **Decode phase** is **memory-bandwidth-bound**, not compute-bound.
- VRAM bandwidth: ~200 GB/s on RTX 1000 Ada.
- DDR5 system RAM bandwidth: ~80 GB/s.
- So CPU layers process tokens ~2.5× slower than GPU layers — and you wait for the slowest part.
- A 10B model with 70% GPU / 30% CPU might run at **5-10 tok/s** instead of 30 tok/s.

Check the split with `ollama ps`:
```
NAME                ID            SIZE      PROCESSOR              UNTIL
phi3:14b-medium    abc123        7.5 GB    65%/35% CPU/GPU       4 minutes from now
```
That `CPU/GPU` indicator = partial offload.

### TTFT vs TPS vs throughput (what you're really measuring)

| Metric | What it measures | Bottleneck |
|---|---|---|
| **TTFT** (time-to-first-token) | Latency before any output appears | Prefill phase (compute-bound) |
| **TPOT / TPS** (time-per-token / tokens-per-sec) | Steady-state generation speed | Decode phase (memory-bandwidth-bound) |
| **Throughput** (tokens/sec across batch) | Total work done per unit time | Compute + memory at scale |

For interactive chat, **TTFT** dominates your perceived "snappiness" and **TPS** dominates your patience during long answers. For batch jobs, throughput matters.

### Prefill vs decode (the two phases of generation)

1. **Prefill:** the model reads your prompt and computes initial KV-cache. Time scales with prompt length. Compute-bound (matrix-matrix ops).
2. **Decode:** the model generates one new token at a time, autoregressively. Memory-bandwidth-bound (matrix-vector ops; weights have to be re-read each step).

Why decode is slow:
- For each new token, ~all of the model weights must be read from VRAM → SM caches.
- On a 7B Q4_K_M model (~4 GB weights), at ~200 GB/s VRAM bandwidth: floor of ~50 tok/s (200/4 = 50).
- Practical numbers are 60-80% of theoretical max → 30-40 tok/s is typical for 7B Q4 on a card like yours.

### Quantization trade-offs (Q4_K_M vs Q5_K_M vs Q8_0)

| Quant | Bits/param | File size for 7B | Quality loss vs FP16 | Speed |
|---|---|---|---|---|
| **Q4_K_M** | ~4.8 | ~4.4 GB | ~1-2% on benchmarks | Fastest in VRAM, less bandwidth |
| **Q5_K_M** | ~5.7 | ~5.1 GB | <1% | Slightly slower (more bandwidth) |
| **Q6_K** | ~6.6 | ~5.7 GB | ~0.5% | Slower |
| **Q8_0** | 8 | ~7.5 GB | ~negligible | Slowest (most bandwidth) — but on 6 GB VRAM, 7B Q8 won't fit fully |

**Rule of thumb:** Q4_K_M is the sweet spot for 6 GB VRAM. Only use Q5_K_M or higher if you have spare VRAM AND care about that last 1% quality.

### Ollama HTTP API surface (memorize these endpoints)

Ollama exposes 11+ endpoints. The ones you'll use:

| Endpoint | Purpose | Method |
|---|---|---|
| `/api/generate` | Single completion (text in, text out) | POST |
| `/api/chat` | Chat with message history | POST |
| `/api/embeddings` | Get vector embeddings (for RAG, semantic search) | POST |
| `/api/tags` | List installed models | GET |
| `/api/ps` | List running (loaded) models | GET |
| `/api/show` | Get details on a model (Modelfile, params) | POST |
| `/api/pull` | Pull a new model | POST |
| `/api/delete` | Delete a model | DELETE |
| `/api/create` | Create custom model from a Modelfile | POST |
| `/v1/chat/completions` | OpenAI-compatible chat | POST |
| `/v1/completions` | OpenAI-compatible completion | POST |

All on `http://localhost:11434`.

### Streaming vs non-streaming

- **Non-streaming** (`"stream": false`): server holds the request open, generates the entire response, sends it all at once. Simpler; better for batch jobs.
- **Streaming** (`"stream": true`, default): server-sent events (SSE) — each new token sent as a JSON line as it's generated. Better UX (you see output appearing); harder to parse.
- Ollama's default is streaming. Most code samples assume `"stream": false` for simplicity.

### Why use the Python `ollama` library vs raw `requests`?

- `ollama` library = thin wrapper, handles streaming parsing, type hints, retries. **Use it for app code.**
- `requests` = lower level, easier to debug, no extra dependency. **Use it when learning or scripting one-offs.**
- Both hit the exact same HTTP API on `localhost:11434`. Pick based on context.

---

## 🛠 Step-by-Step (What I'm Doing)

### Phase 1: Baseline — Free VRAM, Inspect Current State

Before pulling any new model, free VRAM and snapshot baseline:

```bash
# Unload anything currently loaded:
ollama ps   # see what's loaded
ollama stop <any-loaded-model>   # for each one

# Verify clean state:
ollama ps   # should be empty
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
# Expect: ~50-200 MB used (system desktop), ~5.8 GB free
```

### Phase 2: Pull Llama 3.1 8B Q4_K_M (the sweet spot)

```bash
ollama pull llama3.1:8b
# Downloads ~4.7 GB. First time only.
```

Why this model:
- **8B params**, **Q4_K_M** quant → ~4.7 GB on disk
- **~5.5 GB VRAM** at default 4K context — JUST fits on 6 GB ✅
- Genuinely capable: matches GPT-3.5 on many tasks
- Well-supported by every tool

### Phase 3: Load It & Measure (the moment of truth)

Open two terminals.

**Terminal 1** — start a watch on VRAM and processor split:
```bash
watch -n 1 'echo "=== ollama ps ==="; ollama ps; echo; echo "=== nvidia-smi ==="; nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv'
```

**Terminal 2** — fire a real query and measure:
```bash
time ollama run llama3.1:8b "Write a one-paragraph summary of quantum computing."
```

Observe in Terminal 1:
- VRAM jumps from ~200 MB → ~5.2-5.5 GB (the model + KV cache + working memory)
- `ollama ps` shows `100% GPU` ✅ — full GPU placement
- GPU utilization spikes 80-100% during generation

Record your numbers:
- Real time from `time`: ~__ sec
- Wall-clock impression of TPS: ~__ tok/s

### Phase 4: Rigorous TPS Measurement

Ollama's `--verbose` flag (in newer versions) or the API gives exact timing. Simplest approach via API:

```bash
curl -s http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Write a short poem about the ocean.",
  "stream": false
}' | python3 -c "
import sys, json
data = json.load(sys.stdin)
total_dur_ns = data['total_duration']
eval_count = data['eval_count']
eval_dur_ns = data['eval_duration']
print(f'Tokens generated: {eval_count}')
print(f'Eval duration: {eval_dur_ns/1e9:.2f}s')
print(f'Tokens/sec (decode): {eval_count / (eval_dur_ns/1e9):.1f}')
print(f'Total duration (incl prefill+load): {total_dur_ns/1e9:.2f}s')
"
```

Expected on RTX 1000 Ada for 7-8B Q4_K_M: **~25-40 tok/s**.

### Phase 5: Compare 7-8B Models Head-to-Head

```bash
# Pull two more:
ollama pull qwen2.5:7b
ollama pull mistral:7b

# Stop the previous to free VRAM (or let it auto-unload):
ollama stop llama3.1:8b

# Compare on the same prompt:
for model in llama3.1:8b qwen2.5:7b mistral:7b; do
  echo "=== $model ==="
  curl -s http://localhost:11434/api/generate -d "{\"model\":\"$model\",\"prompt\":\"Explain dark matter in 3 sentences.\",\"stream\":false}" \
    | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d['response'])
print(f'-- {d[\"eval_count\"]} tokens in {d[\"eval_duration\"]/1e9:.1f}s = {d[\"eval_count\"]/(d[\"eval_duration\"]/1e9):.1f} tok/s')
"
  ollama stop $model
done
```

Record results in the Measurements table below.

### Phase 6: Push Past the Limit — Observe Partial CPU Offload

Try a model that's TOO BIG to fully fit:

```bash
# Phi-3 Medium 14B Q4_K_M is ~7.5 GB on disk → won't fit fully in 6 GB VRAM:
ollama pull phi3:14b
ollama run phi3:14b "What is the meaning of life?"
```

While it's loading/running, check the split:
```bash
ollama ps
# You'll see something like:
# NAME            ID    SIZE     PROCESSOR              UNTIL
# phi3:14b        ...   8.0 GB   65%/35% CPU/GPU        4 min from now
```

**This is partial offload in action.** Notice:
- Generation is dramatically slower — maybe **5-8 tok/s** instead of 30-40
- CPU usage in `htop` is high (CPU layers working hard)
- GPU utilization is intermittent (waiting on CPU layers each step)

Clean up:
```bash
ollama stop phi3:14b
ollama rm phi3:14b   # optional — 8 GB on disk for a model you can't really use
```

**Lesson:** for daily 6 GB VRAM use, **stay at or below 8B at Q4_K_M**. Partial offload models work for occasional batch use but are too slow for interactive chat.

### Phase 7: Same Model, Different Quantizations

```bash
ollama pull qwen2.5:7b           # default = Q4_K_M, ~4.7 GB
ollama pull qwen2.5:7b-instruct-q8_0   # full Q8, ~8 GB — won't fully fit
```

Compare with the same prompt:
```bash
ollama run qwen2.5:7b "Translate to French: 'Local AI is transformative.'"
ollama run qwen2.5:7b-instruct-q8_0 "Translate to French: 'Local AI is transformative.'"
```

The Q8 version will be slower (partial offload), translation quality essentially identical to Q4_K_M.

**Lesson:** Q4_K_M is the right pick for 7-8B models on 6 GB VRAM. Higher quants don't justify their cost.

### Phase 8: Move to Python — Install the Ollama Library

Open VS Code with the WSL workspace (or a terminal). Make sure your venv is active:

```bash
cd ~/local-ai
source .venv/bin/activate    # prompt should show (.venv)
uv pip install ollama
```

Create a scripts directory:
```bash
mkdir -p ~/local-ai/scripts
cd ~/local-ai/scripts
```

### Phase 9: First Python Script — Basic Chat

Save as `~/local-ai/scripts/01_basic_chat.py`:

```python
"""Basic Ollama chat — the simplest possible call."""
import ollama

response = ollama.chat(
    model='llama3.1:8b',
    messages=[
        {'role': 'user', 'content': 'Why is the sky blue? Answer in 2 sentences.'}
    ]
)
print(response['message']['content'])
```

Run it:
```bash
python 01_basic_chat.py
```

Should print 2 sentences. **You just programmed an LLM.**

### Phase 10: Streaming Python Script — Tokens Appear Live

Save as `~/local-ai/scripts/02_streaming_chat.py`:

```python
"""Streaming Ollama chat — tokens appear as generated, like ChatGPT UI."""
import ollama

stream = ollama.chat(
    model='llama3.1:8b',
    messages=[
        {'role': 'user', 'content': 'Write a haiku about local AI.'}
    ],
    stream=True,
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
print()  # final newline
```

Run it:
```bash
python 02_streaming_chat.py
```

Notice how text appears word-by-word. This is the same experience as the `>>>` prompt, but in Python — meaning you can embed it in any app.

### Phase 11: REST API Directly with `curl` and `requests`

Save as `~/local-ai/scripts/03_raw_api.py`:

```python
"""Hit Ollama's REST API directly with requests — no ollama library."""
import requests
import json

response = requests.post(
    'http://localhost:11434/api/chat',
    json={
        'model': 'llama3.1:8b',
        'messages': [{'role': 'user', 'content': 'List 3 benefits of local LLMs.'}],
        'stream': False,
    },
)
data = response.json()
print(data['message']['content'])
print(f"\n-- {data['eval_count']} tokens, {data['eval_count'] / (data['eval_duration']/1e9):.1f} tok/s")
```

Run it:
```bash
python 03_raw_api.py
```

**Why bother with raw requests?** When debugging weird issues, you can see exactly what's on the wire. Also useful in languages without an `ollama` library (Go, Rust, Java, etc. — all can hit this endpoint).

### Phase 12: OpenAI-Compatible Endpoint

Save as `~/local-ai/scripts/04_openai_compat.py`:

```python
"""Use the OpenAI client library, but pointed at local Ollama."""
import os
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # required by the SDK but ignored by Ollama
)

response = client.chat.completions.create(
    model='llama3.1:8b',
    messages=[
        {'role': 'user', 'content': 'Suggest 3 names for a Python library that simplifies local LLMs.'}
    ],
)
print(response.choices[0].message.content)
```

Install the OpenAI lib first if you haven't:
```bash
uv pip install openai
```

Run:
```bash
python 04_openai_compat.py
```

**The magic:** any tool/library/example written for OpenAI works with your local model just by changing `base_url`. This includes LangChain, LlamaIndex, Continue.dev, instructor, hundreds more.

### Phase 13: Mini-Project — Compare 3 Models on the Same Prompt

Save as `~/local-ai/scripts/05_compare_models.py`:

```python
"""Run the same prompt through multiple local models, report each one's response + speed."""
import ollama
import time

MODELS = ['llama3.1:8b', 'qwen2.5:7b', 'mistral:7b']
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

    # Free VRAM before loading next (since only one fits at a time):
    ollama.generate(model=model, prompt='', keep_alive=0)
```

Run it:
```bash
python 05_compare_models.py
```

You now have a tool you'll re-use whenever evaluating a new model. **Build it, save it, version it.**

---

## 📊 My Measurements (Fill In)

| Model | Disk size | Peak VRAM | TPS (decode) | TTFT | Subjective quality |
|---|---|---|---|---|---|
| Llama 3.2 3B | ~2.0 GB | ~3 GB | __ tok/s | __ ms | __ /10 |
| Qwen 2.5 3B | ~2.0 GB | ~3 GB | __ tok/s | __ ms | __ /10 |
| Llama 3.1 8B | ~4.7 GB | __ GB | __ tok/s | __ ms | __ /10 |
| Qwen 2.5 7B | ~4.7 GB | __ GB | __ tok/s | __ ms | __ /10 |
| Mistral 7B | ~4.4 GB | __ GB | __ tok/s | __ ms | __ /10 |
| Phi-3 14B (partial offload) | ~8 GB | 6 GB + RAM | __ tok/s | __ ms | __ /10 |

### VRAM Math Validation

Pick one of the 7-8B models you measured. Compute by hand:

```
Weights:        params × bits/param ÷ 8 = ____ × ____ ÷ 8 = ____ GB
KV-cache:       2 × layers × heads × head_dim × ctx × bytes
                = 2 × 32 × 8 × 128 × 4096 × 2  bytes  (rough for 7-8B)
                = _____ GB
Activations:    ~0.3 GB
Overhead:       ~0.4 GB
---
Predicted:      _____ GB
Measured:       _____ GB
Difference:     _____ GB  (should be < 0.5 GB)
```

If close: you understand the math.
If way off: probably forgot KV-cache scales with context length — check `num_ctx` in Ollama.

---

## ⚠ Surprises & Lessons Learned

1. **Llama 3.1 8B Q4 fits — barely.** Around 5.2-5.5 GB VRAM. Don't run other GPU apps simultaneously or you'll OOM.
2. **TPS drops noticeably from 3B to 8B.** 3B → ~60 tok/s. 8B → ~30 tok/s. Roughly 2× the params = roughly half the speed. Memory bandwidth is the bottleneck.
3. **Partial CPU offload is a real performance cliff.** 14B model went from "would have been ~15 tok/s if it fit" to "actually 5-7 tok/s." The CPU portion is the bottleneck and you wait for it.
4. **Q8_0 doesn't make 7B models meaningfully smarter** for general chat. The 4 extra bits cost 60% more VRAM for ~1% quality lift. Skip unless you have spare VRAM and specific quality need.
5. **The Python `ollama` library is a one-line install** and turns local AI into a programmable resource. This is the moment LLMs go from "thing I chat with" to "thing I build with."
6. **OpenAI-compat works flawlessly.** Same Python code that hits OpenAI's API works against local Ollama by changing one URL. This is why the ecosystem coalesced around this protocol.
7. **`keep_alive: 0` in API responses unloads immediately** — useful for compare-models scripts that hit different models in sequence (otherwise the first stays in VRAM for 5 min and blocks the second).
8. **`eval_count` and `eval_duration` in API responses** give you exact decode-phase timing. Use these for benchmarks instead of wall-clock.

---

## 🐛 Troubleshooting Cheat Sheet

| Symptom | Likely cause | Fix |
|---|---|---|
| "CUDA out of memory" on 8B model | Other GPU app running, or KV cache too big | Close ComfyUI/games; set `num_ctx` lower (e.g., 2048) |
| 8B model loads but very slow (~5 tok/s) | Partial CPU offload | `ollama ps` → check PROCESSOR column. Switch to smaller model or smaller quant. |
| `import ollama` fails | Not in venv, or package not installed | `source ~/local-ai/.venv/bin/activate && uv pip install ollama` |
| `Connection refused` from Python | Ollama service stopped | `sudo systemctl status ollama`; restart if not running |
| Streaming chunks come as bytes not strings | API returning raw bytes | Use `ollama` library (handles this); or `chunk.decode('utf-8')` with raw requests |
| `eval_duration` is None or 0 | Asked for a model that didn't load successfully | Check `ollama ps` and Ollama logs (`journalctl -u ollama`) |
| Compare-models script: 2nd model takes 5 min to start | First model still warm in VRAM | Use `keep_alive: 0` in the previous call, OR `ollama stop <model>` between |

---

## ✅ Done When

- [ ] You measured TTFT and TPS for at least 3 different models (3B, 7B, 8B tier)
- [ ] You watched VRAM in real time as a model loaded (saw the jump on `nvidia-smi`)
- [ ] You observed partial CPU offload at least once (tried a model too big for 6 GB)
- [ ] You wrote and ran a Python script (`01_basic_chat.py`) that calls Ollama
- [ ] You wrote a streaming script (`02_streaming_chat.py`) — tokens appear live
- [ ] You hit the REST API with raw `requests` once
- [ ] You used the OpenAI-compatible endpoint from Python
- [ ] You built and used a compare-models utility (`05_compare_models.py`)
- [ ] Your VRAM math estimate is within ~0.5 GB of measured VRAM for one model
- [ ] All scripts saved in `~/local-ai/scripts/` (version-controllable)

---

## 🔜 Next: `DAY_3.md` — Streaming Generation, Chat Patterns, & Conversation History

Tomorrow we go deeper into the chat API:
- **Multi-turn conversation** — pass the full message history; observe context handling
- **Streaming UX** — print tokens in a loop, handle `Ctrl+C` cleanly, add a typing indicator
- **Server-Sent Events (SSE)** — parse the streaming protocol with raw `requests`
- **System prompts** — change the "personality" of the same model dramatically
- **Temperature, top_p, top_k** — what each does, when to tune
- **Build a simple terminal chatbot** with conversation memory — your first real Python+Ollama app

Then Day 4: model parameters in depth (`num_ctx`, `temperature`, `repeat_penalty`, all the knobs). Day 5: CLI mastery (Modelfiles, `ollama create`, `ollama push`).

By end of Week 1: you're not just running LLMs — you're **building with them.**
