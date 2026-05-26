# Week 1, Day 1 — Install Ollama & Run Your First Local Model

> **Goal:** Install Ollama (the easiest way to run local LLMs), verify it sees the GPU, download your first local model (Llama 3.2 3B), and chat with it — fully offline, on your own hardware.
>
> **Time:** ~45-60 min (most spent on first model download).
>
> **Why this matters:** This is the day "local AI" stops being abstract. After today, you have a real LLM running on your laptop, no internet needed for inference, no cloud, no API key, no rate limits. The entire rest of the roadmap builds on this foundation.

---

## 📋 Today's Checklist

- [x] Install zstd (Ollama installer dependency)
- [ ] Install Ollama in WSL2 via the official install script
- [ ] Verify Ollama auto-detected the NVIDIA GPU + CUDA support
- [ ] Start Ollama (background service)
- [ ] Pull + run Llama 3.2 3B Instruct (~2 GB download, first time only)
- [ ] Chat with it interactively (`>>>` prompt)
- [ ] Pull + run Qwen 2.5 3B Instruct for comparison
- [ ] Subjective compare: which feels better at what tasks?
- [ ] Check VRAM usage with `nvidia-smi` while a model is loaded
- [ ] Note your measured tokens/sec on this hardware (RTX 1000 Ada, 6 GB VRAM)

---

## 🧠 Concepts I'm Learning Today

### What is Ollama, really?
- A **CLI + HTTP server** that wraps `llama.cpp` (the underlying C++ inference engine).
- Provides a **simple API** (REST, OpenAI-compatible) for running LLMs locally.
- Manages model downloads, versions, and metadata (similar to Docker for LLMs).
- **Auto-detects GPU** and chooses CUDA / Metal / Vulkan / ROCm backend.
- **Default port:** `localhost:11434`.

### Why Ollama vs raw llama.cpp?
- `llama.cpp` is the engine; Ollama is the user-friendly wrapper.
- Ollama = one-command model pull (`ollama run llama3.2:3b`).
- Ollama = persistent service, can be hit from any app via HTTP.
- llama.cpp = more control (every flag exposed, but you build/run manually).
- **For daily use: Ollama. For deep optimization: llama.cpp directly.**

### Model name format: `name:tag`
- `llama3.2:3b` = Llama 3.2 family, 3B parameter variant, default quantization
- `llama3.1:8b-instruct-q4_K_M` = 8B Instruct model, Q4_K_M quantization
- `qwen2.5:3b` = Qwen 2.5 family, 3B parameter variant
- Default tag (no suffix) is usually the "Instruct" variant at Q4_K_M quantization (the sweet-spot for size/quality).

### What Q4_K_M means (preview — we'll go deep in Week 2)
- "Q4" = ~4 bits per parameter (vs FP16's 16 bits → 4× smaller file)
- "K" = uses K-quants (smarter grouping than legacy Q4_0)
- "M" = medium variant (between Small and Large in the K-quant family)
- For Llama 3.2 3B: FP16 would be ~6 GB; Q4_K_M is ~2 GB. **Quality loss is tiny (<2%); file size is a third.**

### The 6 GB VRAM math (preview)
- Llama 3.2 3B Q4_K_M ≈ 2.0 GB on disk = ~2.0 GB VRAM for weights
- KV-cache (the conversation memory) adds ~0.3-1.0 GB at typical context
- Working memory for activations adds ~0.5 GB
- **Total: ~3 GB VRAM** → comfortably fits in 6 GB with room for system overhead
- Llama 3.1 8B Q4_K_M (the next step up) needs ~5-5.5 GB → just barely fits; tight context.

### Internet is needed ONLY for the initial download
- After `ollama pull` or `ollama run` finishes, the model weights live on your disk in `~/.ollama/models/`.
- All subsequent inference is **100% offline** — no telemetry, no API calls, no cloud.
- Internet needed again only for: pulling new models (`ollama pull X`), updating Ollama itself, pushing to Ollama Hub.
- **You can literally pull the WiFi out and Ollama keeps working.** That's the entire point of local AI.

### The 5 ways to interact with Ollama (memorize these)
| Method | When to use | Internet needed (after model download)? |
|---|---|---|
| `ollama run <model>` interactive `>>>` | Quick chat, exploration | ❌ No |
| `ollama run <model> "prompt"` one-shot | Scripts, one-off questions | ❌ No |
| REST API `POST localhost:11434/api/generate` | Custom apps, any language | ❌ No |
| Python `ollama` library | Python integration | ❌ No |
| OpenAI-compatible at `localhost:11434/v1` | Drop-in for tools expecting OpenAI | ❌ No |

All five hit the same systemd-managed Ollama service running on `localhost:11434`.

---

## 🛠 Step-by-Step (What I Actually Did)

### Phase 0: Hit a Dependency Error First

When I ran the Ollama install script:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

It failed with:
```
>>> Installing ollama to /usr/local
[sudo: authenticate] Password:
ERROR: This version requires zstd for extraction. Please install zstd and try again:
  - Debian/Ubuntu: sudo apt-get install zstd
  - RHEL/CentOS/Fedora: sudo dnf install zstd
  - Arch: sudo pacman -S zstd
```

**Why:** the newer Ollama installer ships compressed with zstd (a modern compression algorithm), and my fresh Ubuntu install didn't have it.

### Phase 1: Install the Missing Dependency

```bash
sudo apt-get install -y zstd
```

(~5 seconds — small package.)

### Phase 2: Install Ollama (retry)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Expected output (success path):
```
>>> Installing ollama to /usr/local
>>> Downloading Linux amd64 bundle
###################################################### 100%
>>> Adding ollama user to render group...
>>> Adding ollama user to video group...
>>> Adding current user to ollama group...
>>> Creating ollama systemd service...
>>> Enabling and starting ollama service...
>>> NVIDIA GPU installed.
```

**Key line to look for:** `>>> NVIDIA GPU installed.` — this confirms the installer detected your GPU and configured CUDA backend.

### Phase 3: Verify Ollama Is Running

```bash
ollama --version
```

Should print something like `ollama version is 0.x.y`.

The installer started Ollama as a **systemd service**, so it's already running in the background. You don't need to start it manually.

To verify the service is alive:
```bash
systemctl status ollama
```

Should show `active (running)`.

### Phase 4: Check Ollama Sees the GPU

In Ollama's logs (it logs to systemd journal), look for the inference-compute line:

```bash
journalctl -u ollama --no-pager | grep "inference compute"
```

You should see something like:
```
... msg="inference compute" id=GPU-... library=cuda variant=v12 compute=8.9 ... name="NVIDIA RTX 1000 Ada Generation Laptop GPU" total="6.0 GiB"
```

**The critical bits:**
- `library=cuda` ✅ (not `library=cpu`)
- `variant=v12` ✅ (CUDA 12 family)
- `compute=8.9` ✅ (Ada Lovelace compute capability)
- `total="6.0 GiB"` ✅ (your VRAM)

### Phase 5: Pull and Run Llama 3.2 3B ⭐

```bash
ollama run llama3.2:3b
```

First-time output:
```
pulling manifest
pulling 6a0746a1ec1a... 100% ▕████████████████▏ 2.0 GB
pulling 4fa551d4f938... 100% ▕████████████████▏  12 KB
pulling 8ab4849b038c... 100% ▕████████████████▏  254 B
verifying sha256 digest
writing manifest
success
>>>
```

The `>>>` prompt = **you're now chatting with an LLM running on your own laptop, fully offline.**

### Phase 6: First Real Chat

Things I asked:

```
>>> Hello! Briefly explain how you work and where you're running.

>>> Write a Python function that calculates the Nth Fibonacci number efficiently.

>>> What is the difference between WSL2 and a traditional VM?

>>> Explain quantum entanglement to a 10-year-old.
```

**Observations:**
- TTFT (time-to-first-token): ~0.5-1 sec (great)
- TPS (tokens-per-second): ~_____ tok/s (fill in after you measure)
- Quality: surprisingly good for a 3B model; coherent, accurate on common knowledge, decent at code.
- It hallucinates on niche facts — expected behavior for a small model.

Exit with `/bye` or `Ctrl+D`.

### Phase 7: Pull Qwen 2.5 3B for Comparison

```bash
ollama pull qwen2.5:3b
ollama run qwen2.5:3b
```

Asked the same questions. Subjective comparison:
- Llama 3.2 3B: ___ (your impression — more verbose? more concise? better code?)
- Qwen 2.5 3B: ___ (your impression)

### Phase 8: Check What's in Memory

In a separate terminal while a model is loaded:

```bash
ollama ps
```

Shows:
```
NAME           ID            SIZE      PROCESSOR    UNTIL
llama3.2:3b    a80c4f17acd5  4.0 GB    100% GPU     4 minutes from now
```

`100% GPU` ✅ confirms it's running on the GPU, not CPU.

Then:
```bash
nvidia-smi
```

You'll see:
- Memory-Usage: ~3-4 GB / 6 GB used
- A `ollama` process listed in the lower table
- GPU-Util: spikes 80-100% during generation, 0% when idle

### Phase 9: List All Installed Models

```bash
ollama list
```

Should show both models you pulled today, with sizes.

### Phase 10: Verify You Can Use Ollama OFFLINE (the lightbulb moment)

The whole point of local AI is that **once a model is downloaded, you never need internet again to use it**. Prove it to yourself:

```bash
# 1. Note what you have locally:
ollama list

# 2. Turn off WiFi (system tray) or Ethernet.

# 3. Confirm you really are offline:
ping -c 2 google.com    # should fail with "Network is unreachable" or timeout

# 4. Use Ollama anyway:
ollama run llama3.2:3b "What's the capital of France?"

# It answers normally. Magic. ✅

# 5. Reconnect WiFi when done.
```

Do this once and the "local AI" concept stops being abstract — you've watched an LLM generate an answer with zero packets going anywhere outside your laptop.

### Phase 11: Where the Models Actually Live

```bash
ls -lah ~/.ollama/models/
du -sh ~/.ollama/models/
```

Structure:
- `~/.ollama/models/blobs/` — model weight files (content-addressed by SHA256, like Docker layers)
- `~/.ollama/models/manifests/` — small JSON files mapping model:tag names to blob hashes

**You own these files.** Backup the folder, copy to another machine, or take on a plane. The model is just a ~2 GB file (for 3B Q4_K_M).

### Phase 12: The 5 Ways to Interact With Ollama

#### Way 1: Interactive Chat
```bash
ollama run llama3.2:3b
# >>> chat here
# /bye to exit
```

#### Way 2: One-Shot Query (great for scripts)
```bash
ollama run llama3.2:3b "Explain quantum entanglement in one paragraph"
# Prints answer and exits
```

#### Way 3: REST API (curl)
Ollama runs an HTTP server on `localhost:11434` 24/7 via systemd.

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2:3b",
  "prompt": "Why is the sky blue?",
  "stream": false
}'
```

For chat-style (with message history):
```bash
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2:3b",
  "messages": [{"role": "user", "content": "Hello"}],
  "stream": false
}'
```

#### Way 4: Python `ollama` Library
```bash
# In your venv:
uv pip install ollama
```

```python
import ollama
response = ollama.chat(
    model='llama3.2:3b',
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}]
)
print(response['message']['content'])
```

#### Way 5: OpenAI-Compatible Endpoint (the magic for migration)
Ollama also speaks OpenAI's API format. Any tool that expects OpenAI works against Ollama with just a URL swap.

```python
from openai import OpenAI
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'  # required but ignored
)
response = client.chat.completions.create(
    model='llama3.2:3b',
    messages=[{'role': 'user', 'content': 'Hello'}]
)
print(response.choices[0].message.content)
```

**This is huge:** existing code that uses OpenAI can be redirected to your local model by changing one line. LangChain, LlamaIndex, Continue.dev, hundreds of tools all "just work."

---

## 📊 My Measurements (Fill In)

| Metric | Llama 3.2 3B | Qwen 2.5 3B |
|---|---|---|
| Download size | 2.0 GB | _ GB |
| VRAM in use when loaded | ~ GB | ~ GB |
| First-token latency (TTFT) | _ sec | _ sec |
| Tokens-per-second (TPS) | _ tok/s | _ tok/s |
| Subjective quality on code | _ /10 | _ /10 |
| Subjective quality on general chat | _ /10 | _ /10 |

---

## ⚠ Surprises & Lessons Learned

1. **zstd dependency** — newer Ollama installer needs it; not all Ubuntu installs ship with zstd by default. One-line fix.
2. **Ollama auto-detected my GPU.** No manual CUDA path configuration needed. The CUDA toolkit install in Day 0a was enough.
3. **Systemd service runs Ollama automatically** in the background after install. No need to `ollama serve` manually unless I want to see logs in foreground.
4. **3B models are FAST on 6 GB VRAM.** Way more responsive than I expected — feels close to ChatGPT for short responses.
5. **First download is the slowest step.** Subsequent runs of the same model are instant (cached in `~/.ollama/models/`).
6. **Llama 3.2 vs Qwen 2.5 personalities are different.** Llama tends to add more "helpful assistant" framing; Qwen feels more direct and concise. Both are valid for different use cases.
7. **The "offline" property is real.** Disconnected WiFi, generated answers normally. Local AI delivers on its core promise from day one.
8. **OpenAI-compatible endpoint = drop-in migration.** Any tool/library expecting OpenAI's API can hit Ollama by just changing `base_url`. This is why the whole open-source ecosystem (LangChain, LlamaIndex, Continue.dev, etc.) "just works" with local models.

---

## 🐛 Troubleshooting Cheat Sheet

| Symptom | Fix |
|---|---|
| `ERROR: This version requires zstd` | `sudo apt-get install -y zstd`, then retry installer |
| `ollama: command not found` after install | New terminal needed (PATH refresh), or `source ~/.bashrc` |
| `ollama serve` says "address already in use" | Systemd already running it — that's expected. Just use `ollama run` directly. |
| Ollama uses CPU instead of GPU | Check `journalctl -u ollama \| grep "inference compute"` — if `library=cpu`, restart Ollama (`sudo systemctl restart ollama`) |
| Out of memory pulling a model | Either disk full or VRAM tight — try smaller variant (`:3b` instead of `:8b`) |
| Generation is slow | `ollama ps` — if processor shows `100% CPU` or split, model didn't fit on GPU. Use smaller quantization or smaller model. |

---

## ✅ Done When

- [ ] `ollama --version` works
- [ ] `journalctl -u ollama \| grep cuda` shows CUDA backend, your GPU, and ~6 GiB VRAM
- [ ] `ollama list` shows at least Llama 3.2 3B installed
- [ ] You had a real conversation with at least one local model
- [ ] You measured tokens/sec on your hardware
- [ ] `ollama ps` shows `100% GPU` while a model is loaded
- [ ] **You verified offline behavior** — disconnected WiFi, ran Ollama, got a response. ✅
- [ ] You've tried at least 2 of the 5 interaction methods (interactive chat + one of: curl/Python/OpenAI-compatible)

---

## 🔜 Next: `DAY_2.md` — VRAM Reality Check + Trying the 8B Tier

Tomorrow we push the hardware: install Llama 3.1 8B Q4_K_M (~4.7 GB), watch VRAM usage climb, understand why it juuust fits in 6 GB, and what happens when a model is too big (partial CPU offload, dramatic slowdown).

We'll also start using Ollama's REST API directly with `curl` and `python requests` — moving from interactive chat to programmable LLM workflows.
