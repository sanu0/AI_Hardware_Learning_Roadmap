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
- [ ] Compare 7-8B models head-to-head: Llama 3.1 8B vs Qwen 2.5 7B (both already downloaded)
- [ ] Force heavier partial CPU offload by cranking `num_ctx` on Llama 3.1 8B — observe the speed cliff
- [ ] Compare same model at TWO quants you already have: Llama 3.1 8B Q4_K_M vs Q4_K_S — VRAM + speed + quality
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
| 7B (Mistral 7B, Qwen 2.5 7B) | 4-4.5 GB | 5-5.5 GB | ✅ usually fully on GPU |
| 8B (Llama 3.1 8B) | 4.6-4.9 GB | 5.8-6 GB | ⚠ borderline — often 5-10% CPU offload at 4K context |
| 9-14B (Phi-3 Medium, Yi 1.5 9B) | 5-8 GB | 7-10 GB | ⚠ partial CPU offload (30-50% CPU) |
| 14B+ | >8 GB | >10 GB | ❌ mostly CPU, very slow |

**Real-world gotcha for 8B at 4K context:** Llama 3.1 8B Q4_K_M usually triggers ~5-10% CPU offload on 6 GB cards because weights + KV cache + CUDA overhead totals ~5.9 GB. Fix: lower `num_ctx` to 2048 (frees ~250 MB of KV cache) → typically gets you 100% GPU placement.

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

### Ollama is two things — the server + the CLI wrapper
- **Server (HTTP service):** runs on `localhost:11434` 24/7 via systemd. This is where models load, inference happens, etc.
- **CLI (`ollama` command):** convenience wrapper that makes HTTP calls to that server. Every CLI command maps to an HTTP endpoint.

So when you type `ollama list`, internally it does `GET http://localhost:11434/api/tags` and pretty-prints the JSON. The CLI is a UI; the HTTP API is the real surface.

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

### CLI command → HTTP endpoint cheat sheet

Every `ollama <command>` you've used is equivalent to a `curl` HTTP call:

| CLI command | Equivalent HTTP call |
|---|---|
| `ollama list` | `curl http://localhost:11434/api/tags` |
| `ollama ps` | `curl http://localhost:11434/api/ps` |
| `ollama show <model>` | `curl http://localhost:11434/api/show -d '{"name":"<model>"}'` |
| `ollama pull <model>` | `curl http://localhost:11434/api/pull -d '{"name":"<model>"}'` |
| `ollama rm <model>` | `curl -X DELETE http://localhost:11434/api/delete -d '{"name":"<model>"}'` |
| `ollama run <model> "prompt"` | `curl http://localhost:11434/api/generate -d '{"model":"<model>","prompt":"...","stream":false}'` |
| `ollama stop <model>` | `curl http://localhost:11434/api/generate -d '{"model":"<model>","keep_alive":0}'` |

Once this clicks, you realize: **any language with an HTTP client can drive Ollama.** Python, JavaScript, Go, Rust, even raw Bash with `curl`.

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

### Phase 3b: Real-World VRAM Reality (what I actually observed) ⭐

**Surprise result:** even Llama 3.1 8B Q4_K_M at default 4K context **does NOT fully fit in 6 GB VRAM.**

My actual `ollama ps` after loading:
```
NAME           SIZE      PROCESSOR         CONTEXT
llama3.1:8b    5.9 GB    7%/93% CPU/GPU    4096
```

`nvidia-smi`: 5293 MiB used / 632 MiB free, GPU util ~70%.

**Diagnosis:**
- Total memory needed: ~5.9 GB (weights ~4.6 + KV cache ~0.5 + activations + overhead)
- Available VRAM after system overhead: ~5.3 GB
- ~600 MB short → 7% of layers spilled to CPU
- GPU sits at ~70% util waiting on CPU layers each token
- TPS: 21.1 (vs expected 32+ for fully-on-GPU 8B)

**The lesson:** "X GB on disk → fits in (X + 1) GB VRAM" is too optimistic. For 6 GB cards: **Llama 3.1 8B Q4_K_M is borderline** at default context.

#### Two ways to force full-GPU placement on 6 GB

**Fix A — Lower the context window (smaller KV cache):**
```bash
# Use num_ctx: 2048 instead of default 4096:
curl -s http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Hello",
  "stream": false,
  "options": {"num_ctx": 2048}
}' | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'TPS: {d[\"eval_count\"]/(d[\"eval_duration\"]/1e9):.1f}')"

ollama ps
# Now shows: PROCESSOR = 100% GPU ✅
```
Trade-off: 2K context vs 4K. For most chat, fine; for long-doc Q&A, painful.

**Fix B — Use a slightly smaller model:**
- Mistral 7B Q4_K_M (~4.4 GB on disk) — slightly smaller; usually fits 100% GPU at 4K
- Llama 3.2 3B (~2 GB) — comfortable headroom for 8K+ context fully on GPU
- Qwen 2.5 7B Q4_K_M — similar size to Llama 3.1 8B; same partial-offload risk

#### Quick A/B comparison (run this to feel the difference)

```bash
PROMPT="Write a 3-sentence summary of why the sky is blue."

# 1. Llama 3.1 8B @ default 4K (partial offload):
curl -s http://localhost:11434/api/generate -d "{\"model\":\"llama3.1:8b\",\"prompt\":\"$PROMPT\",\"stream\":false}" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'8B@4K: {d[\"eval_count\"]/(d[\"eval_duration\"]/1e9):.1f} TPS')"
ollama stop llama3.1:8b

# 2. Llama 3.1 8B @ 2K (forced full GPU):
curl -s http://localhost:11434/api/generate -d "{\"model\":\"llama3.1:8b\",\"prompt\":\"$PROMPT\",\"stream\":false,\"options\":{\"num_ctx\":2048}}" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'8B@2K: {d[\"eval_count\"]/(d[\"eval_duration\"]/1e9):.1f} TPS')"
ollama stop llama3.1:8b

# 3. Llama 3.2 3B (way under budget):
curl -s http://localhost:11434/api/generate -d "{\"model\":\"llama3.2:3b\",\"prompt\":\"$PROMPT\",\"stream\":false}" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'3B@4K: {d[\"eval_count\"]/(d[\"eval_duration\"]/1e9):.1f} TPS')"
```

Expected output (rough on RTX 1000 Ada):
- 8B@4K: ~21 TPS, processor `7%/93% CPU/GPU`
- 8B@2K: ~30-35 TPS, processor `100% GPU` ← the fix
- 3B@4K: ~55-65 TPS, processor `100% GPU`

You can now intuitively predict: "this model + this context = X% GPU fit = Y tok/s."

### Phase 3c: Why TPS varies wildly between runs (and how to benchmark properly) ⭐⭐

When I ran the same `curl` 8 times in a row, I saw TPS values: **12.9, 11.1, 18.1, 15.2, 25.1, 23.5, 24.5, 22.2**. Range = 11-25 TPS. Why?

#### The four sources of TPS variance

**1. GPU clock ramp-up (cold-start, runs 1-3 are slow)**
GPUs idle at ~200 MHz to save power. On first inference they boost to ~1300-1700 MHz. The first 1-2 runs see partial boost clocks. *Always discard warmup runs.*

**2. Output length variance (different runs generate different numbers of tokens)**
"Hello" can produce 5 tokens or 50 tokens depending on randomness. Short outputs have proportionally higher per-token overhead → bias TPS downward. *Fix:* set `num_predict: 100` to force a fixed output length.

**3. Prompt length affects decode (subtle but real)**
Each new decoded token requires reading the K/V cache for ALL previous tokens. Longer prompts → larger KV cache → slower per-token decode. My data:
- `"Hello"` (1 token prompt) → ~22-25 TPS
- `"Hello explain me quantum physics"` (5-6 token prompt + likely longer response) → ~17 TPS

This is fundamental: **decode TPS is NOT a single number for a model — it's a function of prompt length and output length.**

**4. Partial CPU offload introduces non-determinism**
When `ollama ps` shows `7%/93% CPU/GPU`, the CPU is the bottleneck. CPU performance depends on what else is happening on your laptop (browser, Slack, antivirus). *This is why my Llama 3.1 8B at 4K context jumped from 11 to 25 TPS run-to-run* — fully-on-GPU models are more consistent.

#### Proper TPS benchmark script (saved to `~/local-ai/scripts/benchmark.sh`)

**Create it in one shot** (the quoted `<<'EOF'` writes the script literally, without expanding `$MODEL`/`$(...)`):
```bash
mkdir -p ~/local-ai/scripts
cat > ~/local-ai/scripts/benchmark.sh <<'EOF'
#!/bin/bash
# Proper TPS benchmark: fixed prompt, fixed output length, warmup + N runs, report median
MODEL="${1:-llama3.1:8b}"
NUM_CTX="${2:-2048}"
NUM_PREDICT=100   # always generate exactly 100 tokens
RUNS=5

echo "Benchmarking $MODEL (num_ctx=$NUM_CTX, num_predict=$NUM_PREDICT)"

# Warmup (discarded)
curl -s http://localhost:11434/api/generate -d "{
  \"model\": \"$MODEL\",
  \"prompt\": \"Count from 1 to 100.\",
  \"stream\": false,
  \"options\": {\"num_ctx\": $NUM_CTX, \"num_predict\": $NUM_PREDICT, \"seed\": 42}
}" > /dev/null
echo "Warmup done."

# Measured runs
TPS_VALUES=()
for i in $(seq 1 $RUNS); do
  TPS=$(curl -s http://localhost:11434/api/generate -d "{
    \"model\": \"$MODEL\",
    \"prompt\": \"Count from 1 to 100.\",
    \"stream\": false,
    \"options\": {\"num_ctx\": $NUM_CTX, \"num_predict\": $NUM_PREDICT, \"seed\": 42}
  }" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d[\"eval_count\"]/(d[\"eval_duration\"]/1e9):.1f}')")
  echo "Run $i: $TPS TPS"
  TPS_VALUES+=("$TPS")
done

echo "Sorted: $(echo "${TPS_VALUES[@]}" | tr ' ' '\n' | sort -n | tr '\n' ' ')"
MEDIAN=$(echo "${TPS_VALUES[@]}" | tr ' ' '\n' | sort -n | awk 'BEGIN{c=0}{a[c++]=$1}END{print a[int(c/2)]}')
echo "Median TPS: $MEDIAN"
ollama ps
EOF
chmod +x ~/local-ai/scripts/benchmark.sh
```

> ⚠ Gotcha I hit: the script is documented here, but you must actually create it on disk before calling it — otherwise you get `No such file or directory`. The block above creates it for you.

For reference, the script contents are also shown below (same thing, un-wrapped):

```bash
#!/bin/bash
# Proper TPS benchmark: fixed prompt, fixed output length, warmup + N runs, report median
MODEL="${1:-llama3.1:8b}"
NUM_CTX="${2:-2048}"
NUM_PREDICT=100   # always generate exactly 100 tokens
RUNS=5

echo "Benchmarking $MODEL (num_ctx=$NUM_CTX, num_predict=$NUM_PREDICT)"

# Warmup (discarded)
curl -s http://localhost:11434/api/generate -d "{
  \"model\": \"$MODEL\",
  \"prompt\": \"Count from 1 to 100.\",
  \"stream\": false,
  \"options\": {\"num_ctx\": $NUM_CTX, \"num_predict\": $NUM_PREDICT, \"seed\": 42}
}" > /dev/null
echo "Warmup done."

# Measured runs
TPS_VALUES=()
for i in $(seq 1 $RUNS); do
  TPS=$(curl -s http://localhost:11434/api/generate -d "{
    \"model\": \"$MODEL\",
    \"prompt\": \"Count from 1 to 100.\",
    \"stream\": false,
    \"options\": {\"num_ctx\": $NUM_CTX, \"num_predict\": $NUM_PREDICT, \"seed\": 42}
  }" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d[\"eval_count\"]/(d[\"eval_duration\"]/1e9):.1f}')")
  echo "Run $i: $TPS TPS"
  TPS_VALUES+=("$TPS")
done

echo "Sorted: $(echo "${TPS_VALUES[@]}" | tr ' ' '\n' | sort -n | tr '\n' ' ')"
MEDIAN=$(echo "${TPS_VALUES[@]}" | tr ' ' '\n' | sort -n | awk 'BEGIN{c=0}{a[c++]=$1}END{print a[int(c/2)]}')
echo "Median TPS: $MEDIAN"
ollama ps
```

Make executable: `chmod +x ~/local-ai/scripts/benchmark.sh`

Run on different models:
```bash
~/local-ai/scripts/benchmark.sh llama3.1:8b 4096    # default ctx (partial offload)
ollama stop llama3.1:8b
~/local-ai/scripts/benchmark.sh llama3.1:8b 2048    # halved ctx
ollama stop llama3.1:8b
~/local-ai/scripts/benchmark.sh mistral:7b 4096      # likely fits 100% GPU
ollama stop mistral:7b
~/local-ai/scripts/benchmark.sh llama3.2:3b 4096     # comfortable headroom
```

Now you'll get reproducible numbers, not single-run noise.

#### `num_ctx: 2048` didn't fully fix my partial offload — and why ⭐⭐⭐

My A/B comparison results (averaged over 2 runs each, same prompt "Write a 3-sentence summary of why the sky is blue."):
| Config | TPS (avg) | Processor split | Verdict |
|---|---|---|---|
| Llama 3.1 8B @ 4K | 21.2 | 7%/93% CPU/GPU | partial offload |
| Llama 3.1 8B @ 2K | 20.9 | 8%/92% CPU/GPU | **STILL partial offload, identical TPS!** |
| Llama 3.2 3B @ 4K | 56.4 | 100% GPU | 2.7× faster |

**Key insight: `num_ctx` was the wrong lever for my hardware.** Halving context saved 250 MB of KV cache, but I was 600 MB short of fitting → net effect = essentially zero.

#### The math I should have done up front

```
Total VRAM for Llama 3.1 8B Q4_K_M @ 4K context:
  Weights:        4.7 GB   ← DOMINANT, fixed by quantization
  KV cache @ 4K:  0.5 GB   ← what num_ctx affects
  Activations:    0.3 GB
  CUDA overhead:  0.4 GB
  ─────────────
  Total:          5.9 GB
  Available:      ~5.3 GB
  Shortfall:      0.6 GB

Halving context (4K → 2K) saves only 0.25 GB.
Still 0.35 GB short → still partial offload → identical TPS.
```

**The weights — not the KV cache — are the binding constraint for 8B-class models on 6 GB.**

#### Memory-bandwidth-bound theory matches my data perfectly

RTX 1000 Ada Laptop has ~192 GB/s memory bandwidth. Theoretical decode TPS for memory-bound inference:
```
TPS_theoretical = memory_bandwidth / model_size_in_VRAM
```

| Model | Size in VRAM | Theoretical max | Realistic (60% efficiency) | My actual |
|---|---|---|---|---|
| Llama 3.2 3B Q4 | 2.0 GB | 96 TPS | ~58 TPS | **56 TPS ✓** |
| Llama 3.1 8B Q4 (full GPU) | 4.7 GB | 41 TPS | ~25 TPS | (target) |
| Llama 3.1 8B Q4 (7% CPU offload) | 4.7 GB | ~25 × 0.85 | ~21 TPS | **21 TPS ✓** |

Both numbers match the bandwidth model within 5%. **Decode is memory-bandwidth-bound, not compute-bound** — this is the most important performance principle in local inference.

#### What actually fixes partial offload on 6 GB

You must attack the **weights**, not the KV cache. Three options that *will* fit:

**Option A — Smaller 4-bit quant (`Q4_K_S`)**
```bash
ollama pull llama3.1:8b-instruct-q4_K_S
ollama run llama3.1:8b-instruct-q4_K_S "test"
ollama ps    # expect 100% GPU
```
~4.4 GB total memory; fits 100% GPU; **expect ~28-32 TPS**; quality drop vs Q4_K_M is negligible (~1% perplexity).

**Option B — 3-bit quant (`Q3_K_M`)**
```bash
ollama pull llama3.1:8b-instruct-q3_K_M
```
~3.9 GB total; comfortable fit; **expect ~38-42 TPS**; quality drop is more noticeable (~3-5% perplexity).

**Option C — Different smaller model**
```bash
ollama pull mistral:7b      # 4.4 GB weights → fits 100% GPU at 4K
```
Or `qwen2.5:7b`, `qwen2.5-coder:7b` — all ~4.4 GB and usually fit fully.

#### The Iron Law of 6 GB inference

> **On 6 GB VRAM, you cannot have all three of: 8B-class quality, 4K+ context, and full-GPU placement. Pick two.**

This is the fundamental constraint that drove the entire small-LLM ecosystem (Phi-3 Mini, Llama 3.2 3B, Gemma 2 2B) into existence. Every consumer GPU has its own version of this tradeoff (8 GB cards hit the same wall around 13B; 12 GB cards around 22B; 24 GB cards around 70B Q4).

**My pragmatic daily workflow on 6 GB:**
| Use case | Model | Speed | Quality |
|---|---|---|---|
| Fast iteration, agents, RAG dev | **Llama 3.2 3B** | ~56 TPS | 7/10 |
| Best general chat at full speed | **Llama 3.1 8B Q4_K_S** | ~30 TPS | 9/10 |
| Best quality, accept partial offload | **Llama 3.1 8B Q4_K_M** | ~21 TPS | 9.5/10 |
| Coding tasks | **Qwen 2.5 Coder 7B** | ~30 TPS | 9/10 (code-specific) |

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

Expected on RTX 1000 Ada for 7-8B Q4_K_M: **~18-25 tok/s** (lower end if partially offloaded like Llama 3.1 8B; higher if fully on GPU like Qwen 2.5 7B).

> **Note on `total_duration` vs `eval_duration`:** the first run after loading a model is *cold* — `total_duration` includes ~8-10s of loading 4.9 GB from disk into VRAM. Measured example: decode 18.2 tok/s but `total_duration` 14.08s (load+prefill ≈ 9.85s). Run the same request immediately again and `total_duration` collapses to ~5s (model warm in VRAM via keep-alive). Always benchmark with a warmup run.

### Phase 5: Compare 7-8B Models Head-to-Head (no new downloads)

You already have two strong 7-8B models from different families — `llama3.1:8b` (Meta) and `qwen2.5:7b` (Alibaba). Same tier, different training → great quality/speed comparison.

```bash
# Confirm what you have (no pulls needed):
ollama list

# Compare both on the same prompt, unloading each to free VRAM (only one fits at a time):
for model in llama3.1:8b qwen2.5:7b; do
  echo "=== $model ==="
  curl -s http://localhost:11434/api/generate -d "{\"model\":\"$model\",\"prompt\":\"Explain dark matter in 3 sentences.\",\"stream\":false}" \
    | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d['response'])
print(f'-- {d[\"eval_count\"]} tokens in {d[\"eval_duration\"]/1e9:.1f}s = {d[\"eval_count\"]/(d[\"eval_duration\"]/1e9):.1f} tok/s')
"
  ollama ps          # note the processor split for each
  ollama stop $model
done
```

What to look for:
- **`ollama ps` SIZE + processor split** — both are ~5 GB models; do they both partially offload like Llama 3.1 8B did, or does Qwen 2.5 7B fit better?
- **Speed** — TPS difference between the two
- **Quality** — which explanation is clearer/more accurate? (Subjective, but you're the judge.)

For a rigorous version, use your benchmark script on each:
```bash
~/local-ai/scripts/benchmark.sh llama3.1:8b 4096
ollama stop llama3.1:8b
~/local-ai/scripts/benchmark.sh qwen2.5:7b 4096
ollama stop qwen2.5:7b
```

Record results in the Measurements table below.

#### What I actually measured (Phase 5) ⭐ — "same tier ≠ same fit"

Single-run, same prompt ("Explain dark matter in 3 sentences."):
| Model | `ollama ps` SIZE | Processor split | Tokens | TPS (single run) |
|---|---|---|---|---|
| `llama3.1:8b` (Q4_K_M) | 5.9 GB | **7%/93% CPU/GPU** ⚠ | 139 | 20.7 |
| `qwen2.5:7b` (Q4_K_M) | **4.9 GB** | **100% GPU** ✅ | 79 | 15.1 |

**The big discovery:** Qwen 2.5 7B fits **100% on GPU** while Llama 3.1 8B partially offloads — even though both are "7-8B Q4_K_M" and nearly identical on disk (4.7 vs 4.9 GB). In VRAM they're ~1 GB apart. Why:
| | Llama 3.1 8B | Qwen 2.5 7B |
|---|---|---|
| Params | 8.0B | 7.6B (smaller weights) |
| Layers | 32 | 28 |
| KV heads (GQA) | 8 | 4 → **~half the KV cache** |
| KV cache @ 4K | ~512 MB | ~230 MB |
| VRAM footprint | 5.9 GB | 4.9 GB |

→ On 6 GB, **Qwen 2.5 7B is a "fits-fully" daily driver; Llama 3.1 8B is "borderline-offload."**

**⚠ Don't trust the single-run TPS column above** — it's a single cold run with *different output lengths* (139 vs 79 tokens). Per Phase 3c, that's not a fair comparison: the offloaded Llama looking "faster" is an artifact of its longer run spending more time at boosted GPU clocks.

**✅ Resolved with the warm benchmark (`benchmark.sh`, fixed num_predict + warmup + median of 5):**
| Model | Median TPS (warm) | Spread | Placement |
|---|---|---|---|
| `llama3.1:8b` | **25.4** | 25.3-25.6 (tight) | 7%/93% CPU/GPU |
| `qwen2.5:7b` | **32.5** | 31.8-33.0 (tight) | 100% GPU ✅ |

Once measured fairly, **Qwen 2.5 7B is ~28% faster** — exactly as bandwidth theory predicts (it's smaller in VRAM and fully resident). The single-run inversion was pure noise. Bandwidth check: Qwen 192÷4.9 = 39 max → 32.5 measured = 83% efficiency; Llama 192÷5.9 ≈ 32.5 max × 83% ≈ 27 if it fit fully, minus the small 7% offload tax = 25.4 measured. ✔

**Most striking: warmup more than DOUBLED Qwen's number** (cold single run 15.1 → warm median 32.5, a 2.15× difference). This is the definitive proof that cold single-run TPS is meaningless. **Verdict: Qwen 2.5 7B is the best 7B-class daily driver on 6 GB — fits fully AND faster.**

### Phase 6: Force Heavier Partial CPU Offload (no 14B download needed)

You don't need a giant model to study partial offload — you can *induce* it on `llama3.1:8b` by inflating the KV cache with a bigger `num_ctx`. Bigger context = bigger KV cache = more layers pushed to CPU = bigger speed cliff. This is a cleaner experiment than downloading a 14B because it isolates a single variable (KV cache) while weights stay fixed.

```bash
# Escalate context and watch the offload grow. Use a fixed output length for fair TPS.
for ctx in 4096 8192 16384; do
  echo "=== llama3.1:8b @ num_ctx=$ctx ==="
  curl -s http://localhost:11434/api/generate -d "{
    \"model\": \"llama3.1:8b\",
    \"prompt\": \"Count slowly from 1 to 30.\",
    \"stream\": false,
    \"options\": {\"num_ctx\": $ctx, \"num_predict\": 80, \"seed\": 42}
  }" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'TPS: {d[\"eval_count\"]/(d[\"eval_duration\"]/1e9):.1f}')"
  ollama ps          # watch SIZE grow and the CPU% climb
  ollama stop llama3.1:8b
done
```

⚠ The quick loop above has **no warmup**, so its first context (4K, run cold) reads noisy/low. For clean numbers, use the dedicated sweep script below.

#### Rigorous version: `benchmark_ctx.sh` (warmup + median per context) ⭐

Create it once:
```bash
cat > ~/local-ai/scripts/benchmark_ctx.sh <<'EOF'
#!/bin/bash
# Sweep num_ctx for one model: warmup + median TPS per context, plus the processor split.
# Usage: ./benchmark_ctx.sh [model] ["ctx1 ctx2 ..."]
MODEL="${1:-llama3.1:8b}"
CTXS="${2:-4096 8192 16384}"
NUM_PREDICT=100
RUNS=5
PROMPT="Count slowly from 1 to 100."

echo "Context sweep for $MODEL (num_predict=$NUM_PREDICT, $RUNS runs each, with warmup)"
echo "------------------------------------------------------------"

for CTX in $CTXS; do
  # Warmup at this context (forces reload + GPU clock ramp; discarded)
  curl -s http://localhost:11434/api/generate -d "{
    \"model\": \"$MODEL\",
    \"prompt\": \"$PROMPT\",
    \"stream\": false,
    \"options\": {\"num_ctx\": $CTX, \"num_predict\": $NUM_PREDICT, \"seed\": 42}
  }" > /dev/null

  TPS_VALUES=()
  for i in $(seq 1 $RUNS); do
    TPS=$(curl -s http://localhost:11434/api/generate -d "{
      \"model\": \"$MODEL\",
      \"prompt\": \"$PROMPT\",
      \"stream\": false,
      \"options\": {\"num_ctx\": $CTX, \"num_predict\": $NUM_PREDICT, \"seed\": 42}
    }" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d[\"eval_count\"]/(d[\"eval_duration\"]/1e9):.1f}')")
    TPS_VALUES+=("$TPS")
  done

  MEDIAN=$(echo "${TPS_VALUES[@]}" | tr ' ' '\n' | sort -n | awk 'BEGIN{c=0}{a[c++]=$1}END{print a[int(c/2)]}')
  INFO=$(ollama ps | awk -v m="$MODEL" '$1==m {print $3" "$4" | "$5" "$6}')
  echo "ctx=$CTX | $INFO | median ${MEDIAN} TPS"

  ollama stop "$MODEL" > /dev/null
done
EOF
chmod +x ~/local-ai/scripts/benchmark_ctx.sh
```

Run:
```bash
~/local-ai/scripts/benchmark_ctx.sh llama3.1:8b
# custom contexts:
~/local-ai/scripts/benchmark_ctx.sh llama3.1:8b "2048 4096 8192 16384"
```
Output (one clean line per context):
```
ctx=4096 | 5.9 GB | 7%/93% CPU/GPU | median 25.4 TPS
ctx=8192 | 6.4 GB | 15%/85% CPU/GPU | median XX.X TPS
ctx=16384 | 8.0 GB | 33%/67% CPU/GPU | median XX.X TPS
```

**Measured on RTX 1000 Ada (6 GB) with `benchmark_ctx.sh` (warmup + median of 5)** ⭐ real data:
| num_ctx | SIZE | Processor split | TPS (warm median) | vs 4K | Notes |
|---|---|---|---|---|---|
| 4096 | 5.9 GB | **7%/93% CPU/GPU** | **24.1** | baseline | |
| 8192 | 6.4 GB | **15%/85% CPU/GPU** | **20.6** | −15% | only ~15% slower despite 2× the offload |
| 16384 | 8.0 GB | **33%/67% CPU/GPU** | **10.3** | −57% | the cliff — <½ of 4K speed |

The non-linear knee holds: 4K→8K (7%→15% offload) costs only ~15%, but 8K→16K (15%→33%) **halves** throughput.

> 🌡 **Thermal wrinkle:** warm 16K here (10.3) came out *lower* than the earlier no-warmup batch 16K (~14.7) — backwards from the usual warmup-helps rule. Likely **thermal throttling**: `benchmark_ctx.sh` runs ~18 inferences before reaching 16K, and 16K leans hardest on the CPU (33% offload), so heat soak → CPU clocks down → the CPU-heavy case takes the biggest hit. The quick one-shot loop ran 16K only once (less heat soak) so it dodged the throttle. **Lesson: sustained-load steady-state TPS can be *lower* than burst TPS, especially for CPU-offloaded models. Burst benchmarks are optimistic for anything leaning on the CPU.**

**This is partial offload in action.** As `num_ctx` rises:
- SIZE in `ollama ps` grows (KV cache *allocation* inflating: ~0.5 GB → 1 GB → 2 GB)
- CPU% in the processor split climbs deterministically (more layers spilled to RAM)
- Generation slows — but **non-linearly** (see below)
- In a second terminal, `nvidia-smi dmon` or `htop` shows the GPU idling more while CPU works

#### ⭐ Key finding: offload cost is NON-LINEAR (gentle slope, then a cliff)

```
 7% offload → ~22 TPS  ┐ barely any cost — the GPU still does 85-93% of the work
15% offload → ~21 TPS  ┘
33% offload → ~14.7 TPS  ← the "knee": CPU work is now a big fraction
```
A *little* offload (7-15%) is nearly free; *a lot* (30%+) falls off a cliff. So the earlier "partial offload is a performance cliff" framing is really a **gentle slope then a cliff** — the knee on this hardware is ~20-30% offload. **Practical rule: if a model spills slightly over VRAM (≤15% offload), don't panic — you lose almost nothing.**

#### Why PCIe bandwidth is NOT the bottleneck (common misconception)

In partial offload, **weights do NOT stream across PCIe each token.** The CPU-offloaded layers keep their weights in system RAM and are *computed by the CPU*; the GPU layers compute in VRAM. The only thing crossing PCIe per token is the hidden-state activation vector (`hidden_dim` ≈ 4096 floats ≈ 8 KB) — about 1 microsecond on any PCIe gen. Negligible.

The real cost of an offloaded layer is **CPU RAM bandwidth (~60 GB/s) + AVX compute**, which is only ~3-5× slower than a GPU layer (VRAM ~192 GB/s). That's why a handful of CPU layers barely hurts:
```
per-token time ≈ (GPU-layer time, VRAM-bound) + (CPU-layer time, RAM-bound)   [PCIe hop is free]
warm 4K  (~2 CPU layers): ~42 ms → 24 TPS
warm 8K  (~5 CPU layers): ~48 ms → 21 TPS   (+3 layers ≈ +6 ms, barely noticeable)
warm 16K (~11 CPU layers): ~68 ms → 14.7 TPS (+6 layers ≈ +20 ms, now it bites)
```

> Note: KV-cache *allocation* (num_ctx) inflates VRAM SIZE and forces the offload, but the KV-cache *read* per token scales with the ACTUAL sequence length (~110 tokens here), not num_ctx. So bigger num_ctx hurts only by evicting model layers to CPU — not by making attention reads heavier in this short-prompt test.

Reset to a sane default afterward (don't leave a huge context as your daily setting):
```bash
ollama stop llama3.1:8b
```

**Lesson:** for daily 6 GB VRAM use, **stay at or below 8B at Q4_K_M with modest context (≤4K)**. Large-context requests on a borderline model push you off the performance cliff. If you genuinely need long context, drop to a 3B model (e.g., `llama3.2:3b`) which has room for 8K+ context fully on GPU.

### Phase 7: Same Model, Different Quantizations (you already have both!)

This is the gold experiment and you have exactly the two models for it:
- `llama3.1:8b` → default **Q4_K_M** (~4.9 GB)
- `llama3.1:8b-instruct-q4_K_S` → **Q4_K_S** (~4.7 GB)

Same model, same family, same training — the *only* difference is the quantization scheme. Q4_K_S is the "small" 4-bit variant (slightly more aggressive rounding); Q4_K_M is "medium" (keeps a bit more precision on important tensors).

**Step 1 — Compare memory footprint + speed:**
```bash
for model in llama3.1:8b llama3.1:8b-instruct-q4_K_S; do
  echo "=== $model ==="
  ~/local-ai/scripts/benchmark.sh "$model" 4096
  ollama stop "$model"
done
```
Watch `ollama ps` SIZE and the processor split for each. Does the ~200 MB-smaller Q4_K_S reduce the CPU offload %? Does it gain TPS?

**Step 2 — Compare quality on identical prompts** (run a few of YOUR standard test prompts through both):
```bash
PROMPTS=(
  "Translate to French: 'Local AI is transformative.'"
  "Write a Python function that checks if a string is a palindrome."
  "Summarize the causes of World War 1 in 3 bullet points."
  "What is 17 * 24? Show your reasoning."
)
for model in llama3.1:8b llama3.1:8b-instruct-q4_K_S; do
  echo "############ $model ############"
  for p in "${PROMPTS[@]}"; do
    echo "--- PROMPT: $p"
    ollama run "$model" "$p"
    echo
  done
  ollama stop "$model"
done
```

What you'll likely observe:
- **Size/speed:** Q4_K_S is slightly smaller and *may* shave the offload a touch, but on 6 GB both are borderline — expect both to still partially offload at 4K and land within a few TPS of each other.
- **Quality:** for general chat/translation, near-identical. On precise tasks (math, code, structured output), Q4_K_M occasionally edges out Q4_K_S because it preserves more precision.

**Lesson:** Q4_K_M is the sweet spot for 7-8B on 6 GB — it's the default for a reason. Q4_K_S saves a little memory at a small, often-imperceptible quality cost. Going *up* to Q8_0 (~8 GB) is pointless here: it won't fit and the quality gain over Q4_K_M is ~1%. (Want to see the official sizes for every quant? `ollama show llama3.1:8b --modelfile` and the model's page on ollama.com.)

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

### Phase 10b: Tour ALL 11 Endpoints with `curl` (the "everything is HTTP" lesson)

Before going to Python, prove to yourself that **every CLI action is just an HTTP call**. Install `jq` for pretty JSON output:

```bash
sudo apt install -y jq
```

Then run each of these in your shell. Compare the output to its CLI equivalent.

```bash
# 1. List installed models (replaces `ollama list`):
curl http://localhost:11434/api/tags | jq

# 2. List running (loaded in VRAM) models (replaces `ollama ps`):
curl http://localhost:11434/api/ps | jq

# 3. Show details on a model (replaces `ollama show <model>`):
curl http://localhost:11434/api/show -d '{"name":"llama3.2:3b"}' | jq

# 4. Single completion (replaces `ollama run <model> "prompt"`):
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2:3b",
  "prompt": "Why is the sky blue?",
  "stream": false
}' | jq

# 5. Chat with message history (no CLI equivalent — only via API):
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2:3b",
  "messages": [
    {"role": "user", "content": "Hello, what is your name?"}
  ],
  "stream": false
}' | jq

# 6. Get embeddings (no CLI equivalent — needed for RAG, Week 9):
ollama pull nomic-embed-text   # tiny 137M embedding model
curl http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "Local AI is awesome"
}' | jq '.embedding | length'   # prints embedding dimension (768 for nomic)

# 7. Pull a new model (replaces `ollama pull <model>`):
curl http://localhost:11434/api/pull -d '{"name":"phi3:mini"}'
# Returns streaming progress; won't pretty-print with jq cleanly

# 8. Delete a model (replaces `ollama rm <model>`):
curl -X DELETE http://localhost:11434/api/delete -d '{"name":"phi3:mini"}'
# Returns nothing on success (HTTP 200)

# 9. Create custom model from Modelfile (replaces `ollama create`):
# (We'll do this on Day 5 — Modelfile + custom personas)

# 10. OpenAI-compatible chat:
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role":"user","content":"Hello"}]
  }' | jq

# 11. OpenAI-compatible completion (raw text completion, not chat):
curl http://localhost:11434/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "prompt": "The capital of France is"
  }' | jq

# Side-by-side proof: these produce SAME data, different format:
ollama list                              # pretty terminal table
curl -s http://localhost:11434/api/tags | jq   # raw JSON
```

**Lesson:** every CLI command is an HTTP call. Once you see this, the door opens — any language, any tool, any environment that can make HTTP requests can drive Ollama. The CLI is convenient but optional.

### Phase 11: REST API Directly with `curl` and `requests` (Python)

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

# All five models you have on disk — a full size + family + quant sweep:
#   3B tier:  llama3.2:3b (Meta), qwen2.5:3b (Alibaba)
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

These are the **5 models I have on disk** (no more downloads). This table is my complete Day 2 dataset.

| Model | Quant | Disk size | Peak VRAM | Processor split | TPS (decode) | Subjective quality |
|---|---|---|---|---|---|---|
| **Llama 3.2 3B @ ctx=4096** ✅ | Q4_K_M | 2.0 GB | ~3 GB | **100% GPU** | **~56 tok/s** | 7/10 |
| Qwen 2.5 3B @ ctx=4096 | Q4_K_M | 1.9 GB | ~3 GB | (expect 100% GPU) | __ tok/s | __ /10 |
| **Qwen 2.5 7B @ ctx=4096** ✅ | Q4_K_M | 4.7 GB | **4.9 GB** | **100% GPU (fits!)** | **32.5 tok/s (warm median)** | 8/10 |
| **Llama 3.1 8B @ ctx=4096** ⚠ | Q4_K_M | 4.9 GB | **5.9 GB** | **7%/93% CPU/GPU** | **25.4 tok/s (warm median)** | 9/10 |
| **Llama 3.1 8B @ ctx=2048** ⚠ | Q4_K_M | 4.9 GB | **5.6 GB** | **8%/92% CPU/GPU (still partial)** | **~21 tok/s (no gain)** | 9/10 |
| **Llama 3.1 8B @ ctx=8192** | Q4_K_M | 4.9 GB | **6.4 GB** | **15%/85% CPU/GPU** | **~21 tok/s** | 9/10 |
| **Llama 3.1 8B @ ctx=16384** | Q4_K_M | 4.9 GB | **8.0 GB** | **33%/67% CPU/GPU** | **~14.7 tok/s** | 9/10 |
| Llama 3.1 8B-instruct-q4_K_S @ ctx=4096 | Q4_K_S | 4.7 GB | __ GB | __ | __ tok/s | __ /10 |

**Bolded rows = real measurements on RTX 1000 Ada Laptop (6 GB) with prompt "Write a 3-sentence summary of why the sky is blue.", averaged over 2 runs.** Fill in the rest as you run Phases 5-7.

Comparisons this dataset unlocks:
- **Size sweep (same family):** Qwen 2.5 3B vs Qwen 2.5 7B → how much does 2× params cost in TPS?
- **Same-size, different family:** Llama 3.2 3B vs Qwen 2.5 3B → which is smarter/faster at 3B?
- **Quantization (same model):** Llama 3.1 8B Q4_K_M vs Q4_K_S → does the smaller quant escape offload? quality delta?
- **Context-induced offload (same model):** Llama 3.1 8B at 4K / 8K → watch the CPU% and TPS cliff grow

Key validated findings so far:
- 3B @ 4K = 56 TPS, 100% GPU placement → matches memory-bandwidth theory (192 GB/s ÷ 2 GB ≈ 96 max → 60% efficiency = 58 predicted)
- 8B @ both 4K and 2K = ~21 TPS, both partial offload — `num_ctx: 2048` gave NO meaningful TPS improvement
- The ~2.7× speed ratio between 3B and 8B = exactly what bandwidth math predicts

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

1. **Llama 3.1 8B Q4_K_M does NOT fully fit on 6 GB regardless of context size.** Both `num_ctx: 4096` and `num_ctx: 2048` produce near-identical TPS (warm 4K median ≈ 25), both with partial offload (`7%/93%` and `8%/92%` respectively). My intuition that "smaller context = fits on GPU" was wrong: weights (4.9 GB) dominate, not KV cache (0.25-0.5 GB). To actually reduce *weights*, use the `llama3.1:8b-instruct-q4_K_S` quant (~4.7 GB) or drop to a 3B model — or just accept ~21 TPS. **Always run `ollama ps` to verify your processor split, AND do the weights-vs-KV-cache math before assuming a fix will work.**
2. **TPS drops ~2× from 3B to 8B.** 3B → ~56 tok/s, 8B → ~25 tok/s (warm). Roughly 2× the params ≈ half the speed. Memory bandwidth is the bottleneck.
3. **Single-run TPS numbers lie — benchmark properly.** Eight back-to-back identical curl requests gave TPS from 11.1 to 25.1 (2× spread). Causes: GPU clock ramp-up (slow first runs), output-length variance, prompt-length (KV cache) effect, and partial-offload CPU jitter. Real benchmarks need warmup + fixed `num_predict` + fixed `seed` + median of 5+ runs. (`~/local-ai/scripts/benchmark.sh`) **Concrete proof: Qwen 2.5 7B measured 15.1 TPS on a cold single run but 32.5 TPS warm median — a 2.15× difference. Trusting the cold number would have led to the wrong model choice.**
4. **Decode TPS is a function of prompt+output length, not a constant.** Same model, same num_ctx: `"Hello"` → ~24 TPS, `"Hello explain me quantum physics"` → ~17 TPS. Each new token reads the K/V cache for ALL prior tokens, so longer context = more memory bandwidth per token. Compare models with identical prompt + `num_predict`.
5. **Partial offload cost is NON-LINEAR — a gentle slope, then a cliff (measured warm with `benchmark_ctx.sh`).** 4K = 7% CPU → **24.1 TPS**; 8K = 15% CPU → **20.6 TPS** (only −15% despite 2× the offload); 16K = 33% CPU → **10.3 TPS** (−57%, the cliff). A *little* offload (≤15%) is nearly free because the GPU still does 85%+ of the work; *a lot* (30%+) bites. **Bonus finding:** warm 16K (10.3) came out *lower* than a cold one-shot 16K (~14.7) — thermal throttling under sustained load, and CPU-offloaded cases throttle hardest. So burst TPS > steady-state TPS for offloaded models. (My original estimate of ~21→12→7 was too pessimistic on the slope but right about the cliff.)
6. **Q4_K_S vs Q4_K_M = small memory save for a usually-imperceptible quality cost.** Same model, the "small" 4-bit quant is ~200 MB lighter; quality is near-identical for chat/translation, with Q4_K_M occasionally better on precise math/code. Going *up* to Q8_0 (~8 GB) is pointless on 6 GB — it won't fit and the gain over Q4_K_M is ~1%.
7. **`num_ctx` is a useful lever but NOT the dominant one for 8B-class on 6 GB.** Halving context saves only ~250 MB of KV cache, but the weights are the dominant cost (~4.9 GB for 8B Q4_K_M). On a card ~600 MB short, that 250 MB doesn't escape offload. To actually fix it, reduce the *weights* (smaller quant or smaller model).
8. **Memory-bandwidth-bound theory predicts my warm numbers within ~5%.** Formula: RTX 1000 Ada (~192 GB/s) ÷ model_size_in_VRAM × efficiency = predicted TPS. With proper warmup, measured efficiency is ~80-83% (not the 60% I first guessed from cold runs). Warm validation: Qwen 2.5 7B @ 4.9 GB, 100% GPU → 192÷4.9 = 39 max × 83% = 32.4 predicted, **32.5 measured** ✔. Llama 3.1 8B @ 5.9 GB w/ 7% offload → ~27 if fully resident, minus offload tax = **25.4 measured** ✔. **Decode is memory-bandwidth-bound, full stop** — the single most important performance principle in local inference at this size class.
9. **On 6 GB you have a 3-corner triangle: {model size, context length, fully-on-GPU} — pick any two.** Llama 3.1 8B + 4K + fully-on-GPU is impossible; 8B + 2K = still partial; a 7B like Qwen 2.5 7B + 4K = closer; Llama 3.2 3B + 8K = comfortable. This trade-off recurs at every VRAM tier in the field.
10. **The Python `ollama` library is a one-line install** and turns local AI into a programmable resource. This is the moment LLMs go from "thing I chat with" to "thing I build with."
11. **OpenAI-compat works flawlessly.** Same Python code that hits OpenAI's API works against local Ollama by changing one URL. This is why the ecosystem coalesced around this protocol.
12. **`keep_alive: 0` unloads immediately** — useful for compare-models scripts that hit different models in sequence (otherwise the first stays in VRAM for 5 min and blocks the second on a 6 GB card).
13. **`eval_count` and `eval_duration` in API responses** give you exact decode-phase timing. Use these for benchmarks instead of wall-clock.
14. **The `ollama` CLI is just a wrapper around HTTP calls.** `ollama list` = `curl /api/tags`, `ollama run X "Y"` = `curl /api/generate -d '{"model":"X","prompt":"Y"}'`, etc. Any HTTP client in any language can drive Ollama.
15. **`jq` is essential for shell API exploration.** `curl ... | jq` makes JSON readable. Install: `sudo apt install -y jq`.
16. **GPU utilization at 70% is a SYMPTOM, not a problem in itself.** Either (a) memory-bandwidth-bound decode (normal — even fully-on-GPU models hit 80-95%, not 100%), or (b) partial CPU offload making the GPU wait. Diagnose with `ollama ps`: `X%/Y% CPU/GPU` = offload; `100% GPU` = fully resident.
17. **Same tier ≠ same fit — measured proof.** Qwen 2.5 7B loads at **4.9 GB → 100% GPU**, but Llama 3.1 8B loads at **5.9 GB → 7% CPU offload**, despite both being "7-8B Q4_K_M" and nearly identical on disk (4.7 vs 4.9 GB). The ~1 GB VRAM gap comes from architecture: Llama has more params (8.0B vs 7.6B), more layers (32 vs 28), and 2× the KV heads (8 vs 4 → ~2× the KV cache). **On 6 GB, prefer Qwen 2.5 7B over Llama 3.1 8B if you want a 7B-class model fully on GPU.** Lesson: always check `ollama ps` for the real VRAM footprint — disk size is a poor predictor. **Warm benchmark confirms the speed too: Qwen 32.5 vs Llama 25.4 TPS median (~28% faster). Qwen 2.5 7B is the best 7B-class daily driver on this hardware — it fits fully AND runs faster.**
18. **Cold-load penalty is real and separate from decode speed.** First request after a model loads: `total_duration` includes ~8-10s to read the weights from disk into VRAM (measured 14.08s total vs 4.23s decode for cold Llama 3.1 8B). Subsequent requests are warm (keep-alive) and `total_duration` drops to ~decode time. Never benchmark the first (cold) run.
19. **PCIe bandwidth is NOT the bottleneck in CPU offload (common misconception).** The CPU-offloaded layers keep their weights in system RAM and are *computed by the CPU* — weights do not stream across PCIe per token. Only the hidden-state activation vector (~8 KB) crosses the bus per token (~1 µs, negligible). The real cost is **CPU RAM bandwidth (~60 GB/s) + AVX compute**, which is ~3-5× slower than a GPU layer (VRAM ~192 GB/s). This is why offloading a *few* layers barely hurts but offloading *many* falls off a cliff.
20. **num_ctx inflates VRAM by *allocation*, not by per-token read cost (for short prompts).** Going 4K→8K→16K grew SIZE 5.9→6.4→8.0 GB because Ollama pre-allocates the full KV-cache buffer. But with only ~110 tokens of real context, the KV *read* per token is tiny regardless. So large num_ctx hurts purely by evicting model layers to CPU — a second-order effect, not a direct attention-cost effect, until you actually fill the context.

---

## 🐛 Troubleshooting Cheat Sheet

| Symptom | Likely cause | Fix |
|---|---|---|
| "CUDA out of memory" on 8B model | Other GPU app running, or KV cache too big | Close ComfyUI/games; set `num_ctx` lower (e.g., 2048) |
| 8B model loads but slow (~20 tok/s instead of expected 30+) | Partial CPU offload (`X%/Y% CPU/GPU` in `ollama ps`) | Try `num_ctx: 2048`; if still partial, switch to Q4_K_S or smaller model |
| TPS varies wildly between runs (e.g. 11-25 TPS) | Cold start + output variance + CPU offload jitter | Use the proper benchmark script with warmup + fixed `num_predict` + median |
| Same model, different TPS for different prompts | KV cache scales with prompt+output length | Expected. Compare with same prompt+`num_predict`+`seed` |
| `import ollama` fails | Not in venv, or package not installed | `source ~/local-ai/.venv/bin/activate && uv pip install ollama` |
| `Connection refused` from Python | Ollama service stopped | `sudo systemctl status ollama`; restart if not running |
| Streaming chunks come as bytes not strings | API returning raw bytes | Use `ollama` library (handles this); or `chunk.decode('utf-8')` with raw requests |
| `eval_duration` is None or 0 | Asked for a model that didn't load successfully | Check `ollama ps` and Ollama logs (`journalctl -u ollama`) |
| Compare-models script: 2nd model takes 5 min to start | First model still warm in VRAM | Use `keep_alive: 0` in the previous call, OR `ollama stop <model>` between |

---

## ✅ Done When

- [ ] You measured TTFT and TPS for at least 3 different models (3B, 7B, 8B tier) — all from your existing 5
- [ ] You watched VRAM in real time as a model loaded (saw the jump on `nvidia-smi`)
- [ ] You observed partial CPU offload and watched it grow by cranking `num_ctx` on Llama 3.1 8B (4K → 8K → 16K)
- [ ] You compared two quants of the same model (Llama 3.1 8B Q4_K_M vs Q4_K_S)
- [ ] You wrote and ran a Python script (`01_basic_chat.py`) that calls Ollama
- [ ] You wrote a streaming script (`02_streaming_chat.py`) — tokens appear live
- [ ] You hit the REST API with raw `requests` once
- [ ] You used the OpenAI-compatible endpoint from Python
- [ ] You built and used a compare-models utility (`05_compare_models.py`)
- [ ] Your VRAM math estimate is within ~0.5 GB of measured VRAM for one model
- [ ] All scripts saved in `~/local-ai/scripts/` (version-controllable)
- [ ] You called at least 5 different API endpoints with raw `curl` from the shell
- [ ] You understand that `ollama list`, `ollama run X`, `ollama pull X`, etc. are just CLI shortcuts for HTTP calls

---

## 🧾 Command Cheatsheet (Quick Lookup)

> Everything from Day 1-2 in one place. Format: **"I want to ___ → run this."** Base URL for API is `http://localhost:11434`.

### Running & chatting
| I want to… | Command |
|---|---|
| Chat interactively | `ollama run qwen2.5:7b` |
| Exit the chat | `/bye` (inside the chat) |
| One-shot prompt (print + exit) | `ollama run qwen2.5:7b "Explain X in 3 lines"` |
| Change a setting mid-chat | `/set parameter num_ctx 8192` (inside the chat) |
| In-chat help | `/?` (inside the chat) |

### Increase context window / generation options (API body)
| I want to… | Add to the JSON body |
|---|---|
| **Increase context window** | `"options": {"num_ctx": 8192}` |
| Cap output length | `"options": {"num_predict": 100}` |
| Make output deterministic | `"options": {"seed": 42, "temperature": 0}` |
| Lower / raise creativity | `"options": {"temperature": 0.2}` (focused) … `1.0` (creative) |

### VRAM & loaded-model management
| I want to… | Command |
|---|---|
| See what's loaded + CPU/GPU split | `ollama ps` |
| **Unload a model now (free VRAM)** | `ollama stop <model>` |
| Watch VRAM live | `watch -n 1 nvidia-smi` |
| Keep a model warm forever | env `OLLAMA_KEEP_ALIVE=-1`, or `"keep_alive": -1` per request |
| Unload right after a request | `"keep_alive": 0` in the request body |

### Benchmarking (your scripts)
| I want to… | Command |
|---|---|
| TPS of one model | `~/local-ai/scripts/benchmark.sh <model> <num_ctx>` |
| TPS across context sizes | `~/local-ai/scripts/benchmark_ctx.sh <model> "4096 8192 16384"` |
| Quick one-off TPS | `curl -s localhost:11434/api/generate -d '{...}' \| python3 -c "..."` |

### Model library
| I want to… | Command |
|---|---|
| List installed models | `ollama list` |
| **Load / pull a new model** | `ollama pull <model>` (e.g. `ollama pull mistral:7b`) |
| Pull any GGUF from HuggingFace | `ollama pull hf.co/<user>/<repo>:<quant>` |
| See a model's details / params | `ollama show <model>` |
| Delete a model | `ollama rm <model>` |

### Disk & storage (models live under the service user, not `~`)
| I want to… | Command |
|---|---|
| Total Ollama disk usage | `sudo du -sh /usr/share/ollama/.ollama/` |
| Biggest model files | `sudo du -h /usr/share/ollama/.ollama/models/blobs/* \| sort -hr \| head` |
| Free space in WSL2 | `df -h ~` |

### API endpoints
| I want to… | Endpoint |
|---|---|
| Single completion | `POST /api/generate` |
| Chat w/ message history | `POST /api/chat` |
| List installed models | `GET /api/tags` |
| List loaded models | `GET /api/ps` |
| Get embeddings (for RAG later) | `POST /api/embeddings` |
| OpenAI drop-in | `POST /v1/chat/completions` |

### Ollama service control
| I want to… | Command |
|---|---|
| Restart Ollama | `sudo systemctl restart ollama` |
| Check service status | `systemctl status ollama` |
| Follow live logs | `journalctl -u ollama -f` |
| Verify GPU/CUDA backend | `journalctl -u ollama \| grep -i cuda` |

### My hardware quick-reference (RTX 1000 Ada, 6 GB) — measured
| Model | Fits? | Warm TPS | Use for |
|---|---|---|---|
| `llama3.2:3b` | ✅ 100% GPU | ~56 | fast iteration, agents, RAG dev |
| `qwen2.5:3b` | ✅ 100% GPU | (bench it) | fast, multilingual |
| `qwen2.5:7b` | ✅ 100% GPU | ~32.5 | **best 7B daily driver** |
| `llama3.1:8b` (Q4_K_M) | ⚠ 7% CPU offload | ~25 | best quality, accept the offload |
| `llama3.1:8b-instruct-q4_K_S` | ? (Phase 7 — bench it) | ? | lighter-quant test |

> Rule of thumb on 6 GB: **default to `qwen2.5:7b`** (fits + fast). Drop to `llama3.2:3b` when you need speed or long context. Reach for `llama3.1:8b` only when you want its specific answer quality and can tolerate ~25 TPS.

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
