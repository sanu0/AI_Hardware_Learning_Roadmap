# Week 1, Day 4 — Model Parameters in Depth (Sampling & Generation Knobs)

> **Goal:** Stop treating the model as a black box. Today you learn every generation parameter — `temperature`, `top_k`, `top_p`, `repeat_penalty`, `num_ctx`, `num_predict`, `seed`, `stop` — what each one *actually* does to the token probabilities, and how to dial them in for different tasks (factual vs creative vs deterministic). You'll build a **parameter playground** to sweep any knob and see the effect.
>
> **Time:** ~3 hours.
>
> **Why this matters:** The same model can feel dumb or brilliant depending on these settings. "My local model gives bad answers" is, 80% of the time, wrong parameters (usually default `temperature` too high for the task, or repetition with no penalty). Knowing these knobs is what separates "I ran a model" from "I tuned a model for my use case" — and it's the foundation for Day 5 (baking presets into a custom model) and for RAG/agents later.

---

## 📋 Today's Checklist

- [ ] Read a model's baked-in defaults (`ollama show --modelfile`, `/show parameters` in the REPL)
- [ ] Sweep `temperature` (0 → 1.6) and watch output go from deterministic → chaotic
- [ ] Confirm `seed` reproducibility (same seed+temp = same output; temp 0 = always deterministic)
- [ ] Sweep `top_k` and `top_p`; understand fixed-count vs cumulative-mass cutoff
- [ ] See how `temperature` × `top_p` interact
- [ ] Induce a repetition loop, then kill it with `repeat_penalty`
- [ ] Control output length with `num_predict`; halt early with `stop` sequences
- [ ] Recap `num_ctx` (from Day 2) and which params cost VRAM/speed (most don't!)
- [ ] Set params 3 ways: API `options`, `/set parameter` in the REPL, (preview) a Modelfile
- [ ] Build `playground.py` — sweep any one numeric knob across values
- [ ] Derive task presets (factual / balanced / creative) and save them
- [ ] Save all scripts to `~/local-ai/scripts/`

---

## 🧠 Concepts I'm Learning Today

### 0. The sampling pipeline (the mental model everything hangs on)

For each new token, the model outputs a **logit** (raw score) for every token in its vocabulary (~150K for Qwen). Those logits become a probability distribution and then ONE token is picked. The parameters reshape that pipeline:

```
logits (one score per vocab token)
   │  top_k   → keep only the K highest-scoring tokens
   │  top_p   → keep the smallest set whose probabilities sum to ≥ P
   │  repeat_penalty → down-weight tokens seen recently
   │  temperature → flatten (high) or sharpen (low) the distribution
   ▼
softmax → probability distribution → SAMPLE one token (seeded RNG)
```

**Two families of knobs:**
- **Truncation/filtering** (`top_k`, `top_p`, `min_p`) — decide *which* tokens are even eligible.
- **Shape/randomness** (`temperature`) — decide *how randomly* you pick among the eligible ones.
- Plus **repetition control** (`repeat_penalty`) and **length/stop** (`num_predict`, `stop`, `num_ctx`).

### 1. `temperature` — the randomness dial

Temperature divides the logits before softmax:
- **`temperature = 0`** → greedy: always pick the single highest-probability token → **fully deterministic**. Best for facts, code, extraction, anything where you want *the* answer.
- **`0.2–0.4`** → focused, mostly deterministic, slight variation. Good for code/RAG/Q&A.
- **`0.7–0.8`** (Ollama default 0.8) → balanced chat. Natural but reliable.
- **`1.0–1.3`** → creative, surprising, more errors. Good for brainstorming/fiction.
- **`>1.5`** → often incoherent (especially on small models).

Intuition: low temp **sharpens** the distribution (the model gets more confident in its top pick); high temp **flattens** it (long-shot tokens get a real chance).

### 2. `top_k` — keep the K best candidates

Only the `k` highest-probability tokens are eligible; the rest are discarded, then you sample among the survivors. Ollama default `40`. Lower `k` (e.g., 10) = safer/more focused; higher = more diverse. It's a **fixed count** regardless of how confident the model is.

### 3. `top_p` (nucleus sampling) — keep the top P probability mass

Sort tokens by probability, then keep the smallest group whose cumulative probability ≥ `p`. Ollama default `0.9`. Unlike `top_k`, this is **adaptive**: when the model is confident (one token has 0.95 prob) the nucleus is tiny; when it's unsure (many similar options) the nucleus is large. This is why `top_p` is usually preferred over `top_k`.

### 4. How `temperature`, `top_k`, `top_p` combine

The filters (`top_k`, `top_p`) decide the candidate set; `temperature` decides how randomly you pick within it. Practical guidance:
- **Tune `temperature` first** — it has the biggest visible effect.
- Leave `top_p ≈ 0.9` and `top_k ≈ 40` as sane defaults; only touch them if output is too samey (raise) or too wild (lower).
- For **fully deterministic** output, `temperature = 0` makes the others irrelevant (greedy ignores sampling).

### 5. `repeat_penalty` (+ `repeat_last_n`) — stop the loops

Small models love to repeat ("I think that I think that I think…"). `repeat_penalty` (default `1.1`) divides the logit of recently-seen tokens by that factor, making them less likely. `repeat_last_n` (default `64`) is how many recent tokens it looks back over.
- `1.0` = no penalty (loops likely on small models)
- `1.1` = gentle (good default)
- `1.3+` = strong (can hurt fluency / break legit repetition like code indentation)

### 6. `num_ctx` — context window (recap from Day 2)

Max tokens (prompt + history + output) the model can "see." **Ollama default is 4096 today** (older builds defaulted to 2048 — the classic gotcha; always set it explicitly). Bigger `num_ctx` = bigger KV cache = more VRAM (Day 2: on 6 GB this is what pushed Llama 3.1 8B into CPU offload). It's the **only** sampling-time knob that meaningfully costs VRAM.

### 7. `num_predict` — max tokens to generate

Caps the **output** length. Ollama default `-1` = generate until the model emits its end token or context fills. `num_predict: 100` = stop after 100 tokens (we used this for fair benchmarks). `-2` = fill the remaining context. More tokens = longer wall time (decode is linear in tokens).

### 8. `seed` — reproducibility

The RNG seed for sampling. Same `seed` + same params + same prompt → **identical output** (great for tests/benchmarks). With `temperature = 0` output is deterministic anyway (no RNG used). Default seed is unset → each run differs.

### 9. `stop` — stop sequences

A list of strings; generation halts the moment one appears. Essential for structured output (stop at `"\n\n"`, `"```"`, `"Q:"`, etc.) and for keeping chat turns clean. Example: `"options": {"stop": ["\nUser:", "\n\n"]}`.

### 10. Advanced samplers (know they exist; rarely needed)

- **`min_p`** — keep tokens with prob ≥ `min_p × (max token prob)`. A newer, often-better alternative to `top_p`. Try `0.05`.
- **`mirostat`** (1 or 2) with `mirostat_tau` / `mirostat_eta` — adaptively targets a constant "surprise" level; an alternative to temperature/top_p. Off by default (`0`).
- **`tfs_z`**, **`typical_p`** — tail-free / typical sampling. Niche.

### 11. Ollama default parameter values (memorize the common ones)

| Param | Default | Notes |
|---|---|---|
| `temperature` | 0.8 | lower for factual/code |
| `top_k` | 40 | fixed-count filter |
| `top_p` | 0.9 | nucleus (cumulative-mass) filter |
| `min_p` | 0.0 | off |
| `repeat_penalty` | 1.1 | anti-repetition |
| `repeat_last_n` | 64 | lookback window |
| `num_ctx` | 4096 | KV-cache size (older builds 2048) |
| `num_predict` | -1 | until stop/EOS |
| `seed` | 0 (unset) | random each run |
| `mirostat` | 0 | off |

### 12. Three ways to set parameters

1. **Per request (API `options`)** — what we've been doing: `options={"temperature": 0.2}`. Highest priority; overrides everything for that call.
2. **In the REPL** — `ollama run qwen2.5:7b` then `/set parameter temperature 0.2`. Applies for that session.
3. **Baked into a Modelfile** (Day 5) — `PARAMETER temperature 0.2` so a custom model *defaults* to your preset.

### 13. Task presets (your cheat sheet)

| Task | temperature | top_p | repeat_penalty | notes |
|---|---|---|---|---|
| Facts / RAG / extraction | 0–0.2 | 0.9 | 1.1 | want *the* answer |
| Code | 0–0.3 | 0.9 | 1.1 | low temp; correctness > creativity |
| Balanced chat | 0.7–0.8 | 0.9 | 1.1 | the all-rounder |
| Brainstorm / creative | 1.0–1.2 | 0.95 | 1.15 | embrace surprise |
| Deterministic / testing | 0 | — | 1.1 | + fixed `seed` |

---

## 🛠 Step-by-Step (What I'm Doing)

> Activate venv: `cd ~/local-ai && source .venv/bin/activate`. Default model below is `qwen2.5:7b`; the looping demo uses `llama3.2:3b` (small models loop more).

### Phase 1: Read the defaults baked into a model

```bash
# Full Modelfile (system prompt, template, and any PARAMETER lines):
ollama show qwen2.5:7b --modelfile

# Just the parameters/template summary:
ollama show qwen2.5:7b

# Or inside the REPL:
ollama run qwen2.5:7b
# >>> /show parameters
# >>> /show info
# >>> /bye
```

Note which params the model ships with — some models bake in their own `temperature`, `top_p`, `stop` tokens. Your per-request `options` override these.

### Phase 2: Temperature sweep (determinism → chaos)

Save as `~/local-ai/scripts/14_temperature_sweep.py`:

```python
"""Day 4 — Same prompt across temperatures. Low = deterministic, high = chaotic."""
import ollama

MODEL = "qwen2.5:7b"
PROMPT = "Give one creative name for a coffee shop on Mars. Just the name."

for temp in [0.0, 0.4, 0.8, 1.2, 1.6]:
    print(f"\n=== temperature={temp} ===")
    for run in range(3):
        r = ollama.chat(model=MODEL,
                        messages=[{"role": "user", "content": PROMPT}],
                        options={"temperature": temp})
        print(f"  run {run+1}: {r['message']['content'].strip()}")
```

Run it. Expect: `temp=0.0` → the same name all 3 runs (deterministic); `temp=1.6` → wild, possibly nonsensical names. The sweet spot for usable creativity is ~0.8–1.2.

### Phase 3: Seed = reproducibility

Save as `~/local-ai/scripts/15_seed_demo.py`:

```python
"""Day 4 — Seed makes temperature>0 reproducible; temp=0 is deterministic regardless."""
import ollama

MODEL = "qwen2.5:7b"
PROMPT = "Invent a fantasy character name."

def gen(temperature, seed):
    r = ollama.chat(model=MODEL, messages=[{"role": "user", "content": PROMPT}],
                    options={"temperature": temperature, "seed": seed})
    return r["message"]["content"].strip()

print("temp=1.0, seed=42 (run twice) — should MATCH:")
print(" ", gen(1.0, 42)); print(" ", gen(1.0, 42))

print("\ntemp=1.0, seed=None-ish (different seeds) — should DIFFER:")
print(" ", gen(1.0, 1)); print(" ", gen(1.0, 2))

print("\ntemp=0 (seed irrelevant) — always the same:")
print(" ", gen(0, 1)); print(" ", gen(0, 999))
```

### Phase 4 & 5: top_k and top_p (use the playground from the project below)

```bash
python playground.py top_k 5 40 100
python playground.py top_p 0.3 0.9 1.0
```

Observe: very low `top_k`/`top_p` = repetitive/safe; `top_p=1.0` = no nucleus cut (most diverse). Combined with the same `temperature`, lower filters tame the chaos.

### Phase 6: temperature × top_p interaction

```bash
# High temp but tight nucleus = creative-but-bounded:
python playground.py temperature 1.2          # wide open
# then compare mentally to running the playground with top_p baked low (edit SEED/opts)
```

Takeaway: `top_p` puts a *ceiling* on how weird high temperature can get. `temp=1.2, top_p=0.9` is far more usable than `temp=1.2, top_p=1.0`.

### Phase 7: Induce a repetition loop, then fix it

Save as `~/local-ai/scripts/16_repeat_penalty.py`:

```python
"""Day 4 — Small models loop without a repetition penalty. Watch it, then fix it."""
import ollama

MODEL = "llama3.2:3b"   # smaller model loops more readily
PROMPT = "Write a motivational paragraph about persistence. Keep going for a while."

for rp in [1.0, 1.1, 1.3]:
    print(f"\n=== repeat_penalty={rp} ===")
    r = ollama.chat(model=MODEL,
                    messages=[{"role": "user", "content": PROMPT}],
                    options={"repeat_penalty": rp, "num_predict": 200, "temperature": 0.8, "seed": 7})
    print(r["message"]["content"].strip())
```

At `1.0` you may see phrases/sentences repeat; `1.1` cleans it up; `1.3` is very strict (can feel choppy). This is the #1 fix for "my small model rambles in circles."

### Phase 8 & 9: num_predict and stop sequences

Save as `~/local-ai/scripts/17_length_and_stop.py`:

```python
"""Day 4 — Control output length (num_predict) and halt early (stop sequences)."""
import ollama

MODEL = "qwen2.5:7b"

print("=== num_predict=30 (hard cap) ===")
r = ollama.chat(model=MODEL,
                messages=[{"role": "user", "content": "Explain photosynthesis."}],
                options={"num_predict": 30})
print(r["message"]["content"].strip())

print("\n=== stop at first blank line (structured) ===")
r = ollama.chat(model=MODEL,
                messages=[{"role": "user", "content": "List 5 fruits, one per line, then explain each."}],
                options={"stop": ["\n\n"]})   # halt before the explanations
print(r["message"]["content"].strip())
```

`num_predict` caps tokens (output cut off mid-sentence if too low); `stop` halts cleanly when a delimiter appears — invaluable for forcing structured output later.

### Phase 10: Which params cost VRAM/speed?

Quick mental model (ties to Day 2):
- **`num_ctx`** → costs **VRAM** (bigger KV cache). The only sampling knob that does.
- **`num_predict`** → costs **time** (more tokens = longer, linear).
- **`temperature`, `top_k`, `top_p`, `repeat_penalty`, `seed`** → essentially **free** — they're cheap math on the logit vector each step. Tune them freely without worrying about your 6 GB budget.

### Phase 11: Set params 3 ways

```bash
# 1) Per request (API) — already doing this in every script.

# 2) In the REPL:
ollama run qwen2.5:7b
# >>> /set parameter temperature 0.2
# >>> /set parameter num_ctx 8192
# >>> Summarize the water cycle.
# >>> /bye

# 3) Modelfile (preview of Day 5):
#   FROM qwen2.5:7b
#   PARAMETER temperature 0.2
#   PARAMETER top_p 0.9
#   SYSTEM "You are a precise technical assistant."
# then: ollama create my-precise-qwen -f Modelfile
```

---

## 🔨 Project: Parameter Playground (`playground.py`)

A reusable tool to sweep ANY single numeric parameter and compare outputs side-by-side (everything else fixed, seed fixed, so you isolate the one knob).

Save as `~/local-ai/scripts/playground.py`:

```python
#!/usr/bin/env python3
"""
Day 4 capstone — Parameter Playground.
Sweep ONE numeric generation parameter across values and compare outputs.
Everything else (and the seed) is held fixed so you isolate the knob's effect.

Usage:
  python playground.py temperature 0 0.4 0.8 1.2 1.6
  python playground.py top_p 0.3 0.9 1.0
  python playground.py top_k 5 40 100
  python playground.py repeat_penalty 1.0 1.1 1.3
  python playground.py num_predict 20 60 150
Optional: set PROMPT / MODEL env-style by editing the constants below.
"""
import sys
import ollama

MODEL = "qwen2.5:7b"
PROMPT = "Write a 2-sentence product description for a smart water bottle."
SEED = 42   # fixed so ONLY the swept parameter changes the output


def parse(v):
    """'40' -> 40 (int), '0.8' -> 0.8 (float), '-1' -> -1 (int)."""
    try:
        return int(v)
    except ValueError:
        return float(v)


def main():
    if len(sys.argv) < 3:
        print("usage: python playground.py <param> <val1> <val2> ...")
        print("example: python playground.py temperature 0 0.8 1.4")
        sys.exit(1)

    param = sys.argv[1]
    values = [parse(v) for v in sys.argv[2:]]

    print(f"Model: {MODEL}\nParam: {param}\nPrompt: {PROMPT!r}\n")
    for val in values:
        print(f"{'='*64}\n{param} = {val}\n{'='*64}")
        r = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT}],
            options={param: val, "seed": SEED},
        )
        print(r["message"]["content"].strip(), "\n")


if __name__ == "__main__":
    main()
```

Run a few sweeps:
```bash
python playground.py temperature 0 0.8 1.4
python playground.py top_p 0.3 0.9 1.0
python playground.py repeat_penalty 1.0 1.1 1.3
```

### Bonus: task presets comparison (`presets.py`)

Save as `~/local-ai/scripts/presets.py`:

```python
"""Day 4 — Run the same prompt under 3 task presets to feel the difference."""
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
```

Keep the preset dict — you'll reuse it everywhere, and bake your favorite into a Modelfile on Day 5.

---

## 📊 My Observations (Fill In)

| Thing | Value / Note |
|---|---|
| Temperature where output started feeling "creative but usable" | ___ |
| Temperature where output became incoherent | ___ |
| Did `seed` reproduce output at temp=1.0? | yes / no |
| `repeat_penalty` that stopped llama3.2:3b looping | ___ |
| Default `temperature` shown by `ollama show qwen2.5:7b` | ___ |
| Preset you'll use as your daily default | factual / balanced / creative |

**Notes to self:**
- Which knob had the biggest visible effect? (probably temperature)
- Any model that baked in a surprising default?

---

## ⚠ Surprises & Lessons Learned (add yours)

1. **`temperature` is the master knob.** Most "bad output" complaints are just temp too high for the task. Facts/code → near 0; chat → ~0.8.
2. **`temperature = 0` is fully deterministic** — same input, same output every time. Perfect for tests, extraction, and debugging.
3. **`top_p` beats `top_k`** in practice because it adapts to the model's confidence. Leave it ~0.9.
4. **Small models loop without `repeat_penalty`.** 1.1 is the cheap fix; don't go overboard (1.3+ hurts fluency).
5. **Sampling knobs are basically free** — only `num_ctx` (VRAM) and `num_predict` (time) have a real cost. Tune temperature/top_p/repeat_penalty without touching your 6 GB budget.
6. **Per-request `options` override the model's baked-in defaults** — and a Modelfile (Day 5) lets you change the *defaults* themselves.
7. **`stop` sequences are how you get clean structured output** — worth remembering for RAG/agents.

---

## 🐛 Troubleshooting Cheat Sheet

| Symptom | Likely cause | Fix |
|---|---|---|
| Output is random/garbled | `temperature` too high | drop to 0.2–0.8; for facts use 0 |
| Same boring answer every time | `temperature` 0 or very low | raise to 0.7–1.0 for variety |
| Model repeats phrases/loops | no/low `repeat_penalty` (common on 3B) | set `repeat_penalty` 1.1–1.3 |
| Output cut off mid-sentence | `num_predict` too small | raise it or set `-1` |
| Can't reproduce a result | no fixed `seed` (with temp>0) | set `seed` + same params |
| Long convo truncated / forgets start | `num_ctx` too small | raise `num_ctx` (watch VRAM, Day 2) |
| Param change had no effect | typo in option name, or temp=0 makes sampling moot | check spelling; raise temp to see sampling knobs work |
| Model ignores my temperature | a baked-in Modelfile param or template | per-request `options` should override; verify with `ollama show` |

---

## ✅ Done When

- [ ] You read a model's baked-in params with `ollama show` / `/show parameters`
- [ ] You ran a temperature sweep and saw determinism → chaos
- [ ] You proved `seed` reproduces output at temp>0 (and temp=0 is deterministic)
- [ ] You swept `top_k` and `top_p` and can explain the difference
- [ ] You induced a repetition loop and fixed it with `repeat_penalty`
- [ ] You capped length with `num_predict` and halted early with a `stop` sequence
- [ ] You can name which params cost VRAM (`num_ctx`) vs are free (the samplers)
- [ ] You set a parameter all 3 ways (API, `/set parameter`, Modelfile preview)
- [ ] `playground.py` works and you used it on ≥2 parameters
- [ ] You have a `presets.py` with factual/balanced/creative dicts saved
- [ ] All scripts saved in `~/local-ai/scripts/`

---

## 🧾 Quick Reference (parameter cheat sheet)

| Param | Default | What it does | Tune when… |
|---|---|---|---|
| `temperature` | 0.8 | randomness (0=greedy) | facts→0, chat→0.8, creative→1.2 |
| `top_k` | 40 | keep K best tokens | rarely; lower to tame |
| `top_p` | 0.9 | keep top P prob-mass | leave ~0.9; raise for variety |
| `min_p` | 0.0 | keep ≥ min_p×max | try 0.05 as a top_p alternative |
| `repeat_penalty` | 1.1 | anti-repetition | 1.1–1.3 if it loops |
| `repeat_last_n` | 64 | lookback for penalty | rarely |
| `num_ctx` | 4096 | context window (VRAM!) | longer convos; mind 6 GB |
| `num_predict` | -1 | max output tokens | cap length / benchmarks |
| `seed` | unset | reproducibility | tests, fair comparisons |
| `stop` | — | halt on string(s) | structured output |

**Set per request:** `options={"temperature": 0.2, "top_p": 0.9, "num_ctx": 4096, "seed": 42}`

---

## 🔜 Next: `DAY_5.md` — `ollama` CLI Mastery + Modelfiles

Tomorrow you stop re-typing parameters and **bake them into a custom model**:
- **Modelfiles** — `FROM`, `PARAMETER`, `SYSTEM`, `TEMPLATE`, `MESSAGE`
- **`ollama create my-assistant -f Modelfile`** — your own model with your preset + persona built in
- **`ollama cp` / `ollama rm` / `ollama push`** — manage and (optionally) share models
- **Env vars** — `OLLAMA_KEEP_ALIVE`, `OLLAMA_NUM_PARALLEL`, `OLLAMA_MAX_LOADED_MODELS` (careful on 6 GB)
- **Project:** build `sanu-assistant` — Qwen 2.5 7B + your favorite balanced preset + a custom system prompt, runnable with a single `ollama run sanu-assistant`

That wraps **Week 1** — you'll have gone from "installed Ollama" to "running, driving, streaming, and tuning local models, with custom presets baked in." Week 2 goes under the hood: GGUF, quantization, and llama.cpp.
