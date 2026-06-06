# Week 1, Day 5 — Ollama CLI Mastery & Modelfiles (Build Your Own Model)

> **Goal:** Master the full `ollama` command-line surface, then learn **Modelfiles** — the recipe format that lets you bake a base model + your favorite parameters + a custom system prompt into a brand-new model you run with one command. By the end you'll have built `local-assistant`: Qwen 2.5 7B with your Day-4 balanced preset and a custom persona, plus a `coder` variant.
>
> **Time:** ~2-3 hours.
>
> **Why this matters:** Up to now you've re-typed `options={...}` and system prompts on every call. A Modelfile makes that *permanent and reusable* — `ollama run local-assistant` and it's already tuned. This is also exactly how you'll later ship fine-tuned models, RAG personas, and agent backends. Plus the CLI/env-var knowledge here (`keep_alive`, `num_parallel`, `max_loaded_models`) is what keeps a 6 GB GPU from thrashing.

---

## 📋 Today's Checklist

- [ ] Tour the full `ollama` CLI (`pull`, `list`, `run`, `ps`, `stop`, `show`, `cp`, `rm`, `create`, `push`, `serve`)
- [ ] Write your first **Modelfile** (`FROM` + `PARAMETER` + `SYSTEM`)
- [ ] `ollama create local-assistant -f Modelfile` and run it
- [ ] Verify the custom model is a tiny manifest (NO re-download of weights)
- [ ] Bake your Day-4 balanced preset + a custom system prompt into it
- [ ] Try `MESSAGE` for few-shot priming (and understand `TEMPLATE` — and why not to touch it)
- [ ] Make a second model (`coder` preset) to feel "a preset *is* a model"
- [ ] Use `ollama cp` and `ollama rm` to manage models
- [ ] Set Ollama env vars **safely for 6 GB** (`KEEP_ALIVE`, `NUM_PARALLEL=1`, `MAX_LOADED_MODELS=1`) via systemd
- [ ] (Optional) understand `ollama push` and its privacy implications
- [ ] Build the capstone: `local-assistant` runnable with one command
- [ ] Save Modelfiles + build script to `~/local-ai/scripts/` (and `codes/.../day5/`)

---

## 🧠 Concepts I'm Learning Today

### 1. The full `ollama` CLI surface

Every one of these is just a client call to the local server (recap from Day 2). The complete set:

| Command | What it does |
|---|---|
| `ollama pull <model>` | download a model from the registry |
| `ollama list` | list installed models |
| `ollama ps` | show models currently loaded in VRAM (+ CPU/GPU split) |
| `ollama run <model> [prompt]` | chat (REPL) or one-shot |
| `ollama stop <model>` | unload a model from VRAM now |
| `ollama show <model>` | show details (`--modelfile`, `--parameters`, `--template`, `--system`) |
| `ollama create <name> -f <Modelfile>` | **build a custom model from a Modelfile** |
| `ollama cp <src> <dst>` | copy/rename a model |
| `ollama rm <model>` | delete a model (frees disk) |
| `ollama push <model>` | upload to ollama.com (needs an account) |
| `ollama serve` | run the server in the foreground (systemd already does this for you) |

### 2. Modelfile anatomy (the recipe format)

A Modelfile is a small text file (like a Dockerfile) describing a model:

```dockerfile
FROM qwen2.5:7b                       # base model (required) — local tag or hf.co/...
PARAMETER temperature 0.7             # bake in generation defaults (Day 4 knobs)
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
SYSTEM """You are a concise assistant."""   # default system prompt / persona
# TEMPLATE "..."                      # prompt format (ADVANCED — usually leave alone)
# MESSAGE user "Hi"                   # optional few-shot priming
# MESSAGE assistant "Hello!"
# ADAPTER ./my-lora.gguf              # attach a LoRA adapter (fine-tuning, later weeks)
# LICENSE "..."                       # license text
```

Key instructions:
- **`FROM`** (required) — the base model. Can be a local tag (`qwen2.5:7b`), a `hf.co/...` GGUF, or even a local `.gguf` file path.
- **`PARAMETER`** — bake in any Day-4 knob (`temperature`, `top_p`, `num_ctx`, `stop`, ...). These become the model's *defaults* (still overridable per request).
- **`SYSTEM`** — the default system prompt. Use `"""triple quotes"""` for multi-line.
- **`MESSAGE`** — prime the conversation with example turns (cheap few-shot).
- **`TEMPLATE`** — the raw prompt format. ⚠ **Don't override unless you know the model's exact template** — getting it wrong silently breaks chat quality. `SYSTEM` + `MESSAGE` cover 95% of needs.

### 3. What `ollama create` actually does (and why it's instant + tiny)

`ollama create local-assistant -f Modelfile` does **not** copy or re-download the 4.7 GB of weights. It writes a **new manifest** that points at the **same base-model blobs** plus a tiny new layer holding your params/system prompt. So:
- It's near-instant.
- It costs ~kilobytes of disk (the weights are shared with `qwen2.5:7b` via Ollama's blob deduplication — Day 1 concept).
- `ollama list` will show `local-assistant` at ~the same size as the base, but `du` confirms little extra disk used.

### 4. `ollama cp` and `ollama rm`

- `ollama cp qwen2.5:7b my-qwen-backup` — instant copy (shares blobs).
- `ollama rm local-assistant` — removes the manifest; shared base blobs stay (the base model still works).

### 5. `ollama push` — sharing (optional, privacy-sensitive!)

`ollama push <user>/<model>` uploads to ollama.com under your account (requires `ollama.com` signup + `ollama` key). ⚠ **On a company laptop, skip this unless you intend to make something public.** A pushed model based on `qwen2.5:7b` only uploads your tiny config layer (others pull the base separately), but the **system prompt and any examples become public**. Never push anything with internal/sensitive content. For this roadmap, building locally is enough.

### 6. Environment variables — and WHERE to set them (recap Day 1)

The **server** reads these, not your shell — so set them in the **systemd service**, not `~/.bashrc` (same lesson as `OLLAMA_MODELS` on Day 1):

```bash
sudo systemctl edit ollama
# add under [Service]:
#   Environment="OLLAMA_KEEP_ALIVE=-1"
#   Environment="OLLAMA_NUM_PARALLEL=1"
#   Environment="OLLAMA_MAX_LOADED_MODELS=1"
sudo systemctl restart ollama
```

| Env var | Effect | Safe value on 6 GB |
|---|---|---|
| `OLLAMA_KEEP_ALIVE` | how long a model stays warm (`-1`=forever, `0`=unload now, `5m`=default) | `-1` if you use one model a lot; else `5m` |
| `OLLAMA_NUM_PARALLEL` | concurrent requests per model (splits context) | **1** (more splits your tiny VRAM) |
| `OLLAMA_MAX_LOADED_MODELS` | how many models held in VRAM at once | **1** (2+ will OOM/offload on 6 GB) |
| `OLLAMA_MODELS` | where models are stored (Day 1) | default, or a bigger drive |
| `OLLAMA_HOST` | bind address/port (Day 2) | default `127.0.0.1:11434` |
| `OLLAMA_FLASH_ATTENTION` | enable flash attention (saves KV-cache VRAM) | `1` worth trying on 6 GB |

> **6 GB rule:** keep `NUM_PARALLEL=1` and `MAX_LOADED_MODELS=1`. Both default to letting Ollama load more, which on your card means OOM or heavy CPU offload. One model, one request at a time, is the sweet spot.

### 7. Where custom models live

Same place as everything else (Day 1): `/usr/share/ollama/.ollama/models/`. Your custom model adds a small manifest under `manifests/.../local-assistant/` and a tiny params blob; the big weight blobs are shared with the base. Confirm with `sudo du -sh` before/after `ollama create` — barely changes.

---

## 🛠 Step-by-Step (What I'm Doing)

### Phase 1: CLI tour

```bash
ollama list                       # what you have
ollama show qwen2.5:7b            # details
ollama show qwen2.5:7b --modelfile   # the recipe Ollama uses for it
ollama ps                         # what's loaded now
ollama run qwen2.5:7b "hi"        # one-shot
ollama stop qwen2.5:7b            # unload
```

### Phase 2: Write your first Modelfile

```bash
mkdir -p ~/local-ai/scripts/Month-1/DAY-5
cd ~/local-ai/scripts/Month-1/DAY-5
nano Modelfile
```

Paste (this is your balanced preset + persona):
```dockerfile
FROM qwen2.5:7b

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

SYSTEM """You are a concise, helpful local AI assistant running on a 6 GB laptop GPU.
Answer clearly and briefly. Prefer concrete examples and code. If unsure, say so."""
```

### Phase 3: Build it and run it

```bash
ollama create local-assistant -f Modelfile
ollama list | grep -E "local-assistant|qwen2.5:7b"   # note: tiny extra disk
ollama run local-assistant "In one sentence, who are you?"
```

Confirm it baked in: now you can run `ollama run local-assistant` *without* passing any options or system prompt — they're built in. Verify:
```bash
ollama show local-assistant --modelfile     # shows your PARAMETER + SYSTEM lines
ollama show local-assistant --system        # just the system prompt
```

### Phase 4: Prove the weights weren't re-downloaded

```bash
sudo du -sh /usr/share/ollama/.ollama/      # before/after create barely changes
# local-assistant shares qwen2.5:7b's 4.7 GB blob — only a tiny config layer is new
```

### Phase 5: Few-shot priming with `MESSAGE` (optional/advanced)

Make a Modelfile that primes a style with example turns:
```dockerfile
FROM qwen2.5:7b
SYSTEM """You answer like a terse Linux man page."""
MESSAGE user "how do I list files"
MESSAGE assistant "ls(1) — list directory contents. Usage: ls [-la] [path]"
MESSAGE user "how do I copy a file"
MESSAGE assistant "cp(1) — copy files. Usage: cp [-r] SOURCE DEST"
```
```bash
ollama create manpage-bot -f Modelfile.manpage
ollama run manpage-bot "how do I move a file"
```
The model mimics the primed style. (Leave `TEMPLATE` alone unless you truly need it — overriding it wrong breaks chat formatting.)

### Phase 6: A second preset = a second model (`coder`)

```dockerfile
FROM qwen2.5:7b
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
SYSTEM """You are a senior software engineer. Output working code first, then a one-line explanation. Be terse, no preamble."""
```
```bash
ollama create coder -f coder.Modelfile
ollama run coder "reverse a linked list in Python"
```
Same base model, totally different behavior — and each is just a tiny manifest.

### Phase 7: Manage models — `cp` / `rm`

```bash
ollama cp local-assistant local-assistant-v2   # snapshot before editing
ollama rm manpage-bot                            # remove an experiment
ollama list
```

### Phase 8: Env vars for 6 GB (via systemd)

```bash
sudo systemctl edit ollama
# [Service]
# Environment="OLLAMA_KEEP_ALIVE=-1"
# Environment="OLLAMA_NUM_PARALLEL=1"
# Environment="OLLAMA_MAX_LOADED_MODELS=1"
# Environment="OLLAMA_FLASH_ATTENTION=1"
sudo systemctl restart ollama

# verify the server picked them up:
systemctl show ollama | grep -i environment
ollama run local-assistant "hi"
ollama ps     # model should stay loaded (KEEP_ALIVE=-1) — note no "UNTIL" countdown shrinking
```

### Phase 9: (Optional) push — only if you mean to make it public

```bash
# Requires ollama.com account + `ollama` set up with your key. SKIP on a company laptop
# unless you intend a public share. Your SYSTEM prompt becomes public.
# ollama cp local-assistant <your-ollama-username>/local-assistant
# ollama push <your-ollama-username>/local-assistant
```

---

## 🔨 Project: `local-assistant` (your one-command tuned model)

The capstone is the Modelfile + a build script so anyone (or future-you) can recreate it in one shot.

`Modelfile` (balanced daily driver) and `coder.Modelfile` (coding) are in `codes/month1/week1/day5/`, plus `build_assistant.sh`:

```bash
bash ~/local-ai/scripts/Month-1/DAY-5/build_assistant.sh
# creates local-assistant + coder, smoke-tests both, shows their baked config
```

Then your daily workflow becomes:
```bash
ollama run local-assistant      # tuned chat, no options to remember
ollama run coder "..."          # low-temp code mode
```

### Stretch goals
- Add `PARAMETER stop "..."` to enforce clean turn boundaries.
- Make a `summarizer` model (low temp, system prompt "summarize in 3 bullets").
- Point `FROM` at a HuggingFace GGUF you like (`FROM hf.co/bartowski/...`).

---

## 📊 My Observations (Fill In)

| Thing | Value / Note |
|---|---|
| Disk used by Ollama before `create` (`sudo du -sh`) | ___ |
| Disk used after creating local-assistant + coder | ___ (should be ~same) |
| `ollama list` size shown for local-assistant | ___ |
| Does `ollama run local-assistant` work with no options passed? | yes / no |
| Did `OLLAMA_KEEP_ALIVE=-1` keep the model warm in `ollama ps`? | yes / no |
| Favorite custom model you made | ___ |

---

## ⚠ Surprises & Lessons Learned (add yours)

1. **A custom model is almost free** — `ollama create` writes a tiny manifest + config layer and **shares the base weights** (no re-download, ~KB of disk).
2. **Modelfile = reusable preset + persona** — bake Day-4 params + a system prompt once, then `ollama run local-assistant` forever.
3. **Per-request `options` still override** the Modelfile's `PARAMETER` defaults — Modelfile sets the *default*, not a lock.
4. **Leave `TEMPLATE` alone** unless you know the model's exact format — `SYSTEM` + `MESSAGE` handle most needs safely.
5. **Env vars belong in the systemd service**, not your shell (same lesson as `OLLAMA_MODELS`) — the server reads them, not your terminal.
6. **On 6 GB: `NUM_PARALLEL=1`, `MAX_LOADED_MODELS=1`.** Defaults try to do more and will OOM/offload your card.
7. **`OLLAMA_FLASH_ATTENTION=1`** can shrink KV-cache VRAM — worth enabling on a tight 6 GB budget.
8. **`ollama push` makes your SYSTEM prompt public** — skip on a work laptop unless intentional.

---

## 🐛 Troubleshooting Cheat Sheet

| Symptom | Likely cause | Fix |
|---|---|---|
| `ollama create` fails: "no such file" | wrong `-f` path or filename case | run from the Modelfile's dir; `ls` to confirm |
| Custom model ignores my system prompt | passing a `messages` system role that overrides it, or a stale build | rebuild; or don't send a system message (let the Modelfile's apply) |
| `FROM` model not found | base not pulled | `ollama pull qwen2.5:7b` first |
| Custom model seems huge in `ollama list` | that's the *shared* base size; `du` shows real extra disk is tiny | `sudo du -sh /usr/share/ollama/.ollama/` to confirm |
| Env var change had no effect | set in shell, not the service | `sudo systemctl edit ollama` + restart |
| Second model OOMs / very slow | `MAX_LOADED_MODELS>1` on 6 GB | set it to 1; `ollama stop` the other first |
| Chat quality broke after editing Modelfile | overrode `TEMPLATE` incorrectly | remove the `TEMPLATE` line; rely on `SYSTEM`/`MESSAGE` |
| `ollama push` rejected | not logged in / no account | needs ollama.com account; optional — skip if unsure |

---

## ✅ Done When

- [ ] You can name what each CLI command does (`create`, `cp`, `rm`, `show`, `push`, ...)
- [ ] You wrote a Modelfile with `FROM` + `PARAMETER` + `SYSTEM`
- [ ] `ollama create local-assistant -f Modelfile` succeeded and runs with no options
- [ ] You confirmed it shares the base weights (tiny extra disk)
- [ ] You made a second model (`coder`) with a different preset
- [ ] You used `ollama cp` and `ollama rm`
- [ ] You set 6 GB-safe env vars via systemd and verified with `systemctl show`
- [ ] You understand `ollama push`'s privacy implications (and chose to skip or not)
- [ ] `build_assistant.sh` recreates your models in one command
- [ ] Modelfiles + script saved to `~/local-ai/scripts/` and `codes/.../day5/`

---

## 🧾 Quick Reference (CLI + Modelfile)

**CLI:**
| Goal | Command |
|---|---|
| Build custom model | `ollama create <name> -f Modelfile` |
| Inspect recipe | `ollama show <model> --modelfile` |
| Copy / rename | `ollama cp <src> <dst>` |
| Delete | `ollama rm <model>` |
| Keep warm forever | env `OLLAMA_KEEP_ALIVE=-1` |
| Share (public!) | `ollama push <user>/<model>` |

**Modelfile skeleton:**
```dockerfile
FROM qwen2.5:7b
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
SYSTEM """Your persona here."""
```

**6 GB-safe env (systemd):** `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KEEP_ALIVE=-1`.

---

## 🎉 Week 1 Complete!

From "what's Ollama?" to **running, driving (Python + API), streaming, tuning, and now packaging** your own local models. You have a tuned `local-assistant` you built yourself.

## 🔜 Next: Week 2 — GGUF, Quantization & llama.cpp (under the hood)

- **GGUF format** — what's actually inside a model file
- **K-quants explained** — why Q4_K_M is the sweet spot (you've felt it; now understand it)
- **llama.cpp directly** — build it, run `llama-cli` / `llama-server`, the engine under Ollama
- **Converting & quantizing** your own models from HuggingFace → GGUF
- **KV-cache quantization** — squeeze more context into 6 GB

You've built the foundation. Week 2 is where you learn what's happening *beneath* Ollama — the knowledge that makes you dangerous.
