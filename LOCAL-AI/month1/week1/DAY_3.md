# Week 1, Day 3 — Streaming, Multi-Turn Chat & Conversation Memory

> **Goal:** Make your local LLM feel *fast* and *alive* by (a) streaming tokens as they generate, and (b) giving it memory across turns. By the end you'll have built a real **terminal chatbot** with streaming output, conversation history, swappable system prompts, and live tokens/sec — your first genuine Python+Ollama app.
>
> **Time:** ~3-4 hours.
>
> **Why this matters:** Every chat app you've ever used (ChatGPT, Claude, etc.) streams tokens and remembers the conversation. Today you learn that **the model itself does neither** — streaming is a transport choice and memory is *your* job. Understanding this is the difference between "I can call an API" and "I can build an AI product." Everything in Weeks 2-12 (RAG, agents, tool-calling) is built on the message-array pattern you master today.

---

## 📋 Today's Checklist

- [ ] Stream a response with the `ollama` library (`stream=True`) — tokens appear live
- [ ] Stream with raw `requests` — parse Ollama's **NDJSON** line-by-line
- [ ] Stream via the OpenAI-compatible endpoint — parse **SSE** (`data: {...}`)
- [ ] Measure TTFT (time-to-first-token) vs total time — feel why streaming wins on UX
- [ ] Prove the API is **stateless** — show the model forgets between separate requests
- [ ] Build a multi-turn loop — append assistant replies so the model "remembers"
- [ ] Swap **system prompts** — same model, wildly different personalities
- [ ] Quick tour of `temperature` / `top_p` / `top_k` (deep dive is Day 4)
- [ ] Watch VRAM/TTFT grow as the conversation gets longer (ties to Day 2 offload findings)
- [ ] Implement a **sliding-window** memory strategy to cap context growth
- [ ] 🔨 **Project:** build `chat.py` — streaming terminal chatbot with memory + slash-commands
- [ ] Save all scripts to `~/local-ai/scripts/`

---

## 🧠 Concepts I'm Learning Today

### 1. Streaming: what it actually is

By default, `"stream": false` makes Ollama generate the **entire** answer, then return it in one blob. With `"stream": true`, Ollama sends **each token as it's produced**, so you can print them live.

```
Non-streaming:  [....... 4 seconds of silence .......] "Here is the full answer."
Streaming:      "Here" "is" "the" "answer" ...          (words appear as typed)
```

**Key truth: streaming does NOT make generation faster.** Total time is identical. What changes is **perceived** latency — the user sees the first word in ~100-300 ms instead of staring at a blank screen for 4 seconds. That's a massive UX win for free.

### 2. Two streaming wire formats (don't confuse them)

| Endpoint | Format | Looks like |
|---|---|---|
| `/api/chat`, `/api/generate` (Ollama native) | **NDJSON** (newline-delimited JSON) | `{"message":{"content":"Hel"},"done":false}\n` per line |
| `/v1/chat/completions` (OpenAI-compatible) | **SSE** (Server-Sent Events) | `data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n` ... `data: [DONE]` |

- **NDJSON** = one complete JSON object per line. Parse with `json.loads(line)`.
- **SSE** = lines prefixed with `data: `, terminated by a blank line, ending with a literal `data: [DONE]`. Strip the `data: ` prefix, then parse.

You'll implement both today so the formats stop being mysterious.

### 3. The API is STATELESS — the model has NO memory (most important concept today)

This trips up *everyone*. Each request to Ollama is completely independent. The model does not remember anything from previous requests.

```
Request 1: "My name is Sanu."        → "Nice to meet you, Sanu!"
Request 2: "What is my name?"          → "I don't have that information." ← it forgot!
```

**Memory is an illusion that YOU create** by sending the entire conversation history back every single turn. The `messages` array IS the memory:

```python
messages = [
  {"role": "system",    "content": "You are a helpful assistant."},
  {"role": "user",      "content": "My name is Sanu."},
  {"role": "assistant", "content": "Nice to meet you, Sanu!"},   # ← you append the model's own reply
  {"role": "user",      "content": "What is my name?"},          # ← now it can answer
]
```

Every turn you: **append the user message → call the model → append the assistant reply.** The model re-reads the whole list each time. This is why "context window" matters so much (next point).

### 4. The three roles

| Role | Purpose |
|---|---|
| `system` | Instructions that steer behavior/personality. Usually the first message. The model treats it as high-priority standing orders. |
| `user` | What the human says. |
| `assistant` | What the model said (you store its replies here to build memory). |

### 5. Conversation growth has a real cost on 6 GB (ties to Day 2)

Because you resend the whole history each turn:
- **Prefill grows every turn** → TTFT creeps up as the conversation gets longer (the model must re-read all prior tokens before generating).
- **KV cache grows** → more VRAM used. On Day 2 you saw that a bigger `num_ctx` pushed Llama 3.1 8B into CPU offload. A long conversation does the same thing — it *fills* the context you allocated.
- **If history exceeds `num_ctx`** → oldest tokens get silently dropped (truncated), and the model "forgets" the start of the conversation.

So on your hardware: prefer `qwen2.5:7b` (fits 100% GPU) or a 3B for long chats, set `num_ctx` big enough to hold the convo, and use a memory strategy (point 6) to stop unbounded growth.

### 6. Memory strategies (how real apps stop context blowup)

| Strategy | How | Trade-off |
|---|---|---|
| **Full history** | Keep everything | Simplest; breaks when it exceeds `num_ctx` |
| **Sliding window** | Keep system + last N turns | Loses old context; cheap & predictable |
| **Summarization** | Periodically compress old turns into a short summary | Keeps gist, costs an extra LLM call (full treatment in the Agents weeks) |
| **Bigger `num_ctx`** | Allocate more context | Uses more VRAM → may trigger offload on 6 GB |

Today you'll implement the sliding window — the 80/20 solution.

### 7. Sampling knobs (intro — full deep dive Day 4)

These change *how* the model picks the next token. Today just know they exist; tune them tomorrow.

| Option | What it does | Default-ish |
|---|---|---|
| `temperature` | Randomness. 0 = deterministic, 0.7 = balanced, 1.2 = wild | ~0.7-0.8 |
| `top_p` | Nucleus sampling — sample from smallest set of tokens summing to p | 0.9 |
| `top_k` | Only consider the top k candidate tokens | 40 |
| `repeat_penalty` | Discourage repetition (small models loop more!) | 1.1 |

> ⚠ **Ollama context-window gotcha:** older docs say the default `num_ctx` is 2048; current Ollama defaults to **4096** (you saw `CONTEXT 4096` in `ollama ps` on Day 2). Either way — **always set it explicitly** for multi-turn chat so you control truncation.

---

## 🛠 Step-by-Step (What I'm Doing)

> Activate your venv first: `cd ~/local-ai && source .venv/bin/activate`. You already installed `ollama`, `requests`, and `openai` on Day 2.

### Phase 1: Streaming with the `ollama` library

Save as `~/local-ai/scripts/06_stream_lib.py`:

```python
"""Streaming with the ollama library — the simplest way to print tokens live."""
import ollama

stream = ollama.chat(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "Write a haiku about local AI, then explain it."}],
    stream=True,                       # ← the magic flag
)

for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)   # flush so it appears immediately
print()   # final newline
```

Run it: `python 06_stream_lib.py`. Watch the text type itself out instead of appearing all at once.

**Why `flush=True`?** Python buffers stdout by default; without flush, tokens would pile up and print in chunks. `flush=True` forces each token to the screen instantly.

### Phase 2: Streaming with raw `requests` (Ollama's NDJSON)

Save as `~/local-ai/scripts/07_stream_requests.py`:

```python
"""Streaming via raw HTTP — parse Ollama's newline-delimited JSON (NDJSON) by hand."""
import requests
import json

resp = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "qwen2.5:7b",
        "messages": [{"role": "user", "content": "Count from 1 to 10 with a word each."}],
        "stream": True,
    },
    stream=True,    # ← tells requests NOT to buffer the whole body; read it as it arrives
)

for line in resp.iter_lines():        # each line is one complete JSON object
    if not line:
        continue
    obj = json.loads(line)
    print(obj["message"]["content"], end="", flush=True)
    if obj.get("done"):               # final object carries timing stats
        print(f"\n\n-- {obj.get('eval_count')} tokens, "
              f"{obj['eval_count'] / (obj['eval_duration']/1e9):.1f} tok/s")
        break
```

Two different `stream` flags here, both required:
- `"stream": True` in the JSON → tells **Ollama** to stream.
- `stream=True` in `requests.post(...)` → tells the **requests library** not to wait for the whole body.

### Phase 3: Streaming via the OpenAI-compatible endpoint (SSE)

Two ways — the `openai` library (easy) and raw `requests` (to see the SSE wire format).

Save as `~/local-ai/scripts/08_stream_openai.py`:

```python
"""Streaming through the OpenAI-compatible endpoint — what most 3rd-party tools use."""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # api_key ignored by Ollama

stream = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[{"role": "user", "content": "Give me 3 tips for fast local inference."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:                               # delta is None on the final chunk
        print(delta, end="", flush=True)
print()
```

To *see* the raw SSE framing (educational), save `~/local-ai/scripts/08b_stream_sse_raw.py`:

```python
"""See the raw SSE wire format the OpenAI endpoint sends."""
import requests, json

resp = requests.post(
    "http://localhost:11434/v1/chat/completions",
    json={"model": "qwen2.5:7b",
          "messages": [{"role": "user", "content": "Say hello in 3 languages."}],
          "stream": True},
    stream=True,
)
for line in resp.iter_lines():
    if not line:
        continue
    s = line.decode("utf-8")
    if s.startswith("data: "):
        payload = s[len("data: "):]
        if payload == "[DONE]":
            break
        delta = json.loads(payload)["choices"][0]["delta"].get("content", "")
        print(delta, end="", flush=True)
print()
```

Notice the difference from Phase 2: SSE lines start with `data: ` and the stream ends with a literal `data: [DONE]`. Ollama-native NDJSON has neither — it's just raw JSON per line with a `"done": true` field.

### Phase 4: Measure TTFT — why streaming *feels* faster

Save as `~/local-ai/scripts/09_ttft.py`:

```python
"""Measure time-to-first-token vs total time to prove streaming's UX win."""
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
```

You'll see something like `TTFT: 180 ms | Total: 6.2 s`. The user perceives "instant" because the answer *starts* in 0.18s, even though it *finishes* in 6s. Non-streaming would show nothing for the full 6s.

### Phase 5: Prove the API is stateless (the "aha" moment)

Save as `~/local-ai/scripts/10_stateless_proof.py`:

```python
"""Prove the model has NO memory across separate requests — then fix it with history."""
import ollama

MODEL = "qwen2.5:7b"

print("=== Two SEPARATE requests (no shared history) ===")
ollama.chat(model=MODEL, messages=[{"role": "user", "content": "My name is Sanu."}])
r = ollama.chat(model=MODEL, messages=[{"role": "user", "content": "What is my name?"}])
print("Q: What is my name? →", r["message"]["content"].strip(), "\n")   # it won't know

print("=== ONE request WITH history (memory you built) ===")
messages = [
    {"role": "user",      "content": "My name is Sanu."},
    {"role": "assistant", "content": "Nice to meet you, Sanu!"},
    {"role": "user",      "content": "What is my name?"},
]
r = ollama.chat(model=MODEL, messages=messages)
print("Q: What is my name? →", r["message"]["content"].strip())          # now it knows
```

This single script makes the stateless concept click permanently. The model isn't "smart enough to remember" — *you* feed it the memory.

### Phase 6: Multi-turn loop (the core pattern)

Save as `~/local-ai/scripts/11_multiturn.py`:

```python
"""Minimal multi-turn loop: append user + assistant messages so the model remembers."""
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
    messages.append({"role": "assistant", "content": reply})   # ← critical: store the reply
print()
```

Drop the `messages.append({"role": "assistant", ...})` line and watch memory break — a great experiment.

### Phase 7: System prompts — same model, different soul

Save as `~/local-ai/scripts/12_system_prompts.py`:

```python
"""Same question, same model, different system prompts → totally different answers."""
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
```

The content is the same; the *voice* is radically different. System prompts are your cheapest, most powerful steering tool.

### Phase 8: Sampling quick tour (intro)

Save as `~/local-ai/scripts/13_sampling_intro.py`:

```python
"""Feel temperature: low = deterministic & safe, high = creative & chaotic."""
import ollama

MODEL = "qwen2.5:7b"
PROMPT = "Invent a name for a coffee shop on Mars. Just the name."

for temp in [0.0, 0.7, 1.3]:
    print(f"\n--- temperature={temp} ---")
    for _ in range(3):
        r = ollama.chat(model=MODEL,
                        messages=[{"role": "user", "content": PROMPT}],
                        options={"temperature": temp, "seed": None})
        print(" ", r["message"]["content"].strip())
```

At `temp=0` the three runs are ~identical; at `temp=1.3` they diverge wildly. (Full parameter deep-dive tomorrow, Day 4.)

### Phase 9: Watch context growth cost (ties to Day 2)

In one terminal, keep a watch running:
```bash
watch -n 1 'ollama ps'
```
Then run `11_multiturn.py` but with a long, many-turn conversation (add 10+ turns). Watch the `CONTEXT` and the processor split. With `qwen2.5:7b` it stays 100% GPU; switch `MODEL` to `llama3.1:8b`, crank `num_ctx` to 8192, and a long convo will start offloading — exactly the Day 2 cliff, now triggered by conversation length instead of a manual `num_ctx` test.

### Phase 10: Sliding-window memory

Cap how much history you send so context never blows up:

```python
def trim_history(messages, keep_turns=6):
    """Keep the system message + the last `keep_turns` user/assistant messages."""
    system = [m for m in messages if m["role"] == "system"][:1]
    rest = [m for m in messages if m["role"] != "system"]
    return system + rest[-keep_turns:]
```

Call `messages = trim_history(messages)` before each model call. The system prompt always survives; only old turns drop off. This is the pattern baked into the project below.

---

## 🔨 Project: Streaming Terminal Chatbot with Memory (`chat.py`)

Your first real app. Save as `~/local-ai/scripts/chat.py`:

```python
#!/usr/bin/env python3
"""
Local streaming chatbot with memory + slash-commands.
Run:  python chat.py
Commands:
  /reset            clear the conversation (keep system prompt)
  /system <text>    set a new system prompt (resets convo)
  /model <name>     switch model (e.g. /model llama3.2:3b)
  /save <file.md>   save the transcript to markdown
  /stats            show message count + last turn's tok/s
  /bye              quit
"""
import time
import ollama

DEFAULT_MODEL = "qwen2.5:7b"          # fits 100% GPU on 6 GB, ~32 tok/s (my Day 2 pick)
DEFAULT_SYSTEM = "You are a concise, helpful local AI assistant. Be clear and brief."
KEEP_TURNS = 8                         # sliding window: last N user/assistant messages
NUM_CTX = 4096                         # set explicitly so we control truncation


def trim_history(messages, keep_turns=KEEP_TURNS):
    system = [m for m in messages if m["role"] == "system"][:1]
    rest = [m for m in messages if m["role"] != "system"]
    return system + rest[-keep_turns:]


def main():
    model = DEFAULT_MODEL
    system_prompt = DEFAULT_SYSTEM
    messages = [{"role": "system", "content": system_prompt}]
    last_stats = ""

    print(f"🤖 Local chatbot | model: {model} | ctx: {NUM_CTX} | window: {KEEP_TURNS} msgs")
    print("Type /bye to quit, /reset to clear, /system <text>, /model <name>, /save <f>, /stats\n")

    while True:
        try:
            user = input("\033[1myou>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye! 👋")
            break
        if not user:
            continue

        # ---- slash commands ----
        if user == "/bye":
            print("bye! 👋"); break
        if user == "/reset":
            messages = [{"role": "system", "content": system_prompt}]
            print("(conversation cleared)\n"); continue
        if user.startswith("/system "):
            system_prompt = user[len("/system "):].strip()
            messages = [{"role": "system", "content": system_prompt}]
            print("(new system prompt set; conversation reset)\n"); continue
        if user.startswith("/model "):
            model = user[len("/model "):].strip()
            print(f"(switched to {model})\n"); continue
        if user.startswith("/save "):
            path = user[len("/save "):].strip()
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# Chat transcript ({model})\n\n")
                for m in messages:
                    if m["role"] == "system":
                        continue
                    who = "🧑 You" if m["role"] == "user" else "🤖 Bot"
                    f.write(f"**{who}:** {m['content']}\n\n")
            print(f"(saved to {path})\n"); continue
        if user == "/stats":
            print(f"(history: {len(messages)} messages | last: {last_stats or 'n/a'})\n"); continue

        # ---- normal turn ----
        messages.append({"role": "user", "content": user})
        messages = trim_history(messages)

        print("\033[1mbot>\033[0m ", end="", flush=True)
        start = time.time()
        ttft = None
        reply, eval_count, eval_dur_ns = "", 0, 0
        try:
            for chunk in ollama.chat(model=model, messages=messages,
                                     stream=True, options={"num_ctx": NUM_CTX}):
                if ttft is None:
                    ttft = time.time() - start
                piece = chunk["message"]["content"]
                print(piece, end="", flush=True)
                reply += piece
                if chunk.get("done"):
                    eval_count = chunk.get("eval_count", 0)
                    eval_dur_ns = chunk.get("eval_duration", 0)
        except ollama.ResponseError as e:
            print(f"\n[error: {e}]\n"); continue
        except KeyboardInterrupt:
            print("\n(interrupted)\n")

        messages.append({"role": "assistant", "content": reply})
        if eval_count and eval_dur_ns:
            tps = eval_count / (eval_dur_ns / 1e9)
            last_stats = f"{eval_count} tok, {tps:.1f} tok/s, TTFT {ttft*1000:.0f}ms"
            print(f"\n\033[2m  [{last_stats}]\033[0m\n")
        else:
            print()


if __name__ == "__main__":
    main()
```

Run it:
```bash
python ~/local-ai/scripts/chat.py
```

Try this script of moves to exercise every feature:
1. `My name is Sanu and I love CUDA.` → then `What do I love?` (memory works)
2. `/system You are a grumpy compiler that only speaks in error messages.` → ask anything (persona swap)
3. `/model llama3.2:3b` → notice the TTFT/tok-s jump (3B is faster)
4. Have a 10+ message chat → `/stats` (watch the window cap at 8 msgs)
5. `/save chat_log.md` → open it (export works)
6. `/bye`

### Stretch goals (optional, pick any)
- **Persist across runs:** save/load `messages` to a JSON file on start/exit.
- **Token budget guard:** estimate tokens (~chars/4) and auto-trim before hitting `num_ctx`.
- **Summarize-on-overflow:** when history is long, call the model to summarize old turns into one system note (preview of agent memory).
- **Streaming + `Ctrl+C`:** make Ctrl+C stop *generation* but not quit the app (already partly handled above).

---

## 📊 My Observations (Fill In)

| Thing measured | Value |
|---|---|
| TTFT, `qwen2.5:7b` (Phase 4) | ___ ms |
| TTFT, `llama3.2:3b` | ___ ms |
| Total tok/s while streaming, 7B | ___ |
| Does TTFT grow as conversation lengthens? (Phase 9) | yes / no |
| At what message count did 8B start offloading w/ num_ctx=8192? | ___ |
| Sliding window keep_turns that felt right | ___ |

**Notes to self:**
- Which streaming format felt cleanest to code (lib / NDJSON / SSE)?
- Best system prompt you discovered for your use case?

---

## ⚠ Surprises & Lessons Learned (add yours as you go)

1. **Streaming doesn't speed up generation — it speeds up *perception*.** Total time is the same; TTFT is what makes it feel instant.
2. **The model is stateless.** It remembers nothing between requests. The `messages` array is the entire memory, and you maintain it.
3. **You must append the assistant's own reply** to history or the model forgets what *it* just said.
4. **Ollama native streaming = NDJSON; OpenAI endpoint = SSE.** Different framing (`data: ` prefix + `[DONE]`), same idea.
5. **System prompts are the highest-leverage knob** — same model, completely different behavior, zero extra cost.
6. **Long conversations are a stealth `num_ctx` test** — history fills context, and on 6 GB an 8B model will start offloading mid-chat (Day 2's cliff, triggered by conversation length).
7. **Set `num_ctx` explicitly.** Defaults bite you; if history exceeds it, the oldest turns silently vanish.
8. _(add your own...)_

---

## 🐛 Troubleshooting Cheat Sheet

| Symptom | Likely cause | Fix |
|---|---|---|
| Tokens print all at once, not live | stdout buffering | add `flush=True` to `print(...)` |
| Raw `requests` returns one big blob | missing `stream=True` in `requests.post` | add it (separate from the JSON `"stream": true`) |
| `delta.content` is `None` (OpenAI stream) | final chunk has empty delta | guard with `if delta:` |
| Model "forgets" mid-conversation | history exceeded `num_ctx` → truncated | raise `num_ctx` or use sliding window |
| Conversation slows down each turn | prefill grows as history grows | trim history; use a smaller/faster model for long chats |
| 8B starts offloading during a long chat | KV cache + history filled VRAM | switch to `qwen2.5:7b`/3B, lower `num_ctx`, or trim harder |
| `ollama.ResponseError: model not found` | typo or not pulled | `ollama list` to check the exact tag |
| `Ctrl+C` kills the whole app | unhandled `KeyboardInterrupt` | wrap the stream loop in try/except (see `chat.py`) |

---

## ✅ Done When

- [ ] You streamed a response three ways: `ollama` lib, raw NDJSON, OpenAI SSE
- [ ] You measured TTFT and can explain why streaming feels faster despite equal total time
- [ ] You proved (with `10_stateless_proof.py`) that the model has no built-in memory
- [ ] You built a multi-turn loop that correctly appends assistant replies
- [ ] You swapped system prompts and saw the personality change
- [ ] You watched context/processor-split change during a long conversation
- [ ] You implemented a sliding window to cap history
- [ ] `chat.py` runs: streams, remembers, supports `/reset` `/system` `/model` `/save` `/stats` `/bye`
- [ ] All scripts saved in `~/local-ai/scripts/`

---

## 🧾 Quick Reference (streaming & chat patterns)

| I want to… | How |
|---|---|
| Stream (ollama lib) | `for c in ollama.chat(..., stream=True): c["message"]["content"]` |
| Stream (raw HTTP) | `requests.post(url, json={...,"stream":True}, stream=True)` → `resp.iter_lines()` → `json.loads(line)` |
| Stream (OpenAI lib) | `for c in client.chat.completions.create(..., stream=True): c.choices[0].delta.content` |
| Give the model memory | maintain a `messages` list; append `{"role":"user",...}` then the `{"role":"assistant",...}` reply |
| Set personality | first message `{"role":"system","content":"..."}` |
| Control truncation | pass `options={"num_ctx": 4096}` |
| Make output deterministic | `options={"temperature":0,"seed":42}` |
| Cap history growth | sliding window: keep system + last N messages |
| Print tokens live | `print(piece, end="", flush=True)` |

---

## 🔜 Next: `DAY_4.md` — Model Parameters in Depth

Tomorrow we turn every knob and measure what it does:
- **`temperature`, `top_p`, `top_k`** — the sampling triangle, with side-by-side outputs
- **`repeat_penalty`** — why small models loop, and how to stop it
- **`num_ctx`, `num_predict`, `seed`, `stop`** — the rest of the options
- **Per-model defaults** — reading them with `ollama show`, overriding per request
- **Build a "parameter playground"** script that sweeps a setting and prints the effect

Then Day 5: Modelfiles & `ollama create` — bake your favorite system prompt + parameters into a reusable custom model (`sanu-assistant`), so you stop re-typing them.

By end of Week 1: you're not just running LLMs — you're **building with them.**
