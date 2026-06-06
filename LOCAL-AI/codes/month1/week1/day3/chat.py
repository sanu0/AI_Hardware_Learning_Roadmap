#!/usr/bin/env python3
"""
Day 3 capstone — Local streaming chatbot with memory + slash-commands.

Features: streaming output, conversation memory (sliding window), swappable system
prompt, model switching, transcript export, and live tokens/sec + TTFT.

Run (venv active):  python chat.py
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

DEFAULT_MODEL = "qwen2.5:7b"          # fits 100% GPU on 6 GB, ~32 tok/s (Day 2 pick)
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
