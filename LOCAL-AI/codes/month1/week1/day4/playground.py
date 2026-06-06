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
Edit MODEL / PROMPT below to taste.
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
