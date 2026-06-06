#!/bin/bash
# Day 1 sanity check — verify the whole local-AI stack is alive in one shot:
#   1) Ollama installed + service running, 2) GPU visible, 3) models present,
#   4) a quick test generation, 5) what's currently loaded in VRAM.
# Usage:  bash check_setup.sh   (inside WSL2 with Ollama installed)

echo "== Ollama version =="
ollama --version

echo; echo "== Ollama service active? =="
systemctl is-active ollama 2>/dev/null || echo "(systemd not managing ollama on this box)"

echo; echo "== GPU (nvidia-smi) =="
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv

echo; echo "== Installed models =="
ollama list

echo; echo "== Quick test generation (llama3.2:3b) =="
curl -s http://localhost:11434/api/generate \
  -d '{"model":"llama3.2:3b","prompt":"Say hello in 5 words.","stream":false}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['response'].strip())"

echo; echo "== Loaded in VRAM right now =="
ollama ps
