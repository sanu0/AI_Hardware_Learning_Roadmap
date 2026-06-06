#!/bin/bash
# Day 5 — Build the custom models from their Modelfiles, smoke-test, and show baked config.
# Custom models share the base weights (no re-download), so this is near-instant + tiny on disk.
# Usage:  bash build_assistant.sh   (run from this directory; needs qwen2.5:7b already pulled)
set -e
cd "$(dirname "$0")"

echo "== Disk used by Ollama BEFORE create =="
sudo du -sh /usr/share/ollama/.ollama/ 2>/dev/null || true

echo; echo "== Creating local-assistant (balanced) =="
ollama create local-assistant -f Modelfile

echo; echo "== Creating coder (low-temp) =="
ollama create coder -f coder.Modelfile

echo; echo "== Models (note: size shown is the SHARED base; real extra disk is tiny) =="
ollama list | grep -E "local-assistant|coder|qwen2.5:7b" || ollama list

echo; echo "== Disk used by Ollama AFTER create (should be ~unchanged) =="
sudo du -sh /usr/share/ollama/.ollama/ 2>/dev/null || true

echo; echo "== Smoke test: local-assistant =="
ollama run local-assistant "In one sentence, who are you?"

echo; echo "== Smoke test: coder =="
ollama run coder "one-line Python to reverse a string"

echo; echo "== Baked-in config for local-assistant =="
ollama show local-assistant --modelfile
