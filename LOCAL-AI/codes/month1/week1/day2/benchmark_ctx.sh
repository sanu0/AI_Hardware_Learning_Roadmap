#!/bin/bash
# Day 2 — Sweep num_ctx for ONE model and show how context size affects placement + speed.
# For each context: a warmup run, then 5 measured runs -> median tok/s, plus the
# CPU/GPU processor split from `ollama ps` (so you can watch offload grow with context).
# Usage:  ./benchmark_ctx.sh [model] ["ctx1 ctx2 ..."]   e.g.  ./benchmark_ctx.sh llama3.1:8b "4096 8192 16384"
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
