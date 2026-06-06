#!/bin/bash
# Day 2 — Proper single-model TPS benchmark.
# Fixed prompt + fixed output length (num_predict) + fixed seed + a warmup run,
# then 5 measured runs, and reports the MEDIAN tok/s (robust to cold-start noise).
# Usage:  ./benchmark.sh [model] [num_ctx]    e.g.  ./benchmark.sh qwen2.5:7b 4096
MODEL="${1:-llama3.1:8b}"
NUM_CTX="${2:-2048}"
NUM_PREDICT=100   # always generate exactly 100 tokens for a fair comparison
RUNS=5

echo "Benchmarking $MODEL (num_ctx=$NUM_CTX, num_predict=$NUM_PREDICT)"

# Warmup (discarded) — gets the model loaded and GPU clocks boosted
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
