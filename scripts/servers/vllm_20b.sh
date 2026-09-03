#!/bin/bash
# GPT-OSS-20B on 1 GPU.
PORT=${PORT:-8126}; UTIL=${UTIL:-0.9}
export TIKTOKEN_RS_CACHE_DIR=${TIKTOKEN_RS_CACHE_DIR:-$HOME/.cache/tiktoken-rs-cache} TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-$HOME/.cache/tiktoken-py}
mkdir -p "$TIKTOKEN_RS_CACHE_DIR" "$TIKTOKEN_CACHE_DIR"
exec vllm serve openai/gpt-oss-20b --served-model-name GPT-OSS-20B --port "$PORT" \
  --max-model-len 131072 --gpu-memory-utilization "$UTIL" --enable-auto-tool-choice --tool-call-parser openai
