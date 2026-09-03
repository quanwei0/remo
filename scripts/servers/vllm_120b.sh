#!/bin/bash
# GPT-OSS-120B on 4 GPUs (tensor parallel). --enable-auto-tool-choice --tool-call-parser openai is REQUIRED by
# FinanceHarness (OpenAI tool calling); without it every FinanceGym task returns an empty answer after 1 step.
# AppWorld/Formula use plain chat completions and are unaffected by the flags.
PORT=${PORT:-8125}; TP=${TP:-4}; UTIL=${UTIL:-0.92}
export TIKTOKEN_RS_CACHE_DIR=${TIKTOKEN_RS_CACHE_DIR:-$HOME/.cache/tiktoken-rs-cache} TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-$HOME/.cache/tiktoken-py}
mkdir -p "$TIKTOKEN_RS_CACHE_DIR" "$TIKTOKEN_CACHE_DIR"   # a world-writable /tmp cache owned by another user breaks the harmony tokenizer
exec vllm serve openai/gpt-oss-120b --served-model-name GPT-OSS-120B -tp "$TP" --port "$PORT" \
  --max-model-len 131072 --gpu-memory-utilization "$UTIL" --enable-auto-tool-choice --tool-call-parser openai
