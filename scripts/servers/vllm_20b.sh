#!/bin/bash
# GPT-OSS-20B on one GPU.
# TOOLS=1 adds --enable-auto-tool-choice --tool-call-parser openai, which FinanceHarness needs (OpenAI tool calling;
# without it every FinanceGym task returns an empty answer after one step). Leave it OFF for Formula and AppWorld:
# the paper's servers ran without it, and with the tool parser vLLM returns only the model's final channel as
# `content` (without it, commentary and final channels are concatenated), which changes what the agent sees.
PORT=${PORT:-8126}; UTIL=${UTIL:-0.9}; TOOLS=${TOOLS:-0}
export TIKTOKEN_RS_CACHE_DIR=${TIKTOKEN_RS_CACHE_DIR:-$HOME/.cache/tiktoken-rs-cache} TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-$HOME/.cache/tiktoken-py}
mkdir -p "$TIKTOKEN_RS_CACHE_DIR" "$TIKTOKEN_CACHE_DIR"
TOOL_FLAGS=""; [ "$TOOLS" = "1" ] && TOOL_FLAGS="--enable-auto-tool-choice --tool-call-parser openai"
exec vllm serve openai/gpt-oss-20b --served-model-name GPT-OSS-20B --port "$PORT" \
  --max-model-len 131072 --gpu-memory-utilization "$UTIL" ${TOOL_FLAGS}
