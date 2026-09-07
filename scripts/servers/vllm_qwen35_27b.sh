#!/bin/bash
# Qwen3.5-27B, tensor parallel over TP GPUs (4 by default; 2 also fits).
# Thinking is disabled: the reasoning parser splits the thinking channel off the reply and the chat-template default
# turns it off, so the agent sees only the answer. Leaving thinking on multiplies completion tokens and truncates
# long ReAct steps. TOOLS=1 adds the tool-call parser FinanceHarness needs; Formula and AppWorld run without it.
PORT=${PORT:-8127}; TP=${TP:-4}; UTIL=${UTIL:-0.9}; TOOLS=${TOOLS:-0}; MODEL=${MODEL:-Qwen/Qwen3.5-27B}
TOOL_FLAGS=""; [ "$TOOLS" = "1" ] && TOOL_FLAGS="--enable-auto-tool-choice --tool-call-parser hermes"
exec vllm serve "$MODEL" --served-model-name Qwen3.5-27B -tp "$TP" --port "$PORT" \
  --max-model-len 131072 --gpu-memory-utilization "$UTIL" \
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}' ${TOOL_FLAGS}
