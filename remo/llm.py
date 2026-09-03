"""Minimal OpenAI-compatible chat helper (vLLM etc.). openai is imported lazily so the core
stays importable in environments that only need the policy."""
import os


def chat(prompt: str, model: str, base_url: str | None = None, max_tokens: int = 4096,
         temperature: float = 0.0, timeout: float = 600.0, system: str | None = None) -> str:
    import openai
    client = openai.OpenAI(api_key=os.environ.get("REMO_API_KEY", "EMPTY"),
                           base_url=base_url or os.environ.get("REMO_BASE_URL", "http://localhost:8125/v1"),
                           timeout=timeout, max_retries=2)
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    r = client.chat.completions.create(model=model, messages=msgs, max_tokens=max_tokens, temperature=temperature)
    return r.choices[0].message.content or ""
