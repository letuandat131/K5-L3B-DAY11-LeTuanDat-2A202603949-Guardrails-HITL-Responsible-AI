"""Vertex AI Express REST helper for keys starting with ``AQ.``."""
from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any

VERTEX_EXPRESS_URL = (
    "https://aiplatform.googleapis.com/v1/publishers/google/models/"
    "{model}:generateContent?key={api_key}"
)


def is_vertex_express_key(api_key: str | None = None) -> bool:
    """Return True when the configured Google API key is a Vertex Express key."""
    key = (api_key if api_key is not None else os.environ.get("GOOGLE_API_KEY", "")).strip()
    return key.startswith("AQ.")


def _prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        parts: list[str] = []
        for item in prompt:
            if isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(prompt)


def generate_vertex_content_sync(
    *,
    model: str,
    api_key: str,
    user_prompt: str,
    system_instruction: str | None = None,
    temperature: float = 0.4,
    max_tokens: int = 1024,
    timeout: int = 45,
) -> str:
    """Synchronously invoke the Vertex AI Express ``generateContent`` HTTP endpoint."""
    payload: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}

    url = VERTEX_EXPRESS_URL.format(model=model, api_key=api_key.strip())
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Vertex Express HTTP {exc.code}: {body[:300]}") from exc

    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()


async def generate_vertex_content(
    *,
    model: str,
    api_key: str,
    user_prompt: str,
    system_instruction: str | None = None,
    temperature: float = 0.4,
    max_tokens: int = 1024,
) -> str:
    """Async wrapper over ``generate_vertex_content_sync``."""
    return await asyncio.to_thread(
        generate_vertex_content_sync,
        model=model,
        api_key=api_key,
        user_prompt=user_prompt,
        system_instruction=system_instruction,
        temperature=temperature,
        max_tokens=max_tokens,
    )


class VertexExpressNemoLLM:
    """NeMo Guardrails ``LLMModel`` adapter backed by Vertex Express REST."""

    def __init__(self, *, model_name: str, api_key: str):
        self._model_name = model_name
        self._api_key = api_key

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def provider_name(self) -> str:
        return "vertex_express"

    @property
    def provider_url(self) -> str:
        return "https://aiplatform.googleapis.com/v1/publishers/google/models"

    async def generate_async(self, prompt: Any, *, stop=None, **kwargs):
        from nemoguardrails.types import LLMResponse

        _ = (stop, kwargs)
        text = await generate_vertex_content(
            model=self._model_name,
            api_key=self._api_key,
            user_prompt=_prompt_to_text(prompt),
            temperature=0.0,
            max_tokens=128,
        )
        return LLMResponse(content=text)

    async def stream_async(self, prompt: Any, *, stop=None, **kwargs):
        from nemoguardrails.types import LLMResponseChunk

        resp = await self.generate_async(prompt, stop=stop, **kwargs)
        yield LLMResponseChunk(delta_content=resp.content or "")
