"""
Thin, provider-agnostic wrapper around the LLM used for:
  - probability estimation reasoning (ai/probability_engine.py)
  - social sentiment classification (ai/sentiment.py)

Swappable between Anthropic and OpenAI via config.LLM_PROVIDER so the rest
of the codebase never imports a provider SDK directly.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from config import settings
from logger import get_logger

log = get_logger(__name__)


class LLMClient:
    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or settings.LLM_PROVIDER
        self._anthropic = None
        self._openai = None

        if self.provider == "anthropic":
            import anthropic
            if not settings.ANTHROPIC_API_KEY:
                log.warning("anthropic_key_missing")
            self._anthropic = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        elif self.provider == "openai":
            import openai
            if not settings.OPENAI_API_KEY:
                log.warning("openai_key_missing")
            self._openai = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        elif self.provider == "openrouter":
            import openai
            if not settings.OPENROUTER_API_KEY:
                log.warning("openrouter_key_missing")
            self._openai = openai.AsyncOpenAI(
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.OPENROUTER_BASE_URL
            )
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.provider}")

    async def complete_text(self, prompt: str, max_tokens: int = 800, system: Optional[str] = None) -> str:
        if self.provider == "anthropic":
            resp = await self._anthropic.messages.create(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=max_tokens,
                system=system or "",
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(block.text for block in resp.content if block.type == "text")
        else:
            model = settings.LLM_MODEL if self.provider == "openrouter" else settings.OPENAI_MODEL
            resp = await self._openai.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=([{"role": "system", "content": system}] if system else [])
                + [{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""

    async def complete_json(self, prompt: str, max_tokens: int = 800) -> dict:
        """Requests strict JSON back and parses it, stripping code fences if present."""
        system = "Respond with ONLY valid JSON. No markdown fences, no commentary, no preamble."
        text = await self.complete_text(prompt, max_tokens=max_tokens, system=system)
        cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Attempt to salvage the first {...} block
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            log.error("llm_json_parse_failed", raw=text[:500])
            return {}


_client_singleton: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = LLMClient()
    return _client_singleton
