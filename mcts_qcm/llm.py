"""Thin LiteLLM wrapper with retries and robust JSON parsing.

The wrapper exposes a single ``LLMClient.chat_json`` method that returns a parsed
Python ``dict`` (or raises ``LLMError``). It handles:

- LiteLLM's multi-provider model strings (``openai/gpt-4o-mini``, ``anthropic/...``).
- ``response_format={"type": "json_object"}`` when supported.
- Stripping accidental markdown code fences from the model's reply.
- One automatic "fix your JSON" follow-up before giving up.
- Bounded retries on transient failures (timeouts, 5xx, JSON parse errors).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from mcts_qcm.prompts import JSON_FIX_SYSTEM

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when a structured LLM call fails after all retries."""


@runtime_checkable
class LLMClient(Protocol):
    """Minimal protocol the engine needs from any LLM client.

    Concrete implementations: ``LiteLLMClient`` for production, ``FakeLLMClient``
    in tests.
    """

    def chat_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.7,
        seed: int | None = None,
        timeout: float = 60.0,
    ) -> dict[str, Any]: ...


_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def _strip_markdown_fence(text: str) -> str:
    """Remove a surrounding ```json ... ``` fence if present."""
    text = text.strip()
    m = _FENCE_RE.match(text)
    if m:
        return m.group(1).strip()
    return text


def _extract_first_json_object(text: str) -> str:
    """Find the first balanced JSON object in `text`.

    Some models prepend a sentence even when told not to. We tolerate that by
    scanning for the first `{` and returning content up to its matching `}`.
    """
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def parse_json_lenient(content: str) -> dict[str, Any]:
    """Parse LLM-emitted JSON tolerating fences, prose, and stray markdown.

    Raises ``json.JSONDecodeError`` if no valid JSON object can be recovered.
    """
    cleaned = _strip_markdown_fence(content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        snippet = _extract_first_json_object(cleaned)
        return json.loads(snippet)


@dataclass
class LiteLLMClient:
    """Default LLM client backed by `litellm.completion`.

    Lazy-imports litellm so unit tests that swap in a fake client don't pay
    the import cost or require API keys.
    """

    max_retries: int = 2
    backoff_seconds: float = 1.5

    def chat_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.7,
        seed: int | None = None,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        import litellm  # type: ignore[import-not-found]

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        last_err: Exception | None = None
        last_content: str | None = None

        for attempt in range(self.max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "timeout": timeout,
                }
                if seed is not None:
                    kwargs["seed"] = seed
                # response_format is silently ignored by providers that don't
                # support it; LiteLLM normalizes the JSON-mode flag.
                kwargs["response_format"] = {"type": "json_object"}
                resp = litellm.completion(**kwargs)
                content = resp["choices"][0]["message"]["content"]
                last_content = content
                return parse_json_lenient(content)
            except json.JSONDecodeError as e:
                logger.warning(
                    "JSON parse failed (attempt %d/%d) for model %s: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    model,
                    e,
                )
                last_err = e
                if attempt < self.max_retries and last_content is not None:
                    messages = [
                        {"role": "system", "content": JSON_FIX_SYSTEM},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": last_content},
                        {
                            "role": "user",
                            "content": "Your previous reply was not valid JSON. Return ONLY the corrected JSON object now.",
                        },
                    ]
                else:
                    break
            except Exception as e:  # noqa: BLE001 - network / provider errors
                logger.warning(
                    "LLM call failed (attempt %d/%d) for model %s: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    model,
                    e,
                )
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * (attempt + 1))

        raise LLMError(f"LLM call to {model} failed after retries: {last_err}")
