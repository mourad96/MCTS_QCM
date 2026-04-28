"""Shared test fixtures: a deterministic ``FakeLLMClient``."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

import pytest

from mcts_qcm.config import MCTSConfig


@dataclass
class FakeLLMClient:
    """Deterministic LLM client for tests.

    `responder` is called with the prompt kwargs and must return the dict that
    ``LiteLLMClient.chat_json`` would have produced. A counter records every
    call so tests can assert how many times the engine hit the LLM.
    """

    responder: Callable[[dict[str, Any]], dict[str, Any]]
    calls: list[dict[str, Any]] = field(default_factory=list)

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
        payload = {
            "model": model,
            "system": system,
            "user": user,
            "temperature": temperature,
            "seed": seed,
            "timeout": timeout,
        }
        self.calls.append(payload)
        return self.responder(payload)


@pytest.fixture
def base_config() -> MCTSConfig:
    return MCTSConfig(
        model_gen="fake/gen",
        model_audit="fake/audit",
        k_children=2,
        iterations=3,
        max_depth=3,
        max_nodes=50,
        prune_on_failed_resource=False,
    )


@pytest.fixture
def fake_client_factory() -> Callable[[Callable[[dict[str, Any]], dict[str, Any]]], FakeLLMClient]:
    def _make(responder: Callable[[dict[str, Any]], dict[str, Any]]) -> FakeLLMClient:
        return FakeLLMClient(responder=responder)

    return _make


@pytest.fixture
def idea_counter() -> Callable[[], int]:
    """An incrementing counter so generator tests can build unique ideas."""
    counter = itertools.count(1)
    return lambda: next(counter)
