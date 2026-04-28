"""Tests for the lenient JSON parser used by ``LLMClient``."""

from __future__ import annotations

import json

import pytest

from mcts_qcm.llm import _extract_first_json_object, _strip_markdown_fence, parse_json_lenient


def test_strip_markdown_fence_with_lang() -> None:
    text = "```json\n{\"a\": 1}\n```"
    assert _strip_markdown_fence(text) == '{"a": 1}'


def test_strip_markdown_fence_without_lang() -> None:
    text = "```\n{\"a\": 1}\n```"
    assert _strip_markdown_fence(text) == '{"a": 1}'


def test_strip_markdown_fence_no_fence() -> None:
    text = '{"a": 1}'
    assert _strip_markdown_fence(text) == '{"a": 1}'


def test_extract_first_json_object_handles_prose_prefix() -> None:
    raw = 'Sure, here is your JSON: {"a": 1, "b": [2, 3]}'
    assert _extract_first_json_object(raw) == '{"a": 1, "b": [2, 3]}'


def test_extract_first_json_object_handles_nested() -> None:
    raw = 'foo {"outer": {"inner": "}{}"} } bar'
    extracted = _extract_first_json_object(raw)
    assert json.loads(extracted) == {"outer": {"inner": "}{}"}}


def test_parse_json_lenient_clean() -> None:
    assert parse_json_lenient('{"x": 42}') == {"x": 42}


def test_parse_json_lenient_with_fence() -> None:
    assert parse_json_lenient('```json\n{"x": 42}\n```') == {"x": 42}


def test_parse_json_lenient_with_prose() -> None:
    raw = 'Here you go!\n```json\n{"x": 42}\n```\nHope that helps.'
    assert parse_json_lenient(raw) == {"x": 42}


def test_parse_json_lenient_raises_on_garbage() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_json_lenient("definitely not json at all")


def test_lite_llm_gemini_extra_kwargs_turns_off_thinking() -> None:
    from mcts_qcm.llm import _lite_llm_gemini_extra_kwargs

    assert _lite_llm_gemini_extra_kwargs("gemini/gemini-2.5-flash") == {"reasoning_effort": "none"}
    assert _lite_llm_gemini_extra_kwargs("openai/gpt-4o-mini") == {}
