"""Testy obsługi modeli "thinking" (Qwen) w scripts/chat_lmstudio.py -- bez serwera LM Studio:
`requests.post` jest podmieniany na atrapę zapisującą zapytania."""

import copy

import chat_lmstudio as cl
import pytest


class FakeResponse:
    def __init__(self, content, finish_reason="stop"):
        self._data = {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def patch_post(monkeypatch, contents):
    calls = []
    queue = list(contents)

    def fake_post(url, json=None, timeout=None, **kwargs):
        calls.append(copy.deepcopy(json))  # kod zmienia słownik przy ponowieniu -- zapisujemy kopię
        return FakeResponse(queue.pop(0))

    monkeypatch.setattr(cl.requests, "post", fake_post)
    return calls


def test_defaults_for_qwen_and_other_models():
    assert cl.resolve_generation_defaults("qwen/qwen3.8-27b", None, None) == (3000, "low")
    assert cl.resolve_generation_defaults("Qwen3-30B", None, None) == (3000, "low")
    assert cl.resolve_generation_defaults("bielik-11b-v3.1-instruct", None, None) == (500, None)


def test_explicit_arguments_win_over_defaults():
    assert cl.resolve_generation_defaults("qwen/qwen3.8-27b", 1200, "high") == (1200, "high")
    assert cl.resolve_generation_defaults("bielik-11b", 800, "low") == (800, "low")


def test_apply_thinking_params_merges_template_kwargs():
    p = cl._apply_thinking_params({}, "low", {"enable_thinking": False})
    assert p["reasoning_effort"] == "low"
    assert p["chat_template_kwargs"] == {"reasoning_effort": "low", "enable_thinking": False}
    assert cl._apply_thinking_params({}, None, None) == {}


def test_rewrite_followup_plain_model_single_short_call(monkeypatch):
    calls = patch_post(monkeypatch, ["Ile dni urlopu po 15 latach?"])
    out = cl.rewrite_followup("http://x/v1", "bielik", [{"role": "user", "content": "a"}], "A po 15 latach?")
    assert out == "Ile dni urlopu po 15 latach?"
    assert len(calls) == 1 and calls[0]["max_tokens"] == 120 and "reasoning_effort" not in calls[0]


def test_rewrite_followup_thinking_model_retries_once_on_empty(monkeypatch):
    calls = patch_post(monkeypatch, ["", "Pełne pytanie"])
    out = cl.rewrite_followup("http://x/v1", "qwen", [], "A po 15 latach?", reasoning_effort="low")
    assert out == "Pełne pytanie"
    assert [c["max_tokens"] for c in calls] == [800, cl.EMPTY_RETRY_MAX_TOKENS]
    assert all(c["reasoning_effort"] == "low" for c in calls)


def test_rewrite_followup_gives_empty_after_failed_retry(monkeypatch):
    patch_post(monkeypatch, ["", ""])
    assert cl.rewrite_followup("http://x/v1", "qwen", [], "q", reasoning_effort="low") == ""


class DummyRag:
    def search(self, *a, **k):
        return []


class DummyFiles:
    def search(self, *a, **k):
        return []


def test_answer_empty_response_raises_and_keeps_history_clean(monkeypatch):
    patch_post(monkeypatch, [""])
    monkeypatch.setattr(cl, "search_with_history", lambda *a, **k: [])
    history = []
    with pytest.raises(RuntimeError, match="pustą odpowiedź"):
        cl.answer("http://x/v1", "qwen", DummyRag(), DummyFiles(), history, "pytanie", 8,
                  rewrite_followups=False, max_tokens=3000, reasoning_effort="low")
    assert history == []


def test_answer_sends_thinking_params_and_updates_history(monkeypatch):
    calls = patch_post(monkeypatch, ["Odpowiedź."])
    monkeypatch.setattr(cl, "search_with_history", lambda *a, **k: [])
    history = []
    text = cl.answer("http://x/v1", "qwen", DummyRag(), DummyFiles(), history, "pytanie", 8,
                     rewrite_followups=False, max_tokens=3000, reasoning_effort="low")
    assert text == "Odpowiedź."
    assert calls[0]["max_tokens"] == 3000 and calls[0]["reasoning_effort"] == "low"
    assert [m["role"] for m in history] == ["user", "assistant"]
