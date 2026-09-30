"""Testy scripts/embedding_cache.py -- przyrostowe embeddingi z cache'em
(projekt "RAG-LoRA - prawo polskie", PROGRESS.md Krok 63). Atrapa modelu zamiast SentenceTransformer: liczy,
ile tekstów faktycznie "zakodowano", żeby sprawdzić, że cache omija już
policzone fragmenty i że wynik jest identyczny jak bez cache'u."""

import numpy as np
import pytest

import embedding_cache as ec


class FakeModel:
    def __init__(self):
        self.encoded: list[str] = []

    def encode(self, texts, batch_size, show_progress_bar, normalize_embeddings):
        self.encoded.extend(texts)
        vecs = np.array([[len(t), sum(map(ord, t)) % 97, 1.0] for t in texts], dtype=np.float32)
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


@pytest.fixture
def cache(tmp_path):
    c = ec.EmbeddingCache(tmp_path / "cache.sqlite", "model|tryb")
    yield c
    c.close()


def test_second_run_uses_cache_and_gives_identical_result(cache):
    texts = ["Art. 1. Treść.", "Art. 2. Inna treść.", "Art. 1. Treść."]  # duplikat liczony raz
    first_model = FakeModel()
    first = ec.encode_with_cache(first_model, texts, cache)
    assert first_model.encoded == ["Art. 1. Treść.", "Art. 2. Inna treść."]

    second_model = FakeModel()
    second = ec.encode_with_cache(second_model, texts, cache)
    assert second_model.encoded == []
    np.testing.assert_array_equal(first, second)


def test_only_changed_text_is_recomputed(cache):
    ec.encode_with_cache(FakeModel(), ["A", "B"], cache)
    model = FakeModel()
    out = ec.encode_with_cache(model, ["A", "B zmienione"], cache)
    assert model.encoded == ["B zmienione"]
    np.testing.assert_array_equal(out, ec.encode_with_cache(FakeModel(), ["A", "B zmienione"], None))


def test_different_model_key_does_not_share_entries(tmp_path):
    a = ec.EmbeddingCache(tmp_path / "c.sqlite", "model-A")
    ec.encode_with_cache(FakeModel(), ["X"], a)
    a.close()
    b = ec.EmbeddingCache(tmp_path / "c.sqlite", "model-B")
    model = FakeModel()
    ec.encode_with_cache(model, ["X"], b)
    b.close()
    assert model.encoded == ["X"]


def test_low_memory_reduces_batch_and_critical_memory_aborts(monkeypatch):
    monkeypatch.setattr(ec, "available_ram_gb", lambda: ec.LOW_MEMORY_GB - 1)
    assert ec._batch_size_for_free_memory() == ec.LOW_MEMORY_BATCH_SIZE
    monkeypatch.setattr(ec, "available_ram_gb", lambda: ec.MIN_FREE_MEMORY_GB - 1)
    monkeypatch.setattr(ec.time, "sleep", lambda s: None)
    with pytest.raises(SystemExit):
        ec._batch_size_for_free_memory()


def test_empty_input(cache):
    assert ec.encode_with_cache(FakeModel(), [], cache).shape == (0, 0)
