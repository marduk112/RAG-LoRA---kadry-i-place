"""Testy dla scripts/file_index.py -- tymczasowego, sesyjnego indeksu
dla wgranego pliku PDF (patrz README, sekcja o uploadzie plików).
PdfReader jest mockowany (zwraca z góry ustalony tekst stron), więc
testy nie potrzebują prawdziwego pliku PDF ani ciężkiego modelu
embeddingowego -- FakeModel zwraca deterministyczne wektory.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from file_index import SessionFileIndex


class FakeModel:
    """Deterministyczny podmiennik SentenceTransformer -- wektor 2D
    zależny tylko od długości tekstu, wystarczający do testowania
    logiki dodawania/wyszukiwania fragmentów, bez ładowania prawdziwego
    modelu embeddingowego."""

    def encode(self, texts, normalize_embeddings=True):
        return np.array([[len(t) % 7, 1.0] for t in texts], dtype=np.float32)


def _mock_reader(monkeypatch, page_texts: list[str]):
    pages = []
    for text in page_texts:
        page = MagicMock()
        page.extract_text.return_value = text
        pages.append(page)
    reader = MagicMock()
    reader.pages = pages
    monkeypatch.setattr("file_index.PdfReader", lambda path: reader)


def test_add_pdf_extracts_and_indexes_text(monkeypatch):
    _mock_reader(monkeypatch, ["Okres wypowiedzenia wynosi trzy miesiące."])
    idx = SessionFileIndex(model=FakeModel())

    n = idx.add_pdf("umowa.pdf")

    assert n == 1
    assert idx.file_name == "umowa.pdf"
    assert idx.embeddings is not None
    assert idx.embeddings.shape[0] == 1


def test_add_pdf_raises_on_empty_text_scan(monkeypatch):
    """PDF-y ze skanem/obrazem nie mają wyciąganego tekstu -- powinniśmy
    to jasno zgłosić błędem, nie ciszej indeksować pusty fragment."""
    _mock_reader(monkeypatch, [""])
    idx = SessionFileIndex(model=FakeModel())

    with pytest.raises(ValueError, match="skan"):
        idx.add_pdf("skan.pdf")


def test_add_pdf_replaces_previous_file(monkeypatch):
    _mock_reader(monkeypatch, ["Pierwszy plik."])
    idx = SessionFileIndex(model=FakeModel())
    idx.add_pdf("pierwszy.pdf")

    _mock_reader(monkeypatch, ["Drugi plik, inna treść."])
    idx.add_pdf("drugi.pdf")

    assert idx.file_name == "drugi.pdf"
    assert len(idx.chunks) == 1
    assert idx.chunks[0]["chunk_text"] == "Drugi plik, inna treść."


def test_search_before_any_file_returns_empty():
    idx = SessionFileIndex(model=FakeModel())
    assert idx.search("cokolwiek") == []


def test_search_returns_source_file_and_text(monkeypatch):
    _mock_reader(monkeypatch, ["Treść umowy o pracę."])
    idx = SessionFileIndex(model=FakeModel())
    idx.add_pdf("umowa.pdf")

    results = idx.search("okres wypowiedzenia", top_k=3)

    assert len(results) == 1
    assert results[0]["source_file"] == "umowa.pdf"
    assert results[0]["text"] == "Treść umowy o pracę."
