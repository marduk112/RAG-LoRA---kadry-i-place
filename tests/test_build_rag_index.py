"""
Testy dla scripts/build_rag_index.py -- głównie `load_articles_with_history()`
i `_entry_into_force()`, czyli logikę wersjonowania czasowego dodaną w ramach
funkcji "as-of" (patrz PROGRESS.md, Krok 18 i plan implementacji).

Nie testujemy tu samego liczenia embeddingów (`main()`) -- to wymaga
prawdziwego modelu i jest pokryte ręczną weryfikacją end-to-end opisaną w
PROGRESS.md, nie testem jednostkowym.
"""
import json

import build_rag_index as bri


def _write_meta(raw_dir, short, entry_into_force):
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{short}_meta.json").write_text(
        json.dumps({"entryIntoForce": entry_into_force}), encoding="utf-8"
    )


def _write_all_articles(processed_dir, articles):
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "all_articles.json").write_text(json.dumps(articles), encoding="utf-8")


def _write_history(processed_dir, short, current_valid_from, articles):
    (processed_dir / f"{short}_history.json").write_text(
        json.dumps({"current_valid_from": current_valid_from, "articles": articles}), encoding="utf-8"
    )


def test_include_history_false_returns_articles_unchanged(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    articles = [{"id": "a_art_1", "act_short": "a", "text": "..."}]
    _write_all_articles(processed_dir, articles)
    monkeypatch.setattr(bri, "RAW_DIR", raw_dir)
    monkeypatch.setattr(bri, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(bri, "ARTICLES_PATH", processed_dir / "all_articles.json")

    result = bri.load_articles_with_history(include_history=False)

    assert result == articles


def test_act_without_history_file_floored_at_entry_into_force(tmp_path, monkeypatch):
    """Regresja realnego błędu znalezionego przy ręcznej weryfikacji Fazy 2:
    ustawa bez pliku _history.json (np. bo jeszcze nie uruchomiono
    download_acts_history.py dla niej), ale z entryIntoForce w przyszłości
    względem starego zapytania as_of, wcześniej zostawała z valid_from=None
    i błędnie "obowiązywała" dla dowolnie starej daty (np. ustawa o rynku
    pracy, w mocy dopiero od 2025-06-01, błędnie zwracana dla as_of=2010)."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _write_meta(raw_dir, "mloda_ustawa", "2025-06-01")
    articles = [{"id": "mloda_ustawa_art_1", "act_short": "mloda_ustawa", "text": "..."}]
    _write_all_articles(processed_dir, articles)
    monkeypatch.setattr(bri, "RAW_DIR", raw_dir)
    monkeypatch.setattr(bri, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(bri, "ARTICLES_PATH", processed_dir / "all_articles.json")

    result = bri.load_articles_with_history(include_history=True)

    assert result[0]["valid_from"] == "2025-06-01"


def test_act_with_history_file_uses_max_of_current_valid_from_and_entry_into_force(tmp_path, monkeypatch):
    """Zwykły przypadek: current_valid_from z _history.json (późniejszy niż
    entryIntoForce) wygrywa -- max() to tylko bezpiecznik, nie zmienia
    zwykłego zachowania z Fazy 2."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _write_meta(raw_dir, "a", "2003-01-01")
    articles = [{"id": "a_art_1", "act_short": "a", "text": "..."}]
    _write_all_articles(processed_dir, articles)
    _write_history(processed_dir, "a", "2024-11-27", [])
    monkeypatch.setattr(bri, "RAW_DIR", raw_dir)
    monkeypatch.setattr(bri, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(bri, "ARTICLES_PATH", processed_dir / "all_articles.json")

    result = bri.load_articles_with_history(include_history=True)

    assert result[0]["valid_from"] == "2024-11-27"


def test_missing_meta_json_leaves_valid_from_none(tmp_path, monkeypatch):
    """Brak data/raw/{short}_meta.json (np. świeży klon repo bez lokalnie
    pobranej bazy) nie wywala się -- po prostu brak dolnej granicy."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    articles = [{"id": "a_art_1", "act_short": "a", "text": "..."}]
    _write_all_articles(processed_dir, articles)
    monkeypatch.setattr(bri, "RAW_DIR", raw_dir)
    monkeypatch.setattr(bri, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(bri, "ARTICLES_PATH", processed_dir / "all_articles.json")

    result = bri.load_articles_with_history(include_history=True)

    assert result[0]["valid_from"] is None


def test_history_articles_appended(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    _write_meta(raw_dir, "a", "2003-01-01")
    _write_all_articles(processed_dir, [{"id": "a_art_1", "act_short": "a", "text": "obecny"}])
    _write_history(
        processed_dir,
        "a",
        "2024-11-27",
        [{"id": "a_art_1@2015-11-09", "act_short": "a", "text": "stary", "valid_from": "2015-11-09", "valid_to": "2017-04-26"}],
    )
    monkeypatch.setattr(bri, "RAW_DIR", raw_dir)
    monkeypatch.setattr(bri, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(bri, "ARTICLES_PATH", processed_dir / "all_articles.json")

    result = bri.load_articles_with_history(include_history=True)

    assert len(result) == 2
    assert {a["id"] for a in result} == {"a_art_1", "a_art_1@2015-11-09"}


def test_build_chunks_omits_version_fields_when_absent():
    """Bajtowa identyczność domyślnej ścieżki: artykuł bez pól wersji nie
    dostaje ich w chunku (żadnych kluczy o wartości null)."""
    articles = [{"id": "a_art_1", "act_short": "a", "act_title": "A", "eli": "DU/1/1", "article": "1", "text": "krótki", "source_url": "http://x"}]
    chunks = bri.build_chunks(articles)
    assert "valid_from" not in chunks[0]
    assert "valid_to" not in chunks[0]
    assert "announcement_eli" not in chunks[0]


def test_build_chunks_includes_version_fields_when_present():
    articles = [
        {
            "id": "a_art_1@2015-11-09",
            "act_short": "a",
            "act_title": "A",
            "eli": "DU/1/1",
            "article": "1",
            "text": "krótki",
            "source_url": "http://x",
            "valid_from": "2015-11-09",
            "valid_to": "2017-04-26",
            "announcement_eli": "DU/2015/2008",
        }
    ]
    chunks = bri.build_chunks(articles)
    assert chunks[0]["valid_from"] == "2015-11-09"
    assert chunks[0]["valid_to"] == "2017-04-26"
    assert chunks[0]["announcement_eli"] == "DU/2015/2008"
