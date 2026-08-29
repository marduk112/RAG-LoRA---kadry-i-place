"""
Testy czystej logiki rankingu w rag_search.py (`rank_chunks`/`_version_covers`)
-- bez modelu embeddingowego, bez wczytywania indeksu z dysku. Konstruujemy
małe, ręczne macierze embeddingów i listy metadanych, tak jak `FakeRag` w
tests/test_prompt.py obchodzi prawdziwy RagIndex, tylko tutaj testujemy
PRAWDZIWĄ logikę rankingu/filtrowania, nie atrapę.

Kontekst: PROGRESS.md, "Co dalej" pkt 4 / Krok 18, oraz plan implementacji
w /Users/szymon/.claude/plans/drifting-toasting-wozniak.md, faza 2.
"""

import numpy as np
import pytest

from rag_search import _version_covers, rank_chunks


def _chunk(act_short="kodeks_pracy", article="1", valid_from=None, valid_to=None, chunk_id=None, **extra):
    d = {
        "chunk_id": chunk_id or f"{act_short}_art_{article}@{valid_from or 'current'}",
        "act_short": act_short,
        "act_title": "Testowa ustawa",
        "eli": "DU/2000/1",
        "article": article,
        "source_url": "http://example.test",
        "chunk_text": "treść",
        "full_article_text": "treść pełna",
        "valid_from": valid_from,
        "valid_to": valid_to,
        "announcement_eli": None,
    }
    d.update(extra)
    return d


def _embeddings_for(scores: list[float]) -> np.ndarray:
    """Zwraca macierz embeddingów i query_vec takie, że embeddings @ query_vec
    == scores (jednowymiarowe wektory, iloczyn skalarny = mnożenie)."""
    return np.array([[s] for s in scores], dtype=np.float32)


QUERY_VEC = np.array([1.0], dtype=np.float32)


class TestVersionCovers:
    def test_no_date_fields_always_covers(self):
        assert _version_covers(_chunk(), "2019-01-01") is True
        assert _version_covers(_chunk(), "1900-01-01") is True

    def test_valid_from_inclusive_lower_bound(self):
        chunk = _chunk(valid_from="2020-01-01", valid_to=None)
        assert _version_covers(chunk, "2020-01-01") is True  # granica dolna: inclusive
        assert _version_covers(chunk, "2019-12-31") is False

    def test_valid_to_exclusive_upper_bound(self):
        chunk = _chunk(valid_from=None, valid_to="2020-01-01")
        assert _version_covers(chunk, "2019-12-31") is True
        assert _version_covers(chunk, "2020-01-01") is False  # granica górna: exclusive

    def test_window_middle(self):
        chunk = _chunk(valid_from="2015-01-01", valid_to="2020-01-01")
        assert _version_covers(chunk, "2017-06-15") is True
        assert _version_covers(chunk, "2010-01-01") is False
        assert _version_covers(chunk, "2025-01-01") is False


class TestRankChunksRegression:
    """as_of=None na danych bez żadnych pól dat -- musi dać identyczny wynik
    co dotychczasowa (sprzed tej funkcji) logika grupowania/rankingu."""

    def test_as_of_none_no_date_fields_matches_legacy_behavior(self):
        meta = [
            _chunk(act_short="a", article="1", chunk_id="a1"),
            _chunk(act_short="a", article="2", chunk_id="a2"),
            _chunk(act_short="b", article="1", chunk_id="b1"),
        ]
        embeddings = _embeddings_for([0.9, 0.5, 0.7])
        results = rank_chunks(embeddings, meta, QUERY_VEC, top_k=5, as_of=None)
        assert [r["article"] for r in results] == ["1", "1", "2"]  # posortowane malejąco po score
        assert [r["act_short"] for r in results] == ["a", "b", "a"]
        assert results[0]["score"] == pytest.approx(0.9)

    def test_multi_chunk_article_dedups_to_best(self):
        meta = [
            _chunk(act_short="a", article="1", chunk_id="a1_part1"),
            _chunk(act_short="a", article="1", chunk_id="a1_part2"),
        ]
        embeddings = _embeddings_for([0.3, 0.8])
        results = rank_chunks(embeddings, meta, QUERY_VEC, top_k=5, as_of=None)
        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(0.8)


class TestRankChunksVersioning:
    def test_as_of_inside_older_window_picks_only_that_version(self):
        meta = [
            _chunk(act_short="a", article="1", chunk_id="old", valid_from="2010-01-01", valid_to="2020-01-01"),
            _chunk(act_short="a", article="1", chunk_id="new", valid_from="2020-01-01", valid_to=None),
        ]
        embeddings = _embeddings_for([0.5, 0.5])
        results = rank_chunks(embeddings, meta, QUERY_VEC, top_k=5, as_of="2015-06-01")
        assert len(results) == 1
        assert results[0]["valid_from"] == "2010-01-01"

    def test_unversioned_act_unaffected_by_other_acts_history(self):
        meta = [
            _chunk(act_short="versioned", article="1", chunk_id="v1", valid_from="2010-01-01", valid_to="2020-01-01"),
            _chunk(act_short="plain", article="1", chunk_id="p1"),  # brak pól dat w ogóle
        ]
        embeddings = _embeddings_for([0.9, 0.6])
        results = rank_chunks(embeddings, meta, QUERY_VEC, top_k=5, as_of="2015-01-01")
        act_shorts = {r["act_short"] for r in results}
        assert "plain" in act_shorts  # nie wykluczony mimo że inne ustawy w indeksie mają historię

    def test_as_of_before_earliest_known_window_excludes_act(self):
        meta = [
            _chunk(act_short="a", article="1", chunk_id="only", valid_from="2015-01-01", valid_to="2020-01-01"),
        ]
        embeddings = _embeddings_for([0.9])
        results = rank_chunks(embeddings, meta, QUERY_VEC, top_k=5, as_of="2005-01-01")
        assert results == []

    def test_as_of_none_current_version_wins_over_higher_scoring_stale_chunk(self):
        """Kluczowy test regresji: nawet gdy przestarzała wersja historyczna ma
        WYŻSZY score niż bieżąca, as_of=None (== 'dziś') musi zwrócić bieżącą,
        nie przestarzałą -- inaczej po --include-history zwykłe pytanie bez
        podanej daty mogłoby po cichu zwrócić złe, historyczne prawo."""
        meta = [
            _chunk(act_short="a", article="1", chunk_id="stale", valid_from="2010-01-01", valid_to="2015-01-01"),
            _chunk(act_short="a", article="1", chunk_id="current", valid_from="2020-01-01", valid_to=None),
        ]
        embeddings = _embeddings_for([0.95, 0.40])  # przestarzała wersja ma WYŻSZY score
        results = rank_chunks(embeddings, meta, QUERY_VEC, top_k=5, as_of=None)
        assert len(results) == 1
        assert results[0]["valid_from"] == "2020-01-01"  # bieżąca, nie przestarzała
