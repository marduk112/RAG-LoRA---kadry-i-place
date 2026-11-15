"""
Testy czystej logiki rankingu w rag_search.py (`rank_chunks`/`_version_covers`)
-- bez modelu embeddingowego, bez wczytywania indeksu z dysku. Konstruujemy
małe, ręczne macierze embeddingów i listy metadanych, tak jak `FakeRag` w
tests/test_prompt.py obchodzi prawdziwy RagIndex, tylko tutaj testujemy
PRAWDZIWĄ logikę rankingu/filtrowania, nie atrapę.

Kontekst: PROGRESS.md, "Co dalej" pkt 4 / Krok 18, oraz plan implementacji
w <lokalny plik planu>, faza 2.
"""

import numpy as np
import pytest

from rag_search import (
    _version_covers,
    apply_rerank,
    fuse_rankings,
    fuse_rankings_multi_query,
    merge_candidates,
    rank_chunks,
)


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


class FakeCrossEncoder:
    """Podmiennik CrossEncoder -- zwraca z góry zdefiniowane wyniki per para
    (query, tekst), żeby testować apply_rerank() bez ładowania modelu."""

    def __init__(self, scores_by_text: dict):
        self.scores_by_text = scores_by_text
        self.calls = []

    def predict(self, pairs):
        self.calls.append(pairs)
        return [self.scores_by_text[text] for _, text in pairs]


class TestApplyRerank:
    def test_reorders_by_rerank_score_not_original_cosine_score(self):
        """Regresja dokładnie tego przypadku, który uzasadnił dodanie
        rerankingu: kandydat z NIŻSZYM podobieństwem kosinusowym ('score'),
        ale wyższą oceną rerankera, musi wylądować wyżej w wyniku."""
        results = [
            {"text": "niszowy, leksykalnie bliski", "score": 0.90},
            {"text": "ogólny, trafny przepis", "score": 0.80},
        ]
        cross_encoder = FakeCrossEncoder(
            {"niszowy, leksykalnie bliski": 0.1, "ogólny, trafny przepis": 0.9}
        )
        reranked = apply_rerank("zapytanie", results, cross_encoder)
        assert [r["text"] for r in reranked] == ["ogólny, trafny przepis", "niszowy, leksykalnie bliski"]
        assert reranked[0]["rerank_score"] == 0.9
        assert reranked[0]["score"] == 0.80  # oryginalny cosinus zostaje, tylko przestaje decydować o kolejności

    def test_empty_results_returns_empty_without_calling_model(self):
        cross_encoder = FakeCrossEncoder({})
        assert apply_rerank("zapytanie", [], cross_encoder) == []
        assert cross_encoder.calls == []

    def test_truncates_text_before_reranking(self):
        """RERANK_TEXT_CHARS obcina tekst przed przekazaniem do modelu --
        cross-encodery mają ograniczoną długość wejścia."""
        from rag_search import RERANK_TEXT_CHARS

        long_text = "x" * (RERANK_TEXT_CHARS + 500)
        cross_encoder = FakeCrossEncoder({long_text[:RERANK_TEXT_CHARS]: 1.0})
        results = [{"text": long_text, "score": 0.5}]
        reranked = apply_rerank("zapytanie", results, cross_encoder)
        assert reranked[0]["rerank_score"] == 1.0


class TestApplyRerankMultiQuery:
    """apply_rerank z LISTĄ zapytań (np. oryginalne pytanie + jego
    sformalizowane rozszerzenie) -- patrz RagIndex.search(extra_queries=...)."""

    def test_takes_max_score_across_query_variants(self):
        """Regresja dokładnie tego przypadku, który uzasadnił branie maksimum:
        kandydat trafiony dobrze przez rozszerzone zapytanie, ale słabo przez
        oryginalne, musi wygrać dzięki wyższemu wyniku z rozszerzenia."""
        results = [{"text": "art. 86", "score": 0.8}]

        class MultiScoreCrossEncoder:
            def predict(self, pairs):
                query = pairs[0][0]
                return [0.2] if query == "oryginalne potoczne pytanie" else [0.9]

        reranked = apply_rerank(
            ["oryginalne potoczne pytanie", "sformalizowane rozszerzenie"], results, MultiScoreCrossEncoder()
        )
        assert reranked[0]["rerank_score"] == 0.9

    def test_single_string_query_still_works(self):
        """Kompatybilność wsteczna: apply_rerank(str, ...) (bez listy) --
        dotychczasowe wywołania/testy nie powinny się zepsuć."""
        results = [{"text": "x", "score": 0.5}]
        cross_encoder = FakeCrossEncoder({"x": 0.7})
        reranked = apply_rerank("zapytanie", results, cross_encoder)
        assert reranked[0]["rerank_score"] == 0.7


class TestFuseRankings:
    """RRF (k=0) łącząca porządek kosinusowy i porządek rerankera -- patrz
    PROGRESS.md, Krok 11. Naprawia przypadki, gdzie sam reranker spycha
    krótki, ogólny przepis mimo bardzo wysokiego podobieństwa kosinusowego
    (KC art. 415, KK art. 148), bez psucia przypadków, gdzie to reranker
    poprawia słaby ranking kosinusowy (KP art. 18, KRO art. 56)."""

    def test_empty_results_returns_empty(self):
        assert fuse_rankings([]) == []

    def test_top_in_both_rankings_stays_top(self):
        results = [
            {"text": "a", "score": 0.9, "rerank_score": 0.95},
            {"text": "b", "score": 0.7, "rerank_score": 0.60},
        ]
        fused = fuse_rankings(results)
        assert fused[0]["text"] == "a"
        assert fused[0]["fused_score"] == pytest.approx(1.0 + 1.0)  # #1 w obu -> 1/1 + 1/1

    def test_rescues_candidate_reranker_demotes_despite_top_cosine(self):
        """Regresja dokładnie znaleziska z Kroku 10 (KK art. 148): kandydat
        #1 kosinusowo, ale zepchnięty daleko przez sam reranking, musi
        wrócić na szczyt po fuzji, jeśli konkurenci nie są #1 w obu."""
        results = [
            {"text": "ogólny przepis", "score": 0.90, "rerank_score": 0.10},  # #1 kosinus, ostatni reranking
            {"text": "pochodna 1", "score": 0.85, "rerank_score": 0.95},  # reranking #1
            {"text": "pochodna 2", "score": 0.84, "rerank_score": 0.80},
            {"text": "pochodna 3", "score": 0.83, "rerank_score": 0.70},
        ]
        fused = fuse_rankings(results)
        ranked_texts = [r["text"] for r in fused]
        assert ranked_texts.index("ogólny przepis") < ranked_texts.index("pochodna 3")
        assert ranked_texts.index("ogólny przepis") <= 1  # wraca w okolice szczytu, nie zostaje ostatni

    def test_does_not_disturb_candidate_agreed_as_best_by_both(self):
        results = [
            {"text": "zgodny zwycięzca", "score": 0.95, "rerank_score": 0.98},
            {"text": "reszta", "score": 0.50, "rerank_score": 0.50},
        ]
        fused = fuse_rankings(results)
        assert fused[0]["text"] == "zgodny zwycięzca"


class TestFuseRankingsMultiQuery:
    """`fuse_rankings_multi_query` -- fuzja licząca ranking OSOBNO w obrębie
    każdej listy kandydatów per wariant zapytania, potem maksimum wyniku
    fuzji per kandydat -- patrz PROGRESS.md, Krok 12."""

    def test_empty_results_returns_empty(self):
        class CE:
            def predict(self, pairs):
                return []

        assert fuse_rankings_multi_query(["v1", "v2"], [[], []], CE()) == []

    def test_candidate_not_dragged_down_by_higher_raw_cosine_in_other_variant(self):
        """Regresja dokładnie znaleziska z Kroku 12: kandydat #1 kosinusowo
        i rerankingowo WE WŁASNEJ liście (np. wariant oryginalny) nie może
        stracić pozycji tylko dlatego, że inny kandydat ma WYŻSZY surowy
        kosinus, ale względem INNEGO wariantu (np. rozszerzenia zapytania).
        Zwykłe scalanie-przed-rerankingiem (merge_candidates + fuse_rankings)
        miałoby ten problem -- ta funkcja liczy ranking osobno per wariant,
        więc surowa wartość z cudzej listy nie ma znaczenia."""
        target = {"act_short": "a", "article": "415", "text": "target", "score": 0.85}
        other_in_v1 = {"act_short": "b", "article": "1", "text": "other", "score": 0.10}
        rival = {"act_short": "c", "article": "361", "text": "rival", "score": 0.99}  # wyższy surowy kosinus niż target, ale w INNYM wariancie

        list_v1 = [target, other_in_v1]
        list_v2 = [rival]

        class CE:
            def predict(self, pairs):
                # każdy kandydat jest najlepszy rerankingowo we własnej liście
                return [0.9 if text in ("target", "rival") else 0.1 for _, text in pairs]

        fused = fuse_rankings_multi_query(["oryginalne", "rozszerzenie"], [list_v1, list_v2], CE())
        target_result = next(r for r in fused if r["act_short"] == "a" and r["article"] == "415")
        assert target_result["fused_score"] == pytest.approx(2.0)  # #1 kosinusowo i rerankingowo we WŁASNEJ liście

    def test_candidate_rescued_by_best_variant_even_if_worst_in_another(self):
        """Analogicznie do apply_rerank (bierze maksimum spośród wariantów):
        kandydat kiepsko oceniony w jednym wariancie, ale #1 kosinusowo i
        rerankingowo we własnej liście innego wariantu, musi dostać wynik
        fuzji z TEGO lepszego wariantu, nie z gorszego."""
        target = {"act_short": "a", "article": "86", "text": "target", "score": 0.3}
        other_in_v1 = {"act_short": "b", "article": "1", "text": "other", "score": 0.9}
        target_v2 = {"act_short": "a", "article": "86", "text": "target", "score": 0.95}

        list_v1 = [target, other_in_v1]  # target słaby kosinusowo i rerankingowo w v1
        list_v2 = [target_v2]  # target jedynym (więc #1) kandydatem w v2

        class CE:
            def predict(self, pairs):
                return [0.1 if text == "target" else 0.9 for _, text in pairs]

        fused = fuse_rankings_multi_query(["v1", "v2"], [list_v1, list_v2], CE())
        target_result = next(r for r in fused if r["act_short"] == "a" and r["article"] == "86")
        assert target_result["fused_score"] == pytest.approx(2.0)


class TestMergeCandidates:
    def test_merges_disjoint_lists(self):
        a = [{"act_short": "a", "article": "1", "score": 0.5}]
        b = [{"act_short": "b", "article": "2", "score": 0.6}]
        merged = merge_candidates(a, b)
        assert {(r["act_short"], r["article"]) for r in merged} == {("a", "1"), ("b", "2")}

    def test_duplicate_article_keeps_higher_score(self):
        a = [{"act_short": "a", "article": "1", "score": 0.3, "tag": "from_a"}]
        b = [{"act_short": "a", "article": "1", "score": 0.9, "tag": "from_b"}]
        merged = merge_candidates(a, b)
        assert len(merged) == 1
        assert merged[0]["tag"] == "from_b"

    def test_single_list_passthrough(self):
        a = [{"act_short": "a", "article": "1", "score": 0.5}]
        assert merge_candidates(a) == a
