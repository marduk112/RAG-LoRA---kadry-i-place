"""Testy dla scripts/download_acts.py -- ekstrakcji i czyszczenia
tekstu artykułów z PDF-ów. Górna część pliku to testy jednostkowe na
małych, ręcznie skonstruowanych fragmentach tekstu (bez zależności od
pobranych danych). Testy na dole (oznaczone `skipif`) to regresje
konkretnych błędów znalezionych w audycie (PROGRESS.md, krok 12/13) --
wymagają lokalnie pobranej bazy (`data/raw/`, `data/processed/`),
patrz README.md, sekcja "Odtworzenie pipeline'u".
"""

import json
from pathlib import Path

import pytest

from download_acts import (
    ARTICLE_SPLIT_RE,
    fix_stray_space_before_period,
    parse_articles,
    recover_midtext_superscript_headers,
    strip_not_yet_in_force_text,
)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


# --- ARTICLE_SPLIT_RE / parse_articles ---


def test_article_split_re_handles_polish_letter_suffix():
    """Regresja: '[a-z]{0,3}' był tylko ASCII, więc np. 'Art. 22ł.'
    (litera 'ł') nigdy nie pasował -- patrz PROGRESS.md, krok 12,
    znalezisko 4."""
    m = ARTICLE_SPLIT_RE.match("Art. 22ł. Treść artykułu.")
    assert m is not None
    assert m.group(1) == "22ł"


def test_parse_articles_basic_split():
    text = "Preambuła.\nArt. 1. Treść pierwszego.\nArt. 2. Treść drugiego."
    articles = parse_articles(text)
    assert [a["article"] for a in articles] == ["1", "2"]
    assert articles[0]["text"].startswith("Art. 1.")


# --- strip_not_yet_in_force_text ---


def test_strip_not_yet_in_force_removes_angle_bracket_span():
    text = "Przed. <Art. 5. Nieobowiązujący tekst.> Po."
    result = strip_not_yet_in_force_text(text)
    assert "Nieobowiązujący" not in result
    assert "Przed." in result and "Po." in result


def test_strip_not_yet_in_force_keeps_square_bracket_content():
    text = "Wciąż obowiązuje: [stary ale ważny tekst]."
    result = strip_not_yet_in_force_text(text)
    assert "stary ale ważny tekst" in result
    assert "[" not in result and "]" not in result


def test_strip_not_yet_in_force_unmatched_bracket_left_untouched(capsys):
    """Bezpiecznik: niesparowany '<' (bez ryzyka pochłonięcia
    poprawnego tekstu) zostaje, zamiast go zjeść -- z ostrzeżeniem."""
    text = "Tekst z niesparowanym < nawiasem bez zamknięcia."
    result = strip_not_yet_in_force_text(text)
    assert "niesparowanym" in result
    assert "UWAGA" in capsys.readouterr().out


# --- recover_midtext_superscript_headers ---


def test_recover_midtext_superscript_headers_merges_split_number():
    text = "koniec zdania. Art. 22 3. Treść nowego artykułu."
    result = recover_midtext_superscript_headers(text)
    assert "Art. 223." in result
    assert "Art. 22 3." not in result


# --- fix_stray_space_before_period ---


def test_fix_stray_space_before_period():
    text = "Art. 52zb . Treść artykułu."
    result = fix_stray_space_before_period(text)
    assert "Art. 52zb." in result
    assert "Art. 52zb ." not in result


# --- Regresje realnych błędów z audytu (PROGRESS.md, krok 12/13) ---
# Wymagają lokalnej bazy (data/raw/, data/processed/) -- pomijane, jeśli
# jej nie ma (np. świeży checkout bez uruchomionego download_acts.py).


@pytest.mark.skipif(
    not (RAW_DIR / "kodeks_pracy.pdf").exists(),
    reason="wymaga data/raw/kodeks_pracy.pdf (README, sekcja 'Odtworzenie pipeline'u', krok 2)",
)
def test_find_article_numbers_matches_parsed_count_for_kodeks_pracy():
    from download_acts import find_article_numbers_pdfplumber, pdf_to_clean_text

    pdf_bytes = (RAW_DIR / "kodeks_pracy.pdf").read_bytes()
    clean_text = pdf_to_clean_text(pdf_bytes)
    clean_text = strip_not_yet_in_force_text(clean_text)
    clean_text = recover_midtext_superscript_headers(clean_text)
    clean_text = fix_stray_space_before_period(clean_text)
    articles = parse_articles(clean_text)
    detected = find_article_numbers_pdfplumber(pdf_bytes)
    assert len(detected) == len(articles)
    assert "11¹" in detected


@pytest.mark.skipif(
    not (PROCESSED_DIR / "all_articles.json").exists(),
    reason="wymaga data/processed/all_articles.json (README, sekcja 'Odtworzenie pipeline'u', krok 2)",
)
class TestProcessedArticles:
    @staticmethod
    def _load():
        return json.loads((PROCESSED_DIR / "all_articles.json").read_text(encoding="utf-8"))

    def test_no_duplicate_article_labels_in_kodeks_pracy(self):
        data = self._load()
        kp = [a for a in data if a["act_title"].startswith("Ustawa z dnia 26 czerwca 1974")]
        labels = [a["article"] for a in kp]
        assert len(labels) == len(set(labels)), "zdublowane numery artykułów -- regresja naprawy indeksu górnego"

    def test_art_223_present_not_merged_into_222(self):
        data = self._load()
        kp = [a for a in data if a["act_title"].startswith("Ustawa z dnia 26 czerwca 1974")]
        assert "22³" in {a["article"] for a in kp}

    def test_not_yet_in_force_articles_excluded_from_social_insurance_act(self):
        data = self._load()
        act = [a for a in data if a["act_title"].startswith("Ustawa z dnia 13 października 1998")]
        numbers = {a["article"] for a in act}
        for n in ("85c", "85d", "85e", "85f", "85g", "85h", "85i", "85j"):
            assert n not in numbers, f"art. {n} (jeszcze nieobowiązujący) nie powinien być w bazie"

    def test_art_52zb_present_in_pit(self):
        data = self._load()
        pit = [a for a in data if a["act_title"].startswith("Ustawa z dnia 26 lipca 1991")]
        assert "52zb" in {a["article"] for a in pit}
