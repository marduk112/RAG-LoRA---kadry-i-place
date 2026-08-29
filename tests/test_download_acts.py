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
    compute_version_windows,
    fix_stray_space_before_period,
    parse_articles,
    process_act_version,
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


# --- compute_version_windows (wersjonowanie czasowe, "as-of", Krok 18) ---
# Testy czystej funkcji (bez HTTP) na ręcznie skonstruowanych metadanych --
# atrapy fragmentów prawdziwych odpowiedzi ELI API zebranych w spike'u
# walidacyjnym (PROGRESS.md, Krok 18).


def _fake_announcement_meta(publisher, year, position, legal_status_date, expiration_date, announcement_date=None):
    return {
        "publisher": publisher,
        "year": year,
        "pos": position,
        "legalStatusDate": legal_status_date,
        "expirationDate": expiration_date,
        "announcementDate": announcement_date,
        "ELI": f"{publisher}/{year}/{position}",
    }


def test_compute_version_windows_normal_chronological_chain():
    announcements = [
        _fake_announcement_meta("DU", 2020, 2207, "2020-11-12", "2024-12-03"),
        _fake_announcement_meta("DU", 2018, 2177, "2018-11-08", "2020-12-10"),
        _fake_announcement_meta("DU", 2024, 1773, "2024-11-27", None),
    ]
    windows = compute_version_windows({}, announcements)
    past = [w for w in windows if not w.get("is_current")]
    # DU/2024/1773 (expirationDate=None, wciąż oficjalnie obowiązujące) NIE
    # trafia do "past" -- opisywałoby ten sam, wciąż otwarty okres co wpis
    # "is_current" (regresja realnego przypadku znalezionego przy smoke
    # teście na ustawie o PPK, patrz docstring compute_version_windows).
    assert [w["valid_from"] for w in past] == ["2018-11-08", "2020-11-12"]
    current = next(w for w in windows if w.get("is_current"))
    assert current["valid_from"] == "2024-11-27"  # najnowsze wciąż otwarte (expirationDate=None) -> legalStatusDate
    assert current["valid_to"] is None


def test_compute_version_windows_sorts_by_announcement_date_when_legal_status_missing():
    """Regresja znaleziska 1 z Kroku 18: dwa najstarsze wpisy bez
    legalStatusDate, w API zwrócone w BŁĘDNEJ (odwrotnej) kolejności
    chronologicznej -- musimy sięgnąć po announcementDate (a NIE
    expirationDate, który dla starego, długo obowiązującego wpisu potrafi
    wypaść PO legalStatusDate kolejnego wpisu i fałszywie go wyprzedzić --
    sprawdzone na tych samych, realnych wartościach co w API: DU/2009/1585
    ma expirationDate 2013-12-04, czyli PO legalStatusDate DU/2013/1442
    (2013-10-16), mimo że DU/2009/1585 jest wcześniejszy). Dane -- dokładnie
    to, co zwróciło żywe ELI API dla ustawy systemowej, zweryfikowane przy
    tej poprawce."""
    announcements = [
        _fake_announcement_meta("DU", 2009, 1585, None, "2013-12-04", announcement_date="2009-11-10"),
        _fake_announcement_meta("DU", 2007, 74, None, "2009-11-10", announcement_date="2007-01-08"),
        _fake_announcement_meta("DU", 2013, 1442, "2013-10-16", "2015-01-22", announcement_date="2013-10-24"),
    ]
    windows = compute_version_windows({}, announcements)
    past = [w for w in windows if not w.get("is_current")]
    assert [w["position"] for w in past] == [74, 1585, 1442]


def test_compute_version_windows_latest_still_open():
    """Jedyne obwieszczenie wciąż otwarte (expirationDate=None) -> zero
    wpisów w 'past' (nic nie trzeba pobierać jako historię), a granica
    bieżącej wersji to jego legalStatusDate."""
    announcements = [_fake_announcement_meta("DU", 2024, 1773, "2024-11-27", None)]
    windows = compute_version_windows({}, announcements)
    past = [w for w in windows if not w.get("is_current")]
    assert past == []
    current = next(w for w in windows if w.get("is_current"))
    assert current["valid_from"] == "2024-11-27"


def test_compute_version_windows_latest_already_expired_uses_expiration_date():
    """Najnowsze obwieszczenie ma już swój expirationDate (zostało
    wyprzedzone przez nowelizacje widoczne tylko w żywym tekście 'U') ->
    granica bieżącej wersji to expirationDate, nie legalStatusDate."""
    announcements = [_fake_announcement_meta("DU", 2020, 2207, "2020-11-12", "2024-12-03")]
    windows = compute_version_windows({}, announcements)
    past = [w for w in windows if not w.get("is_current")]
    assert len(past) == 1  # ma expirationDate ustawione -> to prawdziwa, zamknięta przeszła wersja
    current = next(w for w in windows if w.get("is_current"))
    assert current["valid_from"] == "2024-12-03"


def test_compute_version_windows_no_announcements():
    windows = compute_version_windows({}, [])
    assert [w for w in windows if not w.get("is_current")] == []
    current = next(w for w in windows if w.get("is_current"))
    assert current["valid_from"] is None
    assert current["valid_to"] is None


@pytest.mark.skipif(
    not (RAW_DIR / "ustawa_o_ppk_meta.json").exists(),
    reason="wymaga sieci (żywe zapytanie do ELI API) i lokalnej bazy dla porównania -- pomijane offline",
)
def test_process_act_version_round_trip_on_real_announcement():
    """Smoke test end-to-end na jednym, realnym, małym obwieszczeniu (ustawa
    o PPK ma tylko 4 wersje w historii -- najlżejsza z 7 ustaw)."""
    source = {"publisher": "DU", "year": 2020, "position": 1342}
    articles = process_act_version(
        source,
        "ustawa_o_ppk",
        "DU/2018/2215",
        "Ustawa z dnia 4 października 2018 r. o pracowniczych planach kapitałowych",
        "2020-07-06",
        "2023-01-08",
        "DU/2020/1342",
    )
    assert len(articles) > 0
    a = articles[0]
    assert a["act_short"] == "ustawa_o_ppk"
    assert a["eli"] == "DU/2018/2215"  # akt bazowy, NIE obwieszczenie
    assert a["act_title"].startswith("Ustawa z dnia 4 października 2018")  # akt bazowy, NIE tytuł obwieszczenia
    assert a["announcement_eli"] == "DU/2020/1342"
    assert a["valid_from"] == "2020-07-06"
    assert a["valid_to"] == "2023-01-08"
    assert a["id"].endswith("@2020-07-06")


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
