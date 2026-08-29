"""
Testy dla scripts/download_wage_regulations.py -- pełny scenariusz (żywe
zapytania do ELI API) sprawdzany ręcznie (patrz PROGRESS.md, Krok 21), tu
tylko czyste funkcje bez I/O.
"""
import download_wage_regulations as dwr


def test_extract_year_from_title():
    assert dwr.extract_year({"title": "... w 2018 r."}) == 2018


def test_extract_year_missing_entry_into_force_still_works():
    """Regresja realnego przypadku: dwa najstarsze wpisy (2009, 2010) są
    typu "Obwieszczenie" i nie mają pola entryIntoForce w ogóle -- rok
    musi i tak zostać poprawnie wyciągnięty z tytułu."""
    meta = {
        "title": "Obwieszczenie Prezesa Rady Ministrów [...] w 2010 r.",
        "entryIntoForce": None,
    }
    assert dwr.extract_year(meta) == 2010


def test_extract_year_no_match_returns_none():
    assert dwr.extract_year({"title": "Zupełnie inny tytuł bez roku"}) is None


def test_extract_year_consistent_with_entry_into_force_no_warning(capsys):
    dwr.extract_year({"title": "... w 2025 r.", "entryIntoForce": "2025-01-01"})
    assert "UWAGA" not in capsys.readouterr().out


def test_extract_year_inconsistent_with_entry_into_force_warns(capsys):
    dwr.extract_year({"title": "... w 2025 r.", "entryIntoForce": "2024-01-01", "ELI": "DU/1/1"})
    assert "UWAGA" in capsys.readouterr().out


def test_warn_on_gaps_contiguous_no_warning(capsys):
    dwr._warn_on_gaps([{"year": y} for y in range(2020, 2024)])
    assert "UWAGA" not in capsys.readouterr().out


def test_warn_on_gaps_missing_year_warns(capsys):
    """Regresja realnej luki znalezionej w Kroku 21: rok 2021 brakował w
    references['Akty wykonawcze'], mimo że akt (DU/2020/1596) istnieje w
    ELI -- ten sam typ błędu musi zostać wykryty, gdyby się powtórzył."""
    dwr._warn_on_gaps([{"year": y} for y in [2019, 2020, 2022, 2023]])
    out = capsys.readouterr().out
    assert "UWAGA" in out
    assert "2021" in out
