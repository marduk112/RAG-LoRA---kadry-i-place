"""
Testy dla scripts/download_avg_wage_regulations.py -- pełny scenariusz
(żywe zapytania do ELI API) sprawdzany ręcznie (patrz PROGRESS.md, Krok 23),
tu tylko czyste funkcje bez I/O.
"""
import download_avg_wage_regulations as dawr


def test_extract_year_matches_exact_annual_title():
    title = "Komunikat Prezesa Głównego Urzędu Statystycznego z dnia 9 lutego 2023 r. w sprawie przeciętnego wynagrodzenia w gospodarce narodowej w 2022 r."
    assert dawr.extract_year({"title": title}) == 2022


def test_extract_year_rejects_semiannual_variant():
    """Regresja realnego ryzyka pomyłki znalezionego w Kroku 23: wariant
    'miesięcznego [...] i w drugim półroczu' NIE może zostać zaakceptowany
    -- to inna wielkość, inne zastosowanie prawne."""
    title = (
        "Obwieszczenie Prezesa Głównego Urzędu Statystycznego z dnia 19 lutego 2025 r. "
        "w sprawie przeciętnego wynagrodzenia miesięcznego w gospodarce narodowej w 2024 r. "
        "i w drugim półroczu 2024 r."
    )
    assert dawr.extract_year({"title": title}) is None


def test_extract_year_rejects_voivodeship_variant():
    """Regresja: wariant wojewódzki też musi zostać odrzucony."""
    title = (
        "Obwieszczenie Prezesa Głównego Urzędu Statystycznego z dnia 19 listopada 2025 r. "
        "w sprawie przeciętnego miesięcznego wynagrodzenia brutto w gospodarce narodowej "
        "w województwach w 2024 r."
    )
    assert dawr.extract_year({"title": title}) is None


def test_extract_year_rejects_unrelated_title():
    assert dawr.extract_year({"title": "Zupełnie inny tytuł"}) is None


def test_warn_on_gaps_contiguous_no_warning(capsys):
    dawr._warn_on_gaps([{"year": y} for y in range(2010, 2015)])
    assert "UWAGA" not in capsys.readouterr().out


def test_warn_on_gaps_missing_year_warns(capsys):
    dawr._warn_on_gaps([{"year": y} for y in [2010, 2011, 2013]])
    out = capsys.readouterr().out
    assert "UWAGA" in out
    assert "2012" in out
