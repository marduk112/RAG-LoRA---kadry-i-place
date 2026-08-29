"""
Testy dla scripts/download_zus_limit_regulations.py -- pełny scenariusz
(żywe zapytania do ELI API) sprawdzany ręcznie (patrz PROGRESS.md, Krok 22),
tu tylko czyste funkcje bez I/O.
"""
import download_zus_limit_regulations as dzlr


def test_extract_year_mid_title():
    """Wzorzec różni się od tego w download_wage_regulations.py: rok bywa
    w ŚRODKU tytułu ("w roku 2025 oraz przyjętej [...]"), nie na końcu."""
    title = "Obwieszczenie [...] w sprawie kwoty ograniczenia [...] w roku 2025 oraz przyjętej do jej ustalenia [...]"
    assert dzlr.extract_year({"title": title}) == 2025


def test_extract_year_at_end_of_title():
    """Starsze wpisy (1999-2007) kończą się na roku, bez dalszej klauzuli."""
    assert dzlr.extract_year({"title": "[...] w roku 2008"}) == 2008


def test_extract_year_no_match_returns_none():
    assert dzlr.extract_year({"title": "Zupełnie inny tytuł"}) is None


def test_warn_on_gaps_contiguous_no_warning(capsys):
    dzlr._warn_on_gaps([{"year": y} for y in range(2010, 2015)])
    assert "UWAGA" not in capsys.readouterr().out


def test_warn_on_gaps_missing_year_warns(capsys):
    dzlr._warn_on_gaps([{"year": y} for y in [2010, 2011, 2013]])
    out = capsys.readouterr().out
    assert "UWAGA" in out
    assert "2012" in out
