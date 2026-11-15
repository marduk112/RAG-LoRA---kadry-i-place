import re

from article_excerpt import GAP, article_excerpt, focus_span

ART = (
    "Art. 30. 1. Zgłoszenia dokonuje się organowi. "
    + " ".join(f"{n}. Ustęp numer {n} " + "treść " * 40 + "koniec." for n in range(2, 12))
    + " 12. Organ w terminie 21 dni od dnia doręczenia zgłoszenia może wnieść sprzeciw. 13. Ostatni ustęp."
)


def test_short_text_untouched():
    assert article_excerpt("Art. 1. Krótki.", "Krótki", 6000, 600) == ("Art. 1. Krótki.", False)


def test_window_contains_focus_within_budget():
    focus = "w terminie 21 dni od dnia doręczenia"
    excerpt, shortened = article_excerpt(ART, focus, 1200)
    assert shortened and focus in excerpt and len(excerpt) <= 1200


def test_window_starts_at_unit_boundary():
    start, end = focus_span(ART, "w terminie 21 dni", 1200)
    assert re.match(r"\d+\. [A-Z]", ART[start:]) and end - start <= 1200


def test_head_prepended_when_focus_far_from_start():
    excerpt, _ = article_excerpt(ART, "w terminie 21 dni", 1500, head_chars=100)
    assert excerpt.startswith("Art. 30. 1.") and GAP in excerpt and "21 dni" in excerpt
    assert len(excerpt) <= 1500


def test_unit_boundary_ignores_point_references():
    # "pkt 2. W" to odesłanie, nie początek ustępu.
    text = "Art. 5. 1. Zgodnie z art. 32 ust. 4 pkt 2. W tym przypadku " + "x " * 600 + "2. Drugi ustęp."
    start, _ = focus_span(text, "W tym przypadku", 400)
    assert start == 0


def test_missing_focus_falls_back_to_beginning():
    excerpt, shortened = article_excerpt(ART, "tekstu nie ma w artykule", 800)
    assert shortened and excerpt.startswith("Art. 30.") and len(excerpt) <= 800


def test_single_unit_longer_than_budget_centers_on_focus():
    text = "Art. 21. 1. Wolne od podatku są: " + "; ".join(f"{n}) punkt {n} " + "y " * 30 for n in range(1, 200)) + "."
    focus = "150) punkt 150"
    excerpt, _ = article_excerpt(text, focus, 1000)
    assert focus in excerpt and len(excerpt) <= 1000
