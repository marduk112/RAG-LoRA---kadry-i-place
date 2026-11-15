"""Wycinek długiego artykułu wokół fragmentu trafionego przez wyszukiwanie
(PROGRESS.md, Krok 77).

Indeks RAG tnie długie artykuły na okna po ~900 znaków (build_rag_index.py),
więc etap kosinusowy trafia we właściwy ustęp -- ale reranker dostawał
pierwsze RERANK_TEXT_CHARS znaków artykułu, a kontekst modelu pierwsze
MAX_ARTICLE_CHARS (prompt.build_context). Przy 261 artykułach > 6000 znaków
(art. 21 ustawy o PIT, art. 30 Prawa budowlanego -- ust. 5 zaczyna się na
znaku ~7600) model nie widział trafionego ustępu. Tu: okno wokół trafionego
fragmentu, poszerzane o całe ustępy (paragrafy), plus początek artykułu do
orientacji. Czyste funkcje, bez zależności od modeli.
"""

import re

# Początek ustępu ("2a. Do zgłoszenia") albo paragrafu ("§ 3. Jeżeli") --
# tylko po końcu zdania/punktu, żeby "art. 32 ust. 4 pkt 2. W" nie liczyło się
# jako granica.
UNIT_START_RE = re.compile(r"(?<=[.;)] )(?=(?:§ )?\d+[a-z]*[⁰¹²³⁴⁵⁶⁷⁸⁹]*\. [A-ZĄĆĘŁŃÓŚŹŻ(])")
GAP = " [...] "


def _snap_words(text: str, start: int, end: int) -> tuple[int, int]:
    """Przesuwa granice okna znakowego do spacji (bez ucinania słów)."""
    if start > 0:
        sp = text.find(" ", start)
        start = sp + 1 if 0 <= sp < end else start
    if end < len(text):
        sp = text.rfind(" ", start, end)
        end = sp if sp > start else end
    return start, end


def focus_span(text: str, focus: str | None, max_chars: int) -> tuple[int, int]:
    """Zakres [start, end) najwyżej max_chars znaków obejmujący `focus`
    (fragment z indeksu), poszerzony o całe sąsiednie ustępy, póki się mieszczą.
    Bez `focus` (albo gdy go nie ma w tekście) -- początek artykułu."""
    pos = text.find(focus) if focus else -1
    if pos < 0:
        return _snap_words(text, 0, max_chars)
    f_start, f_end = pos, pos + len(focus)
    if f_end - f_start >= max_chars:
        return _snap_words(text, f_start, f_start + max_chars)
    bounds = [0] + [m.start() for m in UNIT_START_RE.finditer(text)] + [len(text)]
    start = max(b for b in bounds if b <= f_start)
    end = min(b for b in bounds if b >= f_end)
    if end - start > max_chars:
        # Jeden ustęp dłuższy niż budżet (np. art. 21 ust. 1 PIT, ~150 punktów)
        # -- okno znakowe wyśrodkowane na fragmencie.
        pad = (max_chars - (f_end - f_start)) // 2
        start = max(0, min(f_start - pad, len(text) - max_chars))
        return _snap_words(text, start, start + max_chars)
    before = [b for b in bounds if b < start][::-1]
    after = [b for b in bounds if b > end]
    while before or after:
        grown = False
        if after and after[0] - start <= max_chars:
            end = after.pop(0)
            grown = True
        if before and end - before[0] <= max_chars:
            start = before.pop(0)
            grown = True
        if not grown:
            break
    return start, end


def article_excerpt(text: str, focus: str | None, max_chars: int, head_chars: int = 0) -> tuple[str, bool]:
    """(wycinek, czy_skrócony). Krótszy niż max_chars -- cały tekst. Inaczej
    okno wokół `focus` (focus_span) w budżecie max_chars, a gdy okno nie
    zaczyna się od początku artykułu -- najpierw do head_chars znaków początku
    (nagłówek "Art. N. 1. ..."), potem " [...] "."""
    if len(text) <= max_chars:
        return text, False
    start, end = focus_span(text, focus, max_chars)
    if start > 0 and head_chars:
        start, end = focus_span(text, focus, max_chars - head_chars - len(GAP))
        if start > head_chars:
            _, head_end = _snap_words(text, 0, head_chars)
            return text[:head_end].strip() + GAP + text[start:end].strip(), True
    return text[start:end].strip(), True
