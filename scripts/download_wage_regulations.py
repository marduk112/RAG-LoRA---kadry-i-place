"""
Pobiera coroczne rozporządzenia/obwieszczenia w sprawie wysokości
minimalnego wynagrodzenia za pracę -- czyli faktyczną kwotę w złotych,
której SAMA ustawa o minimalnym wynagrodzeniu (DU/2002/1679) NIE zawiera
(ustala ją co roku osobny akt na podstawie delegacji z art. 2 tej ustawy) --
patrz PROGRESS.md, Krok 20, "Co dalej" pkt 5.

Źródło listy: endpoint `/eli/acts/search?title=...` (pełnotekstowe
wyszukiwanie tytułów aktów), NIE `references["Akty wykonawcze"]` w
metadanych ustawy bazowej -- ta druga ścieżka była pierwszą próbą, ale
okazała się NIEKOMPLETNA: brakowało w niej roku 2021 (realny akt,
DU/2020/1596, istnieje i jest poprawnie zindeksowany w ELI). Przyczyna
znaleziona przy odczycie treści tego aktu: rozporządzenie na 2021 r. zostało
wydane na PODSTAWIE PRAWNEJ INNEJ ustawy ("art. 79 ust. 5 ustawy z dnia 31
marca 2020 r. o zmianie ustawy o szczególnych rozwiązaniach związanych z
zapobieganiem [...] COVID-19 [...]"), nie na podstawie art. 2 ust. 5 ustawy
o minimalnym wynagrodzeniu jak wszystkie pozostałe lata -- covidowa
nowelizacja tymczasowo przeniosła tę samą delegację do innej ustawy, więc
ELI poprawnie NIE zlinkował go jako "Akt wykonawczy" ustawy o minimalnym
wynagrodzeniu (to nie błąd/luka w ELI, tylko wierne odzwierciedlenie
faktycznej podstawy prawnej z tamtego roku). Znalezione i potwierdzone
empirycznie przez ręczne porównanie obu ścieżek (Krok 21). Endpoint `search` po tytule "wysokości minimalnego
wynagrodzenia za pracę" zwraca kompletną, ciągłą serię: 23 wpisy, po jednym
na każdy rok 2004-2026 bez przerwy, w tym dwa najstarsze (2009, 2010) jako
"Obwieszczenie Prezesa Rady Ministrów" zamiast "Rozporządzenie Rady
Ministrów" -- inny mechanizm prawny z tamtego okresu, ale ten sam wzorzec
tytułu i ta sama treść merytoryczna. Elementy zwrócone przez `search` to
już PEŁNE metadane aktu (ten sam kształt co pojedyncze zapytanie
`/eli/acts/{publisher}/{year}/{position}`) -- nie trzeba więc osobnego
zapytania o metadane każdego kandydata.

Struktura każdego aktu (sprawdzona na DU/2024/1362): krótki tekst z
paragrafami "§ N." (NIE "Art. N." jak w ustawach -- inny wzorzec numeracji,
inna gałąź prawa), typowo 2-3 paragrafy: kwota minimalnego wynagrodzenia,
kwota minimalnej stawki godzinowej (od ok. 2017), data wejścia w życie.
Cały dokument mieści się dobrze poniżej `MAX_CHUNK_CHARS` (900 w
build_rag_index.py), więc traktowany jest jako JEDEN "artykuł" (`article`
= stała `"1"` dla każdej wersji) -- nie dzielony po "§" osobnym splitterem,
prościej i bezpieczniej niż pisanie nowego parsera dla innego wzorca
numeracji, a użytkownik i tak dostaje obie kwoty razem w jednym fragmencie
kontekstu (przydatne, bo pytania czasem dotyczą obu naraz). Stała wartość
`article` na wszystkie lata też oznacza, że działa od razu z niezmienionym
mechanizmem grupowania/rankingu `(act_short, article)` w rag_search.py --
zero zmian potrzebnych w tamtym pliku.

`valid_from`/`valid_to`: prosto z roku w tytule aktu (regex, niezawodny --
w przeciwieństwie do pola `entryIntoForce`, którego brakuje w dwóch
najstarszych, typu "Obwieszczenie" wpisach), zawsze 1 stycznia danego roku
(akt "wchodzi w życie z dniem 1 stycznia [rok] r." -- ustawowy obowiązek
ogłoszenia do 15 września roku poprzedniego, art. 2 ust. 4 ustawy). W
PRZECIWIEŃSTWIE do obwieszczeń "tekst jednolity" z download_acts_history.py
(Krok 18/19), te okna NIE zachodzą na siebie ani się nie rozjeżdżają --
każdy akt jest jednorazowy i całkowicie odrębny, obowiązujący dokładnie do
dnia wejścia w życie NASTĘPNEGO w kolejności chronologicznej (`valid_to` =
`valid_from` kolejnego wpisu, `None` dla najnowszego).

Wyjście: `data/processed/rozporzadzenie_minimalne_wynagrodzenie_series.json`
-- płaska lista artykułów (ten sam kształt co `all_articles.json`), KAŻDY
już z `valid_from`/`valid_to` ustawionymi. W przeciwieństwie do
`{short}_history.json` z `download_acts_history.py` (wymaga patchowania
"bieżącej", wcześniej już pobranej wersji przez `build_rag_index.py
--include-history`), to kompletny, samodzielny "akt" -- wszystkie wersje,
łącznie z bieżącą, są pobierane tutaj, więc plik `*_series.json` jest
wczytywany bezpośrednio, bez patchowania (patrz `load_articles_with_history()`
w `build_rag_index.py`).

Uruchomienie:
    python scripts/download_wage_regulations.py
"""

import json
import re
import time

import requests

from download_acts import (
    API_BASE,
    HEADERS,
    PROCESSED_DIR,
    fetch_pdf_bytes,
    normalize_text,
    pdf_to_clean_text,
    pick_text_file,
    strip_not_yet_in_force_text,
)

ACT_SHORT = "rozporzadzenie_minimalne_wynagrodzenie"
SEARCH_TITLE_QUERY = "wysokości minimalnego wynagrodzenia za pracę"
TITLE_FILTER = "minimalnego wynagrodzenia"
TITLE_YEAR_RE = re.compile(r"w (\d{4}) r\.\s*$")


def search_acts_by_title(title_query: str) -> list[dict]:
    """Pełne metadane wszystkich aktów, których tytuł zawiera `title_query`
    (endpoint `/eli/acts/search?title=...`) -- każdy element to już KOMPLETNE
    metadane aktu (ten sam kształt co `fetch_meta()` na pojedynczym akcie),
    nie tylko referencja/ID, więc nie trzeba osobnego zapytania na kandydata."""
    r = requests.get(f"{API_BASE}/acts/search", params={"title": title_query}, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json().get("items", [])


def extract_year(meta: dict) -> int | None:
    """Rok, którego dotyczy akt -- z tytułu (niezawodne, obecne we
    WSZYSTKICH sprawdzonych wpisach, w tym dwóch najstarszych typu
    "Obwieszczenie", które NIE mają pola `entryIntoForce` w ogóle). Gdy
    `entryIntoForce` jest dostępne, sprawdzane jako spójność (zgodność
    potwierdzona empirycznie na wszystkich 23 wpisach 2004-2026 -- patrz
    Krok 21), ale to tytuł jest źródłem prawdy, nie `entryIntoForce`."""
    m = TITLE_YEAR_RE.search(meta.get("title", ""))
    if not m:
        return None
    year = int(m.group(1))
    entry = meta.get("entryIntoForce")
    if entry and not entry.startswith(f"{year}-01"):
        print(
            f"  [UWAGA] rok z tytułu ({year}) niezgodny z entryIntoForce "
            f"({entry}) dla {meta.get('ELI')} -- sprawdź ręcznie"
        )
    return year


def _warn_on_gaps(candidates: list[dict]) -> None:
    """Sprawdza, czy lata w `candidates` (już posortowanych) tworzą ciąg
    bez przerw -- czysto diagnostyczne (drukuje ostrzeżenie, nie przerywa
    działania). Nie ufamy samej liczbie wyników z `search` bez sprawdzenia:
    ten sam typ luki (brakujący rok mimo że akt istnieje w ELI) już raz
    wystąpił przy poprzednim podejściu opartym o `references['Akty
    wykonawcze']` -- patrz moduł docstring, Krok 21."""
    years = [c["year"] for c in candidates]
    missing = [y for y in range(years[0], years[-1] + 1) if y not in years]
    if missing:
        print(f"  [UWAGA] brakujące lata w serii: {missing} -- sprawdź ręcznie przez /eli/acts/search")


def fetch_regulation_series() -> list[dict]:
    """Pobiera i paruje wszystkie roczne akty o wysokości minimalnego
    wynagrodzenia, zwraca chronologicznie posortowaną listę słowników
    artykułów (kształt jak w all_articles.json), każdy już z ustawionymi
    valid_from/valid_to."""
    results = search_acts_by_title(SEARCH_TITLE_QUERY)

    candidates = []
    for meta in results:
        if TITLE_FILTER not in meta.get("title", ""):
            continue
        year = extract_year(meta)
        if year is None:
            print(f"  [UWAGA] nie udało się wyciągnąć roku z tytułu: {meta.get('title')!r} -- pomijam")
            continue
        source = {"publisher": meta["publisher"], "year": meta["year"], "position": meta["pos"]}
        candidates.append({"source": source, "meta": meta, "year": year})

    candidates.sort(key=lambda c: c["year"])
    _warn_on_gaps(candidates)
    print(f"[{ACT_SHORT}] znaleziono {len(candidates)} rocznych wersji ({candidates[0]['year']}-{candidates[-1]['year']})")

    articles = []
    for i, c in enumerate(candidates):
        valid_from = f"{c['year']}-01-01"
        valid_to = f"{candidates[i + 1]['year']}-01-01" if i + 1 < len(candidates) else None

        print(f"[{ACT_SHORT}] pobieram tekst za rok {c['year']} ({c['meta'].get('ELI')})...")
        kind, file_name = pick_text_file(c["meta"])
        pdf_bytes = fetch_pdf_bytes(c["source"], kind, file_name)
        clean_text = pdf_to_clean_text(pdf_bytes)
        clean_text = strip_not_yet_in_force_text(clean_text)
        text = normalize_text(clean_text)

        eli = c["meta"].get("ELI", f"{c['source']['publisher']}/{c['source']['year']}/{c['source']['position']}")
        source_url = (
            f"{API_BASE}/acts/{c['source']['publisher']}/{c['source']['year']}/{c['source']['position']}"
            f"/text/{kind}/{file_name}"
        )
        articles.append(
            {
                "id": f"{ACT_SHORT}_rok_{c['year']}",
                "act_short": ACT_SHORT,
                "act_title": c["meta"].get("title", ""),
                "eli": eli,
                "article": "1",
                "text": text,
                "source_url": source_url,
                "valid_from": valid_from,
                "valid_to": valid_to,
                "announcement_eli": None,
            }
        )
        time.sleep(1)  # nie bombardujemy publicznego API bez potrzeby -- ten sam wzorzec co download_acts.py

    return articles


def main():
    articles = fetch_regulation_series()
    out_path = PROCESSED_DIR / f"{ACT_SHORT}_series.json"
    out_path.write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nZapisano {len(articles)} wersji rocznych -> {out_path}")


if __name__ == "__main__":
    main()
