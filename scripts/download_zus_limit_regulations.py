"""
Pobiera coroczne obwieszczenia ministra w sprawie kwoty ograniczenia rocznej
podstawy wymiaru składek na ubezpieczenia emerytalne i rentowe -- czyli
tzw. "limit 30-krotności" (art. 19 ustawy o systemie ubezpieczeń
społecznych, DU/1998/887, deleguje coroczne ogłoszenie konkretnej kwoty w
złotych do osobnego obwieszczenia; sama ustawa podaje tylko wzór/mnożnik,
nie kwotę) -- drugi kandydat z PROGRESS.md, "Co dalej" pkt 5, ten sam
wzorzec co `download_wage_regulations.py` (Krok 21/22): stały `article`,
proste roczne okna, discovery przez `/eli/acts/search`, NIE przez
`references["Akty wykonawcze"]` (świadomie -- ta druga ścieżka okazała się
niekompletna dla minimalnego wynagrodzenia, patrz Krok 21; nie ma powodu
ufać jej bardziej tutaj, więc konsekwentnie użyto od razu `search`).

Struktura każdego aktu (sprawdzona na MP/2024/1051): JEDNO zdanie, bez
podziału na paragrafy/artykuły w ogóle (prostsze niż nawet rozporządzenia
płacowe z "§ N.") -- "ogłasza się, że kwota ograniczenia rocznej podstawy
wymiaru składek [...] w roku RRRR wynosi X zł, a przyjęta do jej ustalenia
kwota prognozowanego przeciętnego wynagrodzenia wynosi Y zł." Cały tekst
mieści się dobrze poniżej `MAX_CHUNK_CHARS`, więc -- jak w
`download_wage_regulations.py` -- traktowany jako JEDEN "artykuł"
(`article = "1"` stałe na wszystkie lata).

Wzorzec tytułu inny niż przy minimalnym wynagrodzeniu: "w roku RRRR"
(czasem w środku tytułu, nie na końcu -- starsze wpisy z lat 1999-2007 nie
mają jeszcze klauzuli "oraz przyjętej do jej ustalenia [...]" na końcu, więc
regex szuka "w roku RRRR" gdziekolwiek w tytule, nie tylko na końcu jak dla
rozporządzeń płacowych "w RRRR r.").

`valid_from`/`valid_to`: rok z tytułu = rok, którego dotyczy limit -> 1
stycznia tego roku, `valid_to` = 1 stycznia roku kolejnego wpisu w
kolejności chronologicznej (`None` dla najnowszego) -- ten sam prosty,
nienakładający się model co przy minimalnym wynagrodzeniu (w
przeciwieństwie do obwieszczeń "tekst jednolity" z Kroku 18).

Wyjście: `data/processed/obwieszczenie_limit_skladek_zus_series.json`,
ten sam kształt co `rozporzadzenie_minimalne_wynagrodzenie_series.json` --
wczytywany przez `build_rag_index.py --include-history` bez żadnych zmian
(ten sam glob `*_series.json`).

Uruchomienie:
    python scripts/download_zus_limit_regulations.py
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

ACT_SHORT = "obwieszczenie_limit_skladek_zus"
SEARCH_TITLE_QUERY = "kwoty ograniczenia rocznej podstawy wymiaru składek"
TITLE_FILTER = "ograniczenia rocznej podstawy wymiaru składek"
TITLE_YEAR_RE = re.compile(r"w roku (\d{4})")


def search_acts_by_title(title_query: str) -> list[dict]:
    """Pełne metadane wszystkich aktów, których tytuł zawiera `title_query`
    -- patrz identyczna funkcja w `download_wage_regulations.py`, ten sam
    endpoint, ta sama semantyka (elementy `search` to już KOMPLETNE
    metadane, nie tylko referencje/ID)."""
    r = requests.get(f"{API_BASE}/acts/search", params={"title": title_query}, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json().get("items", [])


def extract_year(meta: dict) -> int | None:
    """Rok, którego dotyczy limit -- z tytułu (wzorzec "w roku RRRR",
    sprawdzony na wszystkich 28 wpisach 1999-2026, patrz Krok 22).
    `entryIntoForce` jest `None` dla niemal wszystkich wpisów tej serii
    (w przeciwieństwie do rozporządzeń płacowych) -- pomijane jako
    sprawdzenie spójności, bo praktycznie nigdy nie jest dostępne."""
    m = TITLE_YEAR_RE.search(meta.get("title", ""))
    return int(m.group(1)) if m else None


def _warn_on_gaps(candidates: list[dict]) -> None:
    """Patrz identyczna funkcja w `download_wage_regulations.py` -- ten sam
    powód (Krok 21 znalazł realną lukę w innej serii; nie ufamy samej
    liczbie wyników bez sprawdzenia ciągłości)."""
    years = [c["year"] for c in candidates]
    missing = [y for y in range(years[0], years[-1] + 1) if y not in years]
    if missing:
        print(f"  [UWAGA] brakujące lata w serii: {missing} -- sprawdź ręcznie przez /eli/acts/search")


def fetch_regulation_series() -> list[dict]:
    """Pobiera i paruje wszystkie roczne obwieszczenia o limicie 30-krotności,
    zwraca chronologicznie posortowaną listę słowników artykułów (kształt
    jak w all_articles.json), każdy już z ustawionymi valid_from/valid_to."""
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
