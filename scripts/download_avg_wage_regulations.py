"""
Pobiera coroczne komunikaty Prezesa GUS w sprawie przeciętnego wynagrodzenia
w gospodarce narodowej -- trzeci i (na razie) ostatni znany kandydat z
PROGRESS.md, "Co dalej" pkt 5. Ta wartość jest referencyjna dla wielu innych
przeliczeń w prawie pracy/ZUS (m.in. waloryzacja emerytur i rent) i sama w
sobie nie jest zapisana w żadnej pojedynczej ustawie.

**WAŻNE -- ten przypadek jest BARDZIEJ RYZYKOWNY niż minimalne wynagrodzenie
(Krok 21) i limit 30-krotności (Krok 22), i dlatego zakres jest tu celowo
węższy:** zapytanie `/eli/acts/search?title="przeciętnego wynagrodzenia w
gospodarce narodowej"` zwraca 81 wyników, bo GUS publikuje KILKA różnych,
POKREWNYCH, ale prawnie odrębnych wielkości pod bardzo podobnie brzmiącymi
tytułami -- sprawdzone empirycznie (Krok 23):
- "Komunikat [...] w sprawie przeciętnego wynagrodzenia w gospodarce
  narodowej w RRRR r." -- wartość ROCZNA za rok RRRR, podstawa prawna:
  art. 20 pkt 1 lit. a ustawy o emeryturach i rentach z FUS (NIE ustawa o
  systemie ubezpieczeń społecznych z ACTS!) -- **TYLKO TA seria jest tu
  pobierana.**
- "Obwieszczenie [...] w sprawie przeciętnego wynagrodzenia miesięcznego w
  gospodarce narodowej w RRRR r. i w drugim półroczu RRRR r." -- inna
  wielkość (miesięczna + półroczna naraz), inne zastosowania prawne.
  ŚWIADOMIE POMINIĘTE.
- "Obwieszczenie [...] w sprawie przeciętnego miesięcznego wynagrodzenia
  brutto w gospodarce narodowej w województwach w RRRR r." -- rozbicie
  wojewódzkie, inna wielkość. ŚWIADOMIE POMINIĘTE.

Filtr tytułu jest więc EXACT MATCH (regex zakotwiczony na początku i końcu
stringa), NIE luźne sprawdzenie podłańcucha jak w `download_wage_regulations.py`/
`download_zus_limit_regulations.py` -- luźny filtr złapałby też oba pominięte
warianty, mieszając trzy różne wielkości pod jednym `act_short`, co byłoby
gorsze niż brak danych (model mógłby podać niewłaściwą, ale przekonująco
brzmiącą liczbę). Sprawdzone: dokładny wzorzec zwraca 23 wpisy, 2003-2025,
bez przerw.

**`valid_from`/`valid_to` liczone inaczej niż w Krokach 21/22 -- z
`promulgation` (data faktycznego ogłoszenia komunikatu), NIE z roku w
tytule + "1 stycznia".** Powód: rozporządzenia płacowe i obwieszczenia o
limicie 30-krotności miały wprost w tekście klauzulę "wchodzi w życie z
dniem 1 stycznia RRRR r." -- jednoznaczną datę początku obowiązywania.
Komunikat GUS o przeciętnym wynagrodzeniu NIE ma takiej klauzuli w ogóle --
to retrospektywne ogłoszenie faktu ("przeciętne wynagrodzenie w gospodarce
narodowej w RRRR r. wyniosło X zł"), bez wskazania, od kiedy dokładnie ta
opublikowana wartość staje się "tą właściwą" dla różnych, korzystających z
niej ustaw (to zależy od przepisów KAŻDEJ z tych ustaw z osobna, nie od
samego komunikatu). Użycie `promulgation` (data faktycznego ogłoszenia,
zawsze znana i pewna) zamiast zgadywanej daty "1 stycznia roku
następnego" jest bezpieczniejsze -- nie twierdzi nic więcej, niż da się
zweryfikować wprost z metadanych aktu.

Struktura tekstu (sprawdzona na MP/2025/125): jedno zdanie, bez podziału na
"§" -- ten sam wzorzec co limit 30-krotności (Krok 22), jeden akt = jeden
"artykuł" (`article = "1"`).

Wyjście: `data/processed/komunikat_przecietne_wynagrodzenie_series.json`,
wczytywane przez `build_rag_index.py --include-history` bez zmian (ten sam
glob `*_series.json`).

Uruchomienie:
    python scripts/download_avg_wage_regulations.py
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

ACT_SHORT = "komunikat_przecietne_wynagrodzenie"
SEARCH_TITLE_QUERY = "przeciętnego wynagrodzenia w gospodarce narodowej"
# Zakotwiczone na początku I końcu -- patrz moduł docstring: luźny substring
# match złapałby też dwa inne, pokrewne, ale odrębne warianty (miesięczne +
# półroczne, wojewódzkie), które są świadomie poza zakresem tego skryptu.
TITLE_EXACT_RE = re.compile(
    r"^Komunikat Prezesa Głównego Urzędu Statystycznego .* "
    r"w sprawie przeciętnego wynagrodzenia w gospodarce narodowej w (\d{4}) r\.$"
)


def search_acts_by_title(title_query: str) -> list[dict]:
    """Patrz identyczna funkcja w download_wage_regulations.py/
    download_zus_limit_regulations.py."""
    r = requests.get(f"{API_BASE}/acts/search", params={"title": title_query}, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json().get("items", [])


def extract_year(meta: dict) -> int | None:
    """Rok, którego dotyczy komunikat -- TYLKO jeśli tytuł dokładnie
    pasuje do wzorca rocznego komunikatu (patrz TITLE_EXACT_RE i moduł
    docstring) -- w przeciwieństwie do Kroków 21/22, to jednocześnie filtr
    I ekstrakcja roku w jednym kroku, bo tu precyzja dopasowania tytułu
    jest krytyczna (odróżnienie od dwóch pokrewnych, ale odrębnych serii)."""
    m = TITLE_EXACT_RE.match(meta.get("title", ""))
    return int(m.group(1)) if m else None


def _warn_on_gaps(candidates: list[dict]) -> None:
    """Patrz identyczna funkcja w poprzednich dwóch skryptach."""
    years = [c["year"] for c in candidates]
    missing = [y for y in range(years[0], years[-1] + 1) if y not in years]
    if missing:
        print(f"  [UWAGA] brakujące lata w serii: {missing} -- sprawdź ręcznie przez /eli/acts/search")


def fetch_regulation_series() -> list[dict]:
    """Pobiera i paruje wszystkie roczne komunikaty o przeciętnym
    wynagrodzeniu, zwraca chronologicznie posortowaną listę słowników
    artykułów (kształt jak w all_articles.json), każdy już z ustawionymi
    valid_from/valid_to (z `promulgation`, patrz moduł docstring)."""
    results = search_acts_by_title(SEARCH_TITLE_QUERY)

    candidates = []
    for meta in results:
        year = extract_year(meta)
        if year is None:
            continue  # świadomie ciche pominięcie -- to oczekiwane dla 2 z 3 pokrewnych serii, nie błąd
        if not meta.get("promulgation"):
            print(f"  [UWAGA] brak pola 'promulgation' dla {meta.get('ELI')} -- pomijam")
            continue
        source = {"publisher": meta["publisher"], "year": meta["year"], "position": meta["pos"]}
        candidates.append({"source": source, "meta": meta, "year": year})

    candidates.sort(key=lambda c: c["year"])
    _warn_on_gaps(candidates)
    print(f"[{ACT_SHORT}] znaleziono {len(candidates)} rocznych wersji ({candidates[0]['year']}-{candidates[-1]['year']})")

    articles = []
    for i, c in enumerate(candidates):
        valid_from = c["meta"]["promulgation"]
        valid_to = candidates[i + 1]["meta"]["promulgation"] if i + 1 < len(candidates) else None

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
