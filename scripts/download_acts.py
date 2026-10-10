"""
Pobiera polskie akty prawne z oficjalnego, darmowego ELI API Sejmu
(https://api.sejm.gov.pl/eli) i tnie je na fragmenty (artykuły) do JSON.

WAŻNA UWAGA O ŹRÓDLE TEKSTU:
Endpoint /text.html w tym API zwraca tekst PIERWOTNY aktu (z dnia
uchwalenia), a NIE tekst uwzględniający późniejsze nowelizacje --
sprawdzone empirycznie na art. 154 Kodeksu pracy (wymiar urlopu),
gdzie text.html zwracał formułę z 1974 r. ("14/17/20/26 dni
roboczych" zależnie od stażu), a obowiązująca od dawna treść to
"20/26 dni" zależnie od stażu >= 10 lat.

Zamiast tego korzystamy z pliku oznaczonego w metadanych aktu typem
"U" (tekst ujednolicony roboczy, przygotowywany i aktualizowany przez
Kancelarię Sejmu po każdej nowelizacji) -- to PDF, ale zawiera
faktycznie obowiązujący tekst. Pobieramy go z:
  /eli/acts/{publisher}/{year}/{position}/text/U/{fileName}
gdzie {fileName} odczytujemy z metadanych aktu (pole "texts").

Kroki dla każdej ustawy:
1. Pobierz metadane -> znajdź plik typu "U" (fallback: "O", jeśli
   akt nigdy nie był nowelizowany i nie ma osobnego tekstu ujednoliconego).
2. Pobierz PDF, wyciągnij tekst (pypdf).
3. Usuń nagłówki/stopki Kancelarii Sejmu ("©Kancelaria Sejmu ...",
   linie z samą datą) oraz nagłówki działów/rozdziałów.
4. Podziel tekst na artykuły po wzorcu "Art. <numer>." na początku
   linii (wielka litera "Art." zawsze zaczyna nagłówek artykułu;
   odniesienia wewnątrz tekstu używają małej litery "art.").
5. Zapisz:
   - data/raw/{skrot}.pdf         -- surowy PDF (do wglądu)
   - data/raw/{skrot}_meta.json   -- metadane aktu z API
   - data/processed/{skrot}.json  -- lista fragmentów (artykułów)
   - data/processed/all_articles.json -- wszystkie fragmenty razem

Uruchomienie:
    python scripts/download_acts.py
"""

import argparse
import difflib
import io
import json
import multiprocessing
import os
import re
import time
from collections import Counter
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

import pdfplumber
import requests
from pypdf import PdfReader

API_BASE = "https://api.sejm.gov.pl/eli"
HEADERS = {"User-Agent": "kadry-plyace-asystent/0.1 (lokalny projekt edukacyjny)"}

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# Trzy ustawy na start -- publisher/year/position zweryfikowane ręcznie przez API.
# Żeby dodać kolejny akt: znajdź jego identyfikator (np. przez
# /eli/acts/search?title=...) i dopisz wpis do tej listy.
ACTS = [
    {
        "short": "kodeks_pracy",
        "publisher": "DU",
        "year": 1974,
        "position": 141,
    },
    {
        "short": "ustawa_o_systemie_ubezpieczen_spolecznych",
        "publisher": "DU",
        "year": 1998,
        "position": 887,
    },
    {
        "short": "ustawa_o_minimalnym_wynagrodzeniu",
        "publisher": "DU",
        "year": 2002,
        "position": 1679,
    },
    {
        "short": "ustawa_zasilkowa",
        "publisher": "DU",
        "year": 1999,
        "position": 636,
    },
    {
        "short": "ustawa_o_pit",
        "publisher": "DU",
        "year": 1991,
        "position": 350,
    },
    {
        "short": "ustawa_o_ppk",
        "publisher": "DU",
        "year": 2018,
        "position": 2215,
    },
    {
        # Uwaga: dawna "ustawa o promocji zatrudnienia i instytucjach rynku
        # pracy" (DU/2004/1001) została UCHYLONA z dniem 2025-06-01 przez
        # tę ustawę (zweryfikowane przez pole "references"."Akty uchylające"
        # w API ELI) -- to jej następczyni, obecnie obowiązujące źródło
        # przepisów o zasiłku dla bezrobotnych.
        "short": "ustawa_o_rynku_pracy",
        "publisher": "DU",
        "year": 2025,
        "position": 620,
    },
]

HEADER_FOOTER_PATTERNS = [
    re.compile(r"^©Kancelaria Sejmu"),
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    # Ten sam stempel daty wygenerowania w starszym formacie (teksty
    # ujednolicone ~2017-2021, np. "24.04.2018" co stronę) -- trafiał do
    # treści artykułów (projekt prawo polskie, PROGRESS.md Krok 65).
    re.compile(r"^\d{2}\.\d{2}\.\d{4}$"),
    re.compile(r"^s\.\s*\d+/\d+$"),
    # Nagłówek strony tekstu jednolitego w układzie Dziennika Ustaw (pliki
    # "T" obwieszczeń, używane w historii od Kroku 69): "Dziennik Ustaw – 23 – Poz. 17".
    re.compile(r"^Dziennik Ustaw(?: Nr \d+)?\s*[–—-]\s*\d+\s*[–—-]\s*Poz\.\s*\d+$"),
]
SECTION_HEADING_PATTERNS = [
    re.compile(r"^DZIA[ŁL]\b", re.IGNORECASE),
    re.compile(r"^Rozdzia[łl]\b", re.IGNORECASE),
    re.compile(r"^Tytu[łl]\b", re.IGNORECASE),
]

ARTICLE_SPLIT_RE = re.compile(r"(?m)^Art\.\s*(\d+[a-ząćęłńóśźż]{0,3})\.\s*")

SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
SUPERSCRIPT_SIZE_RATIO = 0.85  # znak uznajemy za indeks górny, gdy jego czcionka jest mniejsza niż ten ułamek rozmiaru bazowego strony
QUOTE_CHARS = {"„", '"', "“", "”", "»"}  # "Art. N." w cudzysłowie to zwykle cytat nowelizowanego przepisu innej ustawy, nie prawdziwy nagłówek tego aktu


def find_article_numbers_pdfplumber(pdf_bytes: bytes) -> list[str]:
    """Wykrywa numery artykułów bezpośrednio z geometrii znaków w PDF-ie
    (rozmiar czcionki, pozycja), żeby odróżnić prawdziwe numery z
    indeksem górnym (np. "Art. 11¹" -- artykuły wstawiane między
    istniejące numery bez przenumerowania całego kodeksu) od zwykłych
    numerów wielocyfrowych (np. "Art. 111"), które `pypdf.extract_text()`
    spłaszcza do identycznego tekstu (potwierdzone bezpośrednio w
    surowym PDF-ie Kodeksu pracy -- patrz PROGRESS.md, krok 12).

    Zwraca listę numerów w kolejności występowania w dokumencie, z
    prawdziwym indeksem górnym zapisanym jako znak Unicode (np. "11¹").
    Pomija nagłówki wewnątrz "< ... >" (tekst jeszcze nieobowiązujący --
    ten sam fragment usuwa `strip_not_yet_in_force_text`) oraz nagłówki
    bezpośrednio poprzedzone znakiem cudzysłowu (`QUOTE_CHARS`) -- to
    zwykle cytat treści nowelizowanego przepisu z INNEJ ustawy (np.
    „Art. 7a. 1. Podmiotowi..." wewnątrz opisu "po art. 7 dodaje się
    art. 7a w brzmieniu:"), a nie prawdziwy artykuł tego aktu; oryginalny
    `ARTICLE_SPLIT_RE` poprawnie takie cytaty pomija dzięki wymogowi
    początku linii, więc bez tego wykluczenia liczba nagłówków się nie
    zgadzała (patrz PROGRESS.md, krok 12/13). Żeby liczba wykrytych
    tutaj nagłówków zgadzała się z liczbą artykułów po tamtej poprawce.
    Używana wyłącznie do KOREKTY numeracji już wyodrębnionych
    przez `parse_articles()` -- jeśli mimo to liczby się nie zgadzają,
    wołający ma pominąć korektę (patrz `process_act`)."""
    numbers: list[str] = []
    chars: list[dict] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            chars.extend(page.chars)
    if not chars:
        return numbers
    body_size = Counter(round(c["size"], 1) for c in chars).most_common(1)[0][0]
    i, n = 0, len(chars)
    in_not_yet_in_force = False  # wewnątrz "< ... >" -- patrz strip_not_yet_in_force_text, ten sam tekst tam jest usuwany
    while i < n:
        text = chars[i]["text"]
        if text == "<":
            in_not_yet_in_force = True
            i += 1
            continue
        if text == ">":
            in_not_yet_in_force = False
            i += 1
            continue
        preceded_by_quote = i > 0 and chars[i - 1]["text"] in QUOTE_CHARS
        if in_not_yet_in_force or preceded_by_quote or i >= n - 4 or not (
            chars[i]["text"] == "A" and chars[i + 1]["text"] == "r"
            and chars[i + 2]["text"] == "t" and chars[i + 3]["text"] == "."
        ):
            i += 1
            continue
        j = i + 4
        while j < n and chars[j]["text"] == " ":
            j += 1
        digit_chars = []
        while j < n and chars[j]["text"].isdigit():
            digit_chars.append(chars[j])
            j += 1
        # opcjonalny sufiks literowy (np. "30a", "18c") -- bez indeksu górnego,
        # ale MUSI trafić do wyniku, inaczej "183a"/"183b"/"183c" zlewają się w jedno
        letter_suffix = ""
        while j < n and chars[j]["text"].isalpha() and chars[j]["text"].islower():
            letter_suffix += chars[j]["text"]
            j += 1
        if digit_chars and j < n and chars[j]["text"] == ".":
            normal = "".join(c["text"] for c in digit_chars if c["size"] >= body_size * SUPERSCRIPT_SIZE_RATIO)
            super_ = "".join(c["text"] for c in digit_chars if c["size"] < body_size * SUPERSCRIPT_SIZE_RATIO)
            if normal:
                numbers.append(normal + super_.translate(SUPERSCRIPT_DIGITS) + letter_suffix)
            i = j + 1
        else:
            i += 1
    return numbers


RETRY_DELAYS = (5, 15, 45)  # sekundy między próbami przy chwilowych błędach serwera


def _get_with_retry(url: str, timeout: int) -> requests.Response:
    """GET z ponawianiem przy chwilowych błędach (5xx, zerwane połączenie,
    timeout). Pełne przetworzenie historii to ~150 pobrań z publicznego API
    i jeden "502 Proxy Error" zabijał cały przebieg (Krok 69) -- wyniki
    zapisywane są dopiero na końcu, więc nic nie trafiało na dysk. Błędy 4xx
    (np. 404) nie są ponawiane."""
    for attempt, delay in enumerate((*RETRY_DELAYS, None)):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code < 500:
                r.raise_for_status()
                return r
            err: Exception = requests.HTTPError(f"{r.status_code} dla {url}", response=r)
        except (requests.ConnectionError, requests.Timeout) as e:
            err = e
        if delay is None:
            raise err
        print(f"  [UWAGA] {err} -- ponawiam za {delay} s (próba {attempt + 2})", flush=True)
        time.sleep(delay)
    raise AssertionError("nieosiągalne")


def fetch_meta(act: dict) -> dict:
    url = f"{API_BASE}/acts/{act['publisher']}/{act['year']}/{act['position']}"
    return _get_with_retry(url, timeout=30).json()


CURRENT_TEXT_KINDS = ("U", "O")
# Wersje historyczne (obwieszczenia tekstu jednolitego): "T" (tekst
# obwieszczenia w formacie Kancelarii Sejmu -- migawka z dnia ogłoszenia),
# potem "O" (oryginał z Dz.U., ta sama migawka). NIE "U": dla obwieszczenia
# to tekst ujednolicony AKTUALIZOWANY o kolejne nowelizacje aż do
# wygaśnięcia okna, więc okno "2020-01-21 -- 2023-12-22" zawierało stan z
# końca 2023 r. (Krok 68/69: ustawa o prawach konsumenta, rozdział 5a z
# 2023 r. w oknie od 2020 r.). "U" zostaje tylko jako ostateczność.
HISTORY_TEXT_KINDS = ("T", "O", "U")


def pick_text_file(meta: dict, preferred: tuple[str, ...] = CURRENT_TEXT_KINDS) -> tuple[str, str]:
    """Wybiera plik z tekstem w kolejności `preferred`. Domyślnie (bieżący
    tekst ustawy) wolimy 'U' (ujednolicony), potem 'O' (oryginał); dla
    wersji historycznych -- `HISTORY_TEXT_KINDS`."""
    texts = {t["type"]: t["fileName"] for t in meta.get("texts", [])}
    for kind in preferred:
        if kind in texts:
            return kind, texts[kind]
    raise RuntimeError(f"Brak znanego typu tekstu w metadanych: {texts}")


def fetch_pdf_bytes(act: dict, kind: str, file_name: str) -> bytes:
    url = f"{API_BASE}/acts/{act['publisher']}/{act['year']}/{act['position']}/text/{kind}/{file_name}"
    return _get_with_retry(url, timeout=60).content


FOOTNOTE_MARKER_RE = re.compile(r"^(\d{1,3})\)$")
FOOTNOTE_BODY_SIZE_RATIO = 0.95  # linia "treści" strony ma czcionkę >= 95% bazowej (przypisy: 10 pt przy 12 pt tekstu w "U", 9 pt przy 10 pt w plikach "T" obwieszczeń -- przy 0.9 te drugie były dokładnie na progu, Krok 69)
FOOTNOTE_MARKER_SIZE_RATIO = 0.7  # numer przypisu to indeks górny ~6,5 pt (54% z 12 pt); treść przypisu 10 pt (83%) musi być POWYŻEJ progu


FOOTNOTE_SNIPPET_CHARS = 15  # ile pierwszych znaków (bez spacji) treści przypisu służy do jego odnalezienia w tekście pypdf


def _is_header_footer_row(chars: list[dict]) -> bool:
    text = "".join(c["text"] for c in sorted(chars, key=lambda c: c["x0"])).strip()
    return bool(text) and any(p.match(text) for p in HEADER_FOOTER_PATTERNS)


def find_footnote_markers_pdfplumber(pdf_bytes: bytes) -> list[tuple[str, str] | None]:
    """(numer, początek treści) PIERWSZEGO przypisu dolnego na każdej stronie
    (albo None). Początek treści (pierwsze `FOOTNOTE_SNIPPET_CHARS` znaków
    bez spacji) jest potrzebny, bo przypis sam bywa wyliczeniem -- przypis
    "1) Niniejsza ustawa wdraża: 1) dyrektywę...; 2) ..." na 1. stronie
    ustawy zawiera własne linie "1) ...", więc samo "utnij od ostatniej
    linii '1)'" zostawiało jego nagłówek w art. 1 (projekt "RAG-LoRA - prawo polskie", PROGRESS.md Krok 63).

    Kancelaria Sejmu drukuje przypisy ("15) Zmiany tekstu jednolitego
    wymienionej ustawy zostały ogłoszone w...") na dole strony, mniejszą
    czcionką (10 pt przy 12 pt tekstu), z numerem jako indeksem górnym
    (~6,5 pt). `pypdf` wypluwa je jako zwykłe linie na końcu tekstu strony,
    więc trafiały do środka artykułu łamanego przez stronę -- potwierdzone w
    co najmniej 129 artykułach 28 ustaw, a numer przypisu wyglądał jak
    kolejny punkt wyliczenia (adapter zacytował nieistniejący "art. 389 pkt
    15" Prawa wodnego, projekt "RAG-LoRA - prawo polskie", PROGRESS.md Krok 61/63).

    Marker to wiersz złożony WYŁĄCZNIE z małych cyfr i ")" (numer przypisu
    ma własną wysokość linii), pod którym nie ma już żadnej linii czcionką
    treści -- to odróżnia go od odsyłacza w tekście ("z późn. zm.15)"),
    który zawsze ma pod sobą dalszą treść strony albo leży w tej samej linii
    co tekst 12 pt. Zwykłe wyliczenia "15) ..." w treści mają 12 pt i nigdy
    nie spełniają warunku rozmiaru."""
    markers: list[str | None] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        sizes = Counter(round(c["size"], 1) for page in pdf.pages for c in page.chars)
        if not sizes:
            return [None] * len(pdf.pages)
        body_size = sizes.most_common(1)[0][0]
        for page in pdf.pages:
            rows: dict[int, list[dict]] = {}
            for c in page.chars:
                rows.setdefault(round(c["top"]), []).append(c)
            ordered = sorted(rows.items())
            found = None
            for idx, (_, chars) in enumerate(ordered):
                visible = [c for c in sorted(chars, key=lambda c: c["x0"]) if c["text"].strip()]
                # Numer przypisu = wiodący ciąg małych znaków ("15)"); reszta
                # wiersza (jeśli numer wypadł w tym samym wierszu co treść
                # przypisu) musi być czcionką mniejszą niż treść strony.
                n_lead = 0
                while n_lead < len(visible) and visible[n_lead]["size"] < body_size * FOOTNOTE_MARKER_SIZE_RATIO:
                    n_lead += 1
                lead = "".join(c["text"] for c in visible[:n_lead])
                if lead.isdigit() and n_lead == len(visible) and idx + 1 < len(ordered):
                    # Starszy układ Dz.U. (teksty jednolite ~2006-2014): sam numer
                    # "2" małą czcionką w osobnym wierszu, a ")" już w wierszu z
                    # treścią przypisu (Krok 69: 0/141 stron Prawa
                    # telekomunikacyjnego DU/2014/243).
                    next_visible = [c for c in sorted(ordered[idx + 1][1], key=lambda c: c["x0"]) if c["text"].strip()]
                    if next_visible and next_visible[0]["text"] == ")":
                        lead += ")"
                if not FOOTNOTE_MARKER_RE.match(lead):
                    continue
                if any(c["size"] >= body_size * FOOTNOTE_BODY_SIZE_RATIO for c in visible[n_lead:]):
                    continue
                text = lead
                # Wiersze nagłówka/stopki (np. stempel daty "2024-01-05", który w
                # plikach "T" obwieszczeń stoi POD przypisami, pełną czcionką)
                # nie są treścią -- inaczej żaden przypis na takiej stronie nie
                # zostałby wykryty (Krok 69: 0/84 stron KK DU/2024/17).
                below_has_body = any(
                    c["text"].strip() and c["size"] >= body_size * FOOTNOTE_BODY_SIZE_RATIO
                    for _, cs in ordered[idx + 1:]
                    if not _is_header_footer_row(cs)
                    for c in cs
                )
                if not below_has_body:
                    rest = [c for c in visible[n_lead:]]
                    for _, cs in ordered[idx + 1:]:
                        if len("".join(c["text"] for c in rest if c["text"].strip())) >= FOOTNOTE_SNIPPET_CHARS:
                            break
                        rest.extend(c for c in sorted(cs, key=lambda c: c["x0"]) if c["text"].strip())
                    snippet = "".join(c["text"] for c in rest if c["text"].strip())
                    snippet = snippet.removeprefix(")")[:FOOTNOTE_SNIPPET_CHARS]  # ")" z następnego wiersza (układ z osobnym numerem)
                    found = (FOOTNOTE_MARKER_RE.match(text).group(1), snippet)
                    break
            markers.append(found)
    return markers


def strip_page_footnotes(page_lines: list[str], marker: tuple[str, str] | str | None) -> list[str]:
    """Obcina linie strony od linii, od której zaczyna się blok przypisów:
    "{numer})" + początek treści przypisu (z `find_footnote_markers_pdfplumber`;
    porównanie bez spacji, bo pypdf i pdfplumber różnie je wstawiają).
    Szukamy OD KOŃCA (przypisy są zawsze ostatnim blokiem tekstu strony w
    pypdf, a zwykły punkt "15)" wyżej na stronie ma inną treść). Bez
    dopasowania treści -- ostatnia linia z samym numerem (np. gdy numer i
    treść przypisu leżą w pypdf w różnych liniach). `marker` jako sam numer
    (str) -- tylko to drugie kryterium."""
    if marker is None:
        return page_lines
    number, snippet = (marker, "") if isinstance(marker, str) else marker
    prefix = f"{number})"

    def squash(text: str) -> str:
        return "".join(text.split())

    if snippet:
        joined_tail = ""
        for i in range(len(page_lines) - 1, -1, -1):
            joined_tail = squash(page_lines[i]) + joined_tail
            if page_lines[i].strip().startswith(prefix) and joined_tail[len(prefix):].startswith(squash(snippet)):
                return page_lines[:i]
    for i in range(len(page_lines) - 1, -1, -1):
        if page_lines[i].strip().startswith(prefix):
            return page_lines[:i]
    return page_lines


FROM_SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")


def align_superscript_corrections(pypdf_labels: list[str], detected: list[str]) -> dict[int, str]:
    """Korekta indeksu górnego, gdy liczba nagłówków z pdfplumber różni się
    od liczby artykułów z pypdf (np. jeden nagłówek wykryty tylko przez
    jedno narzędzie). Dotąd korekta była wtedy pomijana W CAŁOŚCI -- każdy
    art. 22¹ zostawał jako "221" i kolidował z prawdziwym art. 221 (13
    historycznych okien, ~75 zdublowanych etykiet, np. KC 2013: 28 --
    projekt "RAG-LoRA - prawo polskie", PROGRESS.md Krok 64). Teraz obie sekwencje dopasowujemy lokalnie
    (`difflib`) po SPŁASZCZONEJ etykiecie z pdfplumber ("22¹" -> "221") i
    poprawiamy tylko pary, które się zgadzają -- rozjazd w jednym miejscu
    nie blokuje korekty w pozostałych. Zwraca {indeks artykułu pypdf:
    poprawna etykieta} tylko dla etykiet, które faktycznie się zmieniają."""
    flattened = [d.translate(FROM_SUPERSCRIPT_DIGITS) for d in detected]
    matcher = difflib.SequenceMatcher(None, flattened, pypdf_labels, autojunk=False)
    fixes: dict[int, str] = {}
    for a, b, size in matcher.get_matching_blocks():
        for k in range(size):
            if detected[a + k] != pypdf_labels[b + k]:
                fixes[b + k] = detected[a + k]
    return fixes


_UPPER = "A-ZĄĆĘŁŃÓŚŹŻ"
INLINE_FOOTNOTE_REF_PATTERNS = [
    # "§ 1.145) Kto..." / "Art. 75a.46) ..." -- odsyłacz do przypisu tuż po numerze
    re.compile(r"((?:§|Art\.)\s*\d+[a-ząćęłńóśźż]{0,3}[⁰¹²³⁴⁵⁶⁷⁸⁹]{0,3}[a-z]?\.)\d{1,3}\)(?=\s)"),
    # "(uchylony)46)" / "(pominięte)9)"
    re.compile(r"(\((?:uchylon[yae]|pominięt[yae])\))\d{1,3}\)"),
    # ustęp "1c.9) W okresie..." -- tylko przed wielką literą, inaczej łapało
    # np. "(PKWiU ex 22.11) – z wyłączeniem" w VAT
    re.compile(rf"((?:^|(?<=\s))\d+[a-z]?\.)\d{{1,3}}\)(?=\s+[{_UPPER}])", re.M),
]


def strip_inline_footnote_refs(text: str) -> str:
    """Usuwa odsyłacze do przypisów wklejone przez pypdf bez spacji za
    numerem jednostki redakcyjnej ("§ 1.145)", "(uchylony)46)", "2.46)").
    W plikach "T" obwieszczeń (historia od Kroku 69) ok. 200 na kodeks; w
    bieżących tekstach "U" pojedyncze resztki (np. Prawo o ruchu drogowym
    art. 75e "4.13)"). Numer przypisu wyglądał jak kolejny punkt wyliczenia
    -- ten sam problem co z treścią przypisów (Krok 61/63)."""
    for pattern in INLINE_FOOTNOTE_REF_PATTERNS:
        text = pattern.sub(r"\1", text)
    return text


_EDITORIAL_FOOTNOTE_PHRASES = (
    r"(?:[Ww] brzmieniu ustalonym przez|[Nn]adan\w* przez|[Dd]odan\w* przez|[Uu]chylon\w* przez"
    r"|[Zz]mienion\w* przez|[Ww]prowadzon\w* przez|Przez art\. \d|Utracił[ay]? moc"
    r"|Zmiany (?:tekstu jednolitego )?wymienionej ustawy zostały ogłoszone)"
)
EDITORIAL_FOOTNOTE_START_RE = re.compile(
    # "12) Zdanie pierwsze w brzmieniu ustalonym przez..." albo, w najstarszych
    # tekstach (Prawo lotnicze 2006), literowy numer "d) Dodany przez..." --
    # przy literze formuła musi stać od razu po nawiasie (zwykłe "a) ..." wyliczeń).
    rf"^(?:\d{{1,3}}\)\s*(?:\S+\s+){{0,4}}?|[a-z]{{1,2}}\)\s*){_EDITORIAL_FOOTNOTE_PHRASES}"
)
EDITORIAL_FOOTNOTE_MAX_LINES = 6


def pdf_to_clean_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    footnote_markers = find_footnote_markers_pdfplumber(pdf_bytes)
    lines: list[str] = []
    for page, marker in zip(reader.pages, footnote_markers):
        footnote_lines_left = 0
        for line in strip_page_footnotes(page.extract_text().split("\n"), marker):
            stripped = line.strip()
            if not stripped:
                continue
            if any(p.match(stripped) for p in HEADER_FOOTER_PATTERNS):
                continue
            # Przypis redakcyjny, którego nie wycięła geometria (dwukolumnowe
            # oryginały Dz.U. ~2006-2012, pliki "O" w historii, Krok 69):
            # "12) W brzmieniu ustalonym przez art. 3 ustawy z dnia ..." -- wiersz
            # i jego kontynuacje aż do kropki kończącej przypis.
            if EDITORIAL_FOOTNOTE_START_RE.match(stripped):
                footnote_lines_left = EDITORIAL_FOOTNOTE_MAX_LINES
            if footnote_lines_left:
                footnote_lines_left = 0 if stripped.endswith(".") else footnote_lines_left - 1
                continue
            if any(p.match(stripped) for p in SECTION_HEADING_PATTERNS):
                continue
            # PDF dzieli słowa na końcu wyjustowanej linii dywizem
            # (np. "zaokrąg-" / "la się" -> "zaokrągla się"); sklejamy je.
            if lines and lines[-1].endswith("-") and stripped[:1].islower():
                lines[-1] = lines[-1][:-1] + stripped
            else:
                lines.append(stripped)
    return strip_inline_footnote_refs("\n".join(lines))


NOT_YET_IN_FORCE_RE = re.compile(r"<.*?>", re.DOTALL)
SUPERSEDED_STILL_VALID_RE = re.compile(r"\[(.*?)\]", re.DOTALL)


# "\S{1,3}" zamiast "ł", bo najstarsze PDF-y (2006) mają zepsute kodowanie
# polskich znaków ("Za∏àcznik", "Marsza∏ek").
ANNOUNCEMENT_ANNEX_RE = re.compile(
    r"^Za[^\n]{1,5}cznik do obwieszczenia\b.*\n(?:.*\n){0,3}?USTAWA\s*$", re.MULTILINE  # "Zał ącznik" -- fałszywa spacja pypdf (Krok 69)
)
# "Marszałek Sejmu: S. Hołownia" albo (starsze pliki "T", Krok 69) samo
# "MARSZAŁEK SEJMU" wielkimi literami z nazwiskiem w następnej linii.
ANNOUNCEMENT_SIGNATURE_RE = re.compile(r"^(?:Marsza\S{1,3}ek Sejmu: .*|MARSZA\S{1,3}EK SEJMU\s*)$", re.MULTILINE)


def strip_announcement_preamble(text: str) -> str:
    """PDF obwieszczenia (tekst typu "O") zawiera przed właściwym tekstem
    jednolitym samo obwieszczenie, a w nim przytoczone przepisy NIEOBJĘTE
    tekstem jednolitym ("tekst jednolity nie obejmuje art. 10-24 ...,
    które stanowią: Art. 10. ..."). Parser brał je za artykuły ustawy --
    stąd podwojone art. 10-24 ustawy o min. wynagrodzeniu (okno 2015) i
    art. 3-6 Prawa lotniczego (okno 2006), projekt prawo polskie, PROGRESS.md Krok 65.

    Obwieszczenie rozpoznajemy po nagłówku "Załącznik do obwieszczenia ..."
    z "USTAWA" kilka linii niżej, ale tniemy za OSTATNIM podpisem
    "Marszałek Sejmu: ..." przed nim, nie na samym nagłówku: na
    dwukolumnowej stronie z 2006 r. pypdf wypisuje nagłówek załącznika PO
    art. 1 ustawy, więc cięcie na nagłówku gubiłoby art. 1. Brak nagłówka
    (zwykły tekst ujednolicony "U") -> tekst bez zmian; brak podpisu -> cięcie
    na nagłówku załącznika (Krok 69)."""
    annex = ANNOUNCEMENT_ANNEX_RE.search(text)
    if not annex:
        return text
    signatures = list(ANNOUNCEMENT_SIGNATURE_RE.finditer(text, 0, annex.start()))
    if not signatures:
        # Bez rozpoznanego podpisu tniemy na samym nagłówku załącznika --
        # zostawienie preambuły dawało przytoczone przepisy innych ustaw jako
        # artykuły tej ustawy (Krok 69: 467 zdublowanych etykiet w historii).
        return text[annex.start():]
    return text[signatures[-1].end():]


def strip_not_yet_in_force_text(text: str) -> str:
    """Kancelaria Sejmu oznacza w tekście ujednoliconym fragmenty prawa
    uchwalonego, ale jeszcze nieobowiązującego (przyszła data wejścia w
    życie) nawiasami ostrymi "< >", a fragmenty aktualnie obowiązujące,
    które te przepisy docelowo zastąpią, nawiasami kwadratowymi "[ ]"
    (stary tekst, wciąż ważny do dnia wejścia w życie nowelizacji).

    Asystent ma podawać WYŁĄCZNIE aktualnie obowiązujące prawo, więc
    "< >" trzeba usunąć w całości (razem z zawartością), a z "[ ]"
    zostawić samą treść w środku (to nadal ważne prawo, nawiasy to
    tylko adnotacja edytorska). Potwierdzony przypadek: Art. 85c-85j
    ustawy systemowej (nieobowiązujące jeszcze zmiany w orzecznictwie
    lekarskim ZUS) były przed tą poprawką po cichu wchłaniane jako
    treść poprzedniego, prawdziwego artykułu (patrz PROGRESS.md, krok
    12) -- czyli przedstawiane tak, jakby już obowiązywały.

    Podstawienie jest nie-zachłanne (`.*?`), więc niesparowany "<" bez
    odpowiadającego ">" po prostu nie zostanie dopasowany (nic się nie
    usuwa) zamiast ryzykować pochłonięcie zbyt dużego fragmentu poprawnego
    tekstu -- ale wtedy wypisujemy ostrzeżenie do ręcznego sprawdzenia."""
    if text.count("<") != text.count(">"):
        print(
            "[UWAGA] niesparowane nawiasy < > w tekście -- część fragmentu "
            "'jeszcze nieobowiązującego' mogła nie zostać usunięta, sprawdź ręcznie"
        )
    if text.count("[") != text.count("]"):
        print("[UWAGA] niesparowane nawiasy [ ] w tekście -- sprawdź ręcznie")
    text = NOT_YET_IN_FORCE_RE.sub("", text)
    text = SUPERSEDED_STILL_VALID_RE.sub(r"\1", text)
    return text


MIDTEXT_SUPERSCRIPT_HEADER_RE = re.compile(r"Art\. (\d+) (\d+[a-ząćęłńóśźż]{0,3})\.")


def recover_midtext_superscript_headers(clean_text: str) -> str:
    """`pypdf` czasem wstawia spację między głównym numerem a indeksem
    górnym artykułu, gdy nagłówek trafia się w środku akapitu (bez
    łamania linii) -- np. "Art. 22 3." zamiast "Art. 223." tak jak przy
    normalnie złamanych nagłówkach (bez spacji). Taki nagłówek nigdy nie
    trafia na `ARTICLE_SPLIT_RE` (wymaga początku linii), więc cała
    treść artykułu zostaje pochłonięta przez poprzedni artykuł --
    potwierdzony przypadek: Art. 22³ Kodeksu pracy (monitoring poczty
    elektronicznej) całkowicie zniknął jako osobny wpis, wtopiony w
    Art. 22² (patrz PROGRESS.md, krok 12). Sklejamy numer i wymuszamy
    początek nowej linii, żeby split go złapał."""
    return MIDTEXT_SUPERSCRIPT_HEADER_RE.sub(r"\nArt. \1\2.", clean_text)


STRAY_SPACE_BEFORE_PERIOD_RE = re.compile(r"(Art\. \d+[a-ząćęłńóśźż]{0,3}) \.")


def fix_stray_space_before_period(clean_text: str) -> str:
    """`pypdf` czasem wstawia dodatkową spację tuż przed kropką kończącą
    nagłówek artykułu (np. "Art. 52zb ." zamiast "Art. 52zb.") -- mimo
    poprawnego złamania linii przed nagłówkiem, `ARTICLE_SPLIT_RE` wymaga
    kropki bezpośrednio po numerze, więc taki artykuł (potwierdzony
    przypadek: Art. 52zb ustawy o PIT) był po cichu wchłaniany przez
    poprzedni, tak samo jak w `recover_midtext_superscript_headers`
    (patrz PROGRESS.md, krok 13)."""
    return STRAY_SPACE_BEFORE_PERIOD_RE.sub(r"\1.", clean_text)


def normalize_text(s: str) -> str:
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", " ", s)
    return s.strip()


def parse_articles(clean_text: str) -> list[dict]:
    parts = ARTICLE_SPLIT_RE.split(clean_text)
    # parts = [preambuła, numer_art_1, tekst_1, numer_art_2, tekst_2, ...]
    articles = []
    for i in range(1, len(parts) - 1, 2):
        article_no = parts[i]
        body = normalize_text(parts[i + 1])
        if not body:
            continue
        articles.append({"article": article_no, "text": f"Art. {article_no}. {body}"})
    return articles


def _fetch_source(
    source: dict, act_short: str, preferred: tuple[str, ...] = CURRENT_TEXT_KINDS
) -> tuple[dict, str, str, bytes, str, str]:
    """Sieciowa połowa `_extract_articles()`: metadane + PDF jednego aktu ELI
    (akt bazowy albo obwieszczenie tekstu jednolitego). Wydzielona, żeby
    pobieranie (sekwencyjne, łagodne dla publicznego API) dało się
    oddzielić od parsowania (CPU, równolegle -- patrz `ParsePool`)."""
    print(f"[{act_short}] pobieram metadane...", flush=True)
    meta = fetch_meta(source)
    title = meta.get("title", "")
    eli = meta.get("ELI", f"{source['publisher']}/{source['year']}/{source['position']}")
    print(f"[{act_short}] {title}  (status: {meta.get('status')})", flush=True)

    kind, file_name = pick_text_file(meta, preferred)
    print(f"[{act_short}] pobieram tekst typu '{kind}' ({file_name})...", flush=True)
    if preferred == HISTORY_TEXT_KINDS and kind == "U":
        print(f"  [UWAGA] {act_short}: brak tekstu 'T'/'O' -- 'U' może zawierać późniejsze nowelizacje", flush=True)
    pdf_bytes = fetch_pdf_bytes(source, kind, file_name)
    return meta, title, eli, pdf_bytes, kind, file_name


def _extract_articles(
    source: dict, act_short: str, preferred: tuple[str, ...] = CURRENT_TEXT_KINDS
) -> tuple[dict, str, str, list[dict], bytes, str, str]:
    """Pobiera i parsuje DOWOLNY akt ELI (`{'publisher','year','position'}`)
    -- akt bazowy z ACTS ALBO obwieszczenie ogłaszające historyczny tekst
    jednolity, oba mają identyczny kształt -- do listy artykułów. Wydzielone
    z `process_act()`, żeby ten sam pipeline (fetch meta -> wybór tekstu ->
    PDF -> czyszczenie -> parsowanie -> korekta indeksu górnego) obsługiwał
    też historyczne wersje w `process_act_version()`
    (`download_acts_history.py`) bez duplikacji kodu.

    Czysta ekstrakcja+parsowanie -- żadnych zapisów na dysk (wołający
    decyduje, co i czy w ogóle zapisać). Zwraca też `meta`/`pdf_bytes`, żeby
    `process_act()` mógł zapisać `data/raw/{short}.pdf`/`_meta.json` bez
    drugiego zapytania HTTP o te same metadane. Sekwencyjny odpowiednik
    pary `_fetch_source()` + `parse_pdf_articles()`."""
    meta, title, eli, pdf_bytes, kind, file_name = _fetch_source(source, act_short, preferred)
    articles = parse_pdf_articles(pdf_bytes, act_short)
    return meta, title, eli, articles, pdf_bytes, kind, file_name


def parse_pdf_articles(pdf_bytes: bytes, act_short: str) -> list[dict]:
    """CPU-owa połowa `_extract_articles()`: PDF -> lista artykułów. Czysta
    funkcja na poziomie modułu (bez sieci, bez zapisu), więc może działać w
    osobnym procesie `ParsePool`."""
    print(f"[{act_short}] wyciągam i czyszczę tekst z PDF...")
    clean_text = pdf_to_clean_text(pdf_bytes)
    clean_text = strip_announcement_preamble(clean_text)
    clean_text = strip_not_yet_in_force_text(clean_text)
    clean_text = recover_midtext_superscript_headers(clean_text)
    clean_text = fix_stray_space_before_period(clean_text)

    print(f"[{act_short}] tnę na artykuły...")
    articles = parse_articles(clean_text)
    print(f"[{act_short}] znaleziono {len(articles)} artykułów")

    print(f"[{act_short}] sprawdzam numerację pod kątem indeksów górnych...")
    detected_numbers = find_article_numbers_pdfplumber(pdf_bytes)
    if len(detected_numbers) == len(articles):
        fixes = {i: d for i, (a, d) in enumerate(zip(articles, detected_numbers)) if d != a["article"]}
    else:
        fixes = align_superscript_corrections([a["article"] for a in articles], detected_numbers)
        print(
            f"[{act_short}] liczba nagłówków pdfplumber ({len(detected_numbers)}) != artykułów z pypdf "
            f"({len(articles)}) -- korekta indeksu górnego przez lokalne dopasowanie"
        )
    for i, detected in fixes.items():
        article = articles[i]
        article["text"] = article["text"].replace(f"Art. {article['article']}.", f"Art. {detected}.", 1)
        article["article"] = detected
    if fixes:
        print(f"[{act_short}] poprawiono numerację indeksu górnego w {len(fixes)} artykułach")

    return articles


PARSE_WORKER_RAM_GB = 3  # zmierzony szczyt jednego procesu: 2,36 GB (Prawo lotnicze, 446 stron) + zapas


def _available_ram_gb() -> float | None:
    """WOLNA (nie całkowita) pamięć -- liczy się to, co zostało np. przy
    załadowanym w LM Studio Qwenie (~30 GB); ta maszyna już się zawieszała
    pod presją pamięci. `psutil` (wieloplatformowy); bez niego -- połowa
    całkowitego RAM-u z sysconf (macOS/Linux), na Windowsie bez psutil None."""
    try:
        import psutil

        return psutil.virtual_memory().available / 2**30
    except ImportError:
        pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30 / 2
    except (AttributeError, ValueError, OSError):
        return None


def default_parse_workers(n_jobs: int | None = None) -> int:
    """Liczba procesów do równoległego parsowania PDF-ów, dobierana do maszyny:
    rdzenie dostępne dla procesu minus jeden (zostaje dla wątku pobierającego
    i systemu), ograniczone WOLNĄ pamięcią (`PARSE_WORKER_RAM_GB` na proces)
    i liczbą zadań. Działa na macOS/Linuksie/Windowsie."""
    process_cpu_count = getattr(os, "process_cpu_count", None)  # Python 3.13+: respektuje affinity/cgroups
    cpus = (process_cpu_count() if process_cpu_count else None) or os.cpu_count() or 1
    workers = max(1, cpus - 1)
    ram_gb = _available_ram_gb()
    if ram_gb is not None:
        workers = min(workers, max(1, int(ram_gb // PARSE_WORKER_RAM_GB)))
    if n_jobs is not None:
        workers = min(workers, max(1, n_jobs))
    return workers


class ParsePool:
    """Parsowanie PDF-ów (`parse_pdf_articles`) w osobnych procesach -- czysta
    praca CPU (pypdf + pdfplumber, jednowątkowe), bez modeli w pamięci, więc
    bezpieczna do zrównoleglenia (w odróżnieniu od MLX/embeddingów). Pobieranie
    zostaje sekwencyjne u wołającego; parsowanie już pobranych PDF-ów idzie w
    tle, równolegle z pobieraniem kolejnych. `workers=1` -- bez puli, wszystko
    w bieżącym procesie (identyczny wynik, łatwiejsze debugowanie).
    Kontekst "spawn" jawnie: ten sam na każdym systemie (na Linuksie domyślny
    "fork" bywa niebezpieczny z bibliotekami trzymającymi wątki)."""

    def __init__(self, workers: int):
        self.workers = workers
        self._executor = (
            ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
            if workers > 1
            else None
        )

    def submit(self, pdf_bytes: bytes, label: str) -> Future:
        if self._executor is not None:
            return self._executor.submit(parse_pdf_articles, pdf_bytes, label)
        future: Future = Future()
        future.set_result(parse_pdf_articles(pdf_bytes, label))
        return future

    def __enter__(self) -> "ParsePool":
        return self

    def __exit__(self, *exc) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=exc[0] is not None)


def save_act_raw(act: dict, meta: dict, pdf_bytes: bytes) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / f"{act['short']}.pdf").write_bytes(pdf_bytes)
    (RAW_DIR / f"{act['short']}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_act_chunks(act: dict, title: str, eli: str, kind: str, file_name: str, articles: list[dict]) -> dict:
    source_url = (
        f"{API_BASE}/acts/{act['publisher']}/{act['year']}/{act['position']}/text/{kind}/{file_name}"
    )
    chunks = [
        {
            "id": f"{act['short']}_art_{a['article']}",
            "act_short": act["short"],
            "act_title": title,
            "eli": eli,
            "article": a["article"],
            "text": a["text"],
            "source_url": source_url,
        }
        for a in articles
    ]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / f"{act['short']}.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"short": act["short"], "title": title, "eli": eli, "chunks": chunks}


def process_act(act: dict) -> dict:
    """Sekwencyjna ścieżka dla jednego aktu (pobranie + parsowanie + zapis);
    `main()` robi to samo z parsowaniem w `ParsePool`."""
    meta, title, eli, articles, pdf_bytes, kind, file_name = _extract_articles(act, act["short"])
    save_act_raw(act, meta, pdf_bytes)
    return write_act_chunks(act, title, eli, kind, file_name, articles)


def _parse_eli_ref(ref_id: str) -> dict:
    """'DU/2020/2207' -> {'publisher': 'DU', 'year': 2020, 'position': 2207}
    -- kształt identyczny z wpisami ACTS, gotowy do fetch_meta()/_extract_articles()."""
    publisher, year, position = ref_id.split("/")
    return {"publisher": publisher, "year": int(year), "position": int(position)}


def fetch_unified_text_announcements(base_meta: dict) -> list[dict]:
    """PURE (bez HTTP) -- parsuje `references['Inf. o tekście jednolitym']`
    z już pobranych metadanych aktu bazowego (lista obwieszczeń ogłaszających
    kolejne teksty ujednolicone w historii ustawy) na listę słowników
    `{'publisher','year','position'}`. Pusta lista, jeśli akt nigdy nie miał
    ogłoszonego tekstu jednolitego (np. bardzo młode ustawy -- sprawdzone
    empirycznie na ustawie o rynku pracy, PROGRESS.md krok 18)."""
    refs = base_meta.get("references", {}).get("Inf. o tekście jednolitym", [])
    return [_parse_eli_ref(ref["id"]) for ref in refs]


def _version_sort_key(meta: dict) -> str:
    """Klucz do chronologicznego sortowania obwieszczeń: `legalStatusDate`
    kiedy jest ustawione, inaczej `announcementDate` (data ogłoszenia SAMEGO
    obwieszczenia -- niezawodnie uzupełniona nawet w najstarszych wpisach, w
    przeciwieństwie do `legalStatusDate`, i bliska rzeczywistemu początkowi
    okresu, w przeciwieństwie do `expirationDate`, który bywa o lata późniejszy
    dla starych, długo obowiązujących wpisów i fałszywie przesuwa je ZA
    późniejsze wpisy z ustawionym `legalStatusDate` -- sprawdzone empirycznie
    na ustawie systemowej: `DU/2009/1585` ma `expirationDate=2013-12-04`, co
    plasowałoby go PO `DU/2013/1442` (`legalStatusDate=2013-10-16`), mimo że
    `DU/2009/1585` jest faktycznie wcześniejszy -- `announcementDate` obu
    (2009-11-10 vs 2013-10-24) sortuje poprawnie). `expirationDate` zostaje
    jako ostateczny fallback, gdyby oba pozostałe pola brakowały. Poleganie
    na kolejności zwróconej przez API jest błędne dla wpisów bez
    `legalStatusDate` (sprawdzone empirycznie: `DU/2009/1585` występuje PRZED
    `DU/2007/74` w surowej liście `references`, mimo że jest chronologicznie
    późniejsze) -- patrz PROGRESS.md, krok 18, znalezisko 1."""
    return meta.get("legalStatusDate") or meta.get("announcementDate") or meta.get("expirationDate") or ""


def compute_version_windows(base_meta: dict, announcement_metas: list[dict]) -> list[dict]:
    """PURE (bez HTTP). Zwraca chronologicznie posortowaną listę okien
    ważności tekstu ustawy: po jednym słowniku na każde PRZESZŁE
    obwieszczenie -- `{'publisher','year','position','valid_from','valid_to',
    'announcement_eli'}` -- plus syntetyczny wpis na końcu,
    `{'is_current': True, 'valid_from': <granica>, 'valid_to': None}`,
    reprezentujący już pobrany bieżący tekst "U" (nieparsowany tu ponownie).

    `valid_from`/`valid_to` każdej wersji to bezpośrednio jej własne pola
    `legalStatusDate`/`expirationDate` z ELI -- CELOWO nie wymuszamy, żeby
    sąsiednie okna stykały się idealnie (empirycznie się nie stykają, różnice
    rzędu dni-tygodni na wszystkich 7 sprawdzonych ustawach, czasem się nawet
    zachodzą -- patrz PROGRESS.md, krok 18, znalezisko 2). To cecha danych
    ELI, nie błąd do naprawienia tutaj -- nie logujemy ostrzeżenia przy
    każdym niedopasowaniu granic, bo to oczekiwany szum, już scharakteryzowany.

    Granica bieżącej wersji: dla chronologicznie najnowszego obwieszczenia
    -- jego `expirationDate`, jeśli ustawione (oznacza, że nawet ono zostało
    już wyprzedzone przez nowelizacje widoczne tylko w żywym tekście "U"),
    w przeciwnym razie jego `legalStatusDate`. Zero obwieszczeń -> granica
    `None` (brak znanej dolnej granicy dla bieżącego tekstu).

    Obwieszczenia z `expirationDate=None` (wciąż oficjalnie obowiązujące, w
    praktyce zawsze najnowsze w łańcuchu -- nic go jeszcze nie zastąpiło) są
    CELOWO pomijane w liście przeszłych okien, nie tylko przy wyliczaniu
    granicy: taki wpis opisuje dokładnie ten sam, wciąż otwarty okres co
    syntetyczny wpis "is_current" (zweryfikowane na ustawie o PPK -- oba
    miałyby `valid_from=2026-02-10, valid_to=None`) -- zwrócenie go też jako
    osobnej "przeszłej" wersji dałoby dwie identyczne, nakładające się na
    siebie wersje tego samego artykułu w indeksie. Jego treść i tak pokrywa
    się (lub prawie pokrywa) z już posiadanym bieżącym tekstem "U", więc nie
    ma potrzeby pobierać go drugi raz jako "historię"."""
    sorted_metas = sorted(announcement_metas, key=_version_sort_key)

    windows = []
    for m in sorted_metas:
        if m.get("expirationDate") is None:
            continue
        windows.append(
            {
                "publisher": m["publisher"],
                "year": m["year"],
                "position": m["pos"],
                "valid_from": m.get("legalStatusDate"),
                "valid_to": m.get("expirationDate"),
                "announcement_eli": m.get("ELI", f"{m['publisher']}/{m['year']}/{m['pos']}"),
            }
        )

    if sorted_metas:
        latest = sorted_metas[-1]
        current_valid_from = latest.get("expirationDate") or latest.get("legalStatusDate")
    else:
        current_valid_from = None

    windows.append({"is_current": True, "valid_from": current_valid_from, "valid_to": None})
    return windows


def fetch_version_windows(base_act: dict) -> tuple[str, str, list[dict]]:
    """I/O wrapper wokół `compute_version_windows()`: pobiera metadane aktu
    bazowego + każdego jego obwieszczenia (`fetch_unified_text_announcements`),
    potem woła czystą `compute_version_windows()`. Zwraca `(base_eli,
    base_title, windows)` -- `base_eli`/`base_title` osobno, żeby
    `process_act_version()` mógł zawsze podpisywać zwrócone artykuły
    tytułem/ELI AKTU BAZOWEGO (tym się cytuje ustawę użytkownikowi), nie
    tytułem/numerem samego obwieszczenia (`_extract_articles()` zwróciłaby
    tytuł obwieszczenia "Obwieszczenie Marszałka Sejmu ..." dla źródła
    historycznego -- pomylenie tych dwóch pól było realnym błędem znalezionym
    przy smoke teście na ustawie o PPK: `eli` poprawnie wskazywał akt bazowy,
    ale `act_title` po cichu wyciekał tytuł obwieszczenia)."""
    base_meta = fetch_meta(base_act)
    base_eli = base_meta.get("ELI", f"{base_act['publisher']}/{base_act['year']}/{base_act['position']}")
    base_title = base_meta.get("title", "")

    announcement_sources = fetch_unified_text_announcements(base_meta)
    announcement_metas = []
    for src in announcement_sources:
        announcement_metas.append(fetch_meta(src))
        time.sleep(0.3)

    windows = compute_version_windows(base_meta, announcement_metas)
    return base_eli, base_title, windows


def process_act_version(
    source: dict, act_short: str, base_eli: str, base_title: str, valid_from, valid_to, announcement_eli
) -> list[dict]:
    """Pobiera i parsuje JEDNĄ historyczną wersję tekstu (obwieszczenie) tym
    samym pipeline'em co `process_act()` (przez `_extract_articles()`). NIE
    zapisuje niczego do `data/raw/` (kolidowałoby z plikami bieżącego tekstu
    już tam obecnymi dla tej samej ustawy) -- zwraca tylko listę artykułów;
    zapis do `data/processed/{short}_history.json` należy do wołającego
    (`download_acts_history.py`).

    Pola `eli`/`act_title` w zwróconych artykułach to zawsze `base_eli`/
    `base_title` (akt bazowy) -- NIE to, co zwróciłaby `_extract_articles()`
    dla samego obwieszczenia -- bo to akt bazowy ma być cytowany
    użytkownikowi, nie techniczny numer/tytuł obwieszczenia (ten trafia
    osobno do pola `announcement_eli`)."""
    label = f"{act_short}@{valid_from or 'current'}"
    _meta, _announcement_title, _announcement_eli, articles, _pdf_bytes, kind, file_name = _extract_articles(
        source, label, HISTORY_TEXT_KINDS
    )
    return build_version_articles(
        source, act_short, base_eli, base_title, valid_from, valid_to, announcement_eli, articles, kind, file_name
    )


def drop_future_wording_duplicates(articles: list[dict]) -> list[dict]:
    """Tekst jednolity w wersji z dnia ogłoszenia (plik "T", historia od
    przeniesienia z projektu prawo polskie, Krok 69) podaje artykuł zmieniony
    przepisem, który jeszcze nie wszedł w życie, DWA razy pod rząd: najpierw
    brzmienie obowiązujące (przypis "W tym brzmieniu obowiązuje do wejścia w
    życie zmiany..."), zaraz potem nowe ("W brzmieniu ustalonym przez...").
    Zostawiamy pierwsze -- obowiązujące w oknie (w plikach "U" tę rolę
    pełniło wycinanie "< >"). Tylko KOLEJNE duplikaty."""
    kept: list[dict] = []
    for a in articles:
        if kept and kept[-1]["article"] == a["article"]:
            continue
        kept.append(a)
    return kept


def build_version_articles(
    source: dict,
    act_short: str,
    base_eli: str,
    base_title: str,
    valid_from,
    valid_to,
    announcement_eli,
    articles: list[dict],
    kind: str,
    file_name: str,
) -> list[dict]:
    """Opakowuje sparsowane artykuły jednej historycznej wersji w rekordy
    `_history.json` -- wydzielone z `process_act_version()`, żeby
    `download_acts_history.py` mógł parsować wersje w `ParsePool`."""
    source_url = (
        f"{API_BASE}/acts/{source['publisher']}/{source['year']}/{source['position']}/text/{kind}/{file_name}"
    )
    # Najstarsze okna nie mają znanego początku (valid_from=None) -- dawniej
    # dostawały sufiks "@current" (mylący, a dla kilku takich okien jednej
    # ustawy, np. 4 w Prawie budowlanym, powtarzający się); projekt prawo polskie, PROGRESS.md Krok 65.
    if valid_from:
        version_suffix = valid_from
    elif valid_to:
        version_suffix = f"do_{valid_to}"
    else:
        version_suffix = "current"
    articles = drop_future_wording_duplicates(articles)
    return [
        {
            "id": f"{act_short}_art_{a['article']}@{version_suffix}",
            "act_short": act_short,
            "act_title": base_title,
            "eli": base_eli,
            "article": a["article"],
            "text": a["text"],
            "source_url": source_url,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "announcement_eli": announcement_eli,
        }
        for a in articles
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Liczba procesów parsujących PDF-y (domyślnie: dobrane do rdzeni i wolnej pamięci maszyny, "
        "patrz default_parse_workers; 1 = bez równoległości).",
    )
    args = parser.parse_args()

    workers = args.workers or default_parse_workers(len(ACTS))
    print(f"[INFO] parsowanie PDF-ów: {workers} proces(ów) równolegle (pobieranie sekwencyjne)")
    all_chunks = []
    with ParsePool(workers) as pool:
        pending = []
        for act in ACTS:
            meta, title, eli, pdf_bytes, kind, file_name = _fetch_source(act, act["short"])
            save_act_raw(act, meta, pdf_bytes)
            pending.append((act, title, eli, kind, file_name, pool.submit(pdf_bytes, act["short"])))
            time.sleep(1)  # nie bombardujemy publicznego API bez potrzeby
        for act, title, eli, kind, file_name, future in pending:
            all_chunks.extend(write_act_chunks(act, title, eli, kind, file_name, future.result())["chunks"])

    out_path = PROCESSED_DIR / "all_articles.json"
    out_path.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRazem: {len(all_chunks)} fragmentów (artykułów) zapisanych w {out_path}")

if __name__ == "__main__":
    main()
