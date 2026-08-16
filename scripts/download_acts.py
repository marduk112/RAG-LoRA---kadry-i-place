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

import io
import json
import re
import time
from collections import Counter
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
    re.compile(r"^s\.\s*\d+/\d+$"),
]
SECTION_HEADING_PATTERNS = [
    re.compile(r"^DZIA[ŁL]\b", re.IGNORECASE),
    re.compile(r"^Rozdzia[łl]\b", re.IGNORECASE),
    re.compile(r"^Tytu[łl]\b", re.IGNORECASE),
]

ARTICLE_SPLIT_RE = re.compile(r"(?m)^Art\.\s*(\d+[a-ząćęłńóśźż]{0,3})\.\s*")

SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
SUPERSCRIPT_SIZE_RATIO = 0.85  # znak uznajemy za indeks górny, gdy jego czcionka jest mniejsza niż ten ułamek rozmiaru bazowego strony


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
    ten sam fragment usuwa `strip_not_yet_in_force_text`), żeby liczba
    wykrytych tutaj nagłówków zgadzała się z liczbą artykułów po tamtej
    poprawce. Używana wyłącznie do KOREKTY numeracji już wyodrębnionych
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
        if in_not_yet_in_force or i >= n - 4 or not (
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


def fetch_meta(act: dict) -> dict:
    url = f"{API_BASE}/acts/{act['publisher']}/{act['year']}/{act['position']}"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def pick_text_file(meta: dict) -> tuple[str, str]:
    """Wybiera plik z tekstem: wolimy 'U' (ujednolicony), potem 'O' (oryginał)."""
    texts = {t["type"]: t["fileName"] for t in meta.get("texts", [])}
    for kind in ("U", "O"):
        if kind in texts:
            return kind, texts[kind]
    raise RuntimeError(f"Brak znanego typu tekstu w metadanych: {texts}")


def fetch_pdf_bytes(act: dict, kind: str, file_name: str) -> bytes:
    url = f"{API_BASE}/acts/{act['publisher']}/{act['year']}/{act['position']}/text/{kind}/{file_name}"
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.content


def pdf_to_clean_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    lines: list[str] = []
    for page in reader.pages:
        for line in page.extract_text().split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if any(p.match(stripped) for p in HEADER_FOOTER_PATTERNS):
                continue
            if any(p.match(stripped) for p in SECTION_HEADING_PATTERNS):
                continue
            # PDF dzieli słowa na końcu wyjustowanej linii dywizem
            # (np. "zaokrąg-" / "la się" -> "zaokrągla się"); sklejamy je.
            if lines and lines[-1].endswith("-") and stripped[:1].islower():
                lines[-1] = lines[-1][:-1] + stripped
            else:
                lines.append(stripped)
    return "\n".join(lines)


NOT_YET_IN_FORCE_RE = re.compile(r"<.*?>", re.DOTALL)
SUPERSEDED_STILL_VALID_RE = re.compile(r"\[(.*?)\]", re.DOTALL)


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


def process_act(act: dict) -> dict:
    print(f"[{act['short']}] pobieram metadane...")
    meta = fetch_meta(act)
    title = meta.get("title", "")
    eli = meta.get("ELI", f"{act['publisher']}/{act['year']}/{act['position']}")
    print(f"[{act['short']}] {title}  (status: {meta.get('status')})")

    kind, file_name = pick_text_file(meta)
    print(f"[{act['short']}] pobieram tekst typu '{kind}' ({file_name})...")
    pdf_bytes = fetch_pdf_bytes(act, kind, file_name)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / f"{act['short']}.pdf").write_bytes(pdf_bytes)
    (RAW_DIR / f"{act['short']}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[{act['short']}] wyciągam i czyszczę tekst z PDF...")
    clean_text = pdf_to_clean_text(pdf_bytes)
    clean_text = strip_not_yet_in_force_text(clean_text)
    clean_text = recover_midtext_superscript_headers(clean_text)

    print(f"[{act['short']}] tnę na artykuły...")
    articles = parse_articles(clean_text)
    print(f"[{act['short']}] znaleziono {len(articles)} artykułów")

    print(f"[{act['short']}] sprawdzam numerację pod kątem indeksów górnych...")
    detected_numbers = find_article_numbers_pdfplumber(pdf_bytes)
    if len(detected_numbers) == len(articles):
        fixed = 0
        for article, detected in zip(articles, detected_numbers):
            if detected != article["article"]:
                article["text"] = article["text"].replace(f"Art. {article['article']}.", f"Art. {detected}.", 1)
                article["article"] = detected
                fixed += 1
        if fixed:
            print(f"[{act['short']}] poprawiono numerację indeksu górnego w {fixed} artykułach")
    else:
        print(
            f"[{act['short']}] UWAGA: liczba nagłówków wykrytych przez pdfplumber "
            f"({len(detected_numbers)}) != liczba artykułów z pypdf ({len(articles)}) "
            "-- pomijam korektę indeksu górnego dla tego aktu, numeracja zostaje jak z pypdf"
        )

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


def main():
    all_chunks = []
    for act in ACTS:
        result = process_act(act)
        all_chunks.extend(result["chunks"])
        time.sleep(1)  # nie bombardujemy publicznego API bez potrzeby

    out_path = PROCESSED_DIR / "all_articles.json"
    out_path.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRazem: {len(all_chunks)} fragmentów (artykułów) zapisanych w {out_path}")


if __name__ == "__main__":
    main()
