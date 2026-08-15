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
from pathlib import Path

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

ARTICLE_SPLIT_RE = re.compile(r"(?m)^Art\.\s*(\d+[a-z]{0,3})\.\s*")


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

    print(f"[{act['short']}] tnę na artykuły...")
    articles = parse_articles(clean_text)
    print(f"[{act['short']}] znaleziono {len(articles)} artykułów")

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
