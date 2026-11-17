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

Lista aktów do pobrania (z podziałem na dziedziny prawa) jest w
data/acts_config.json -- nie w tym pliku.

Uruchomienie:
    python scripts/download_acts.py                    # wszystkie akty z konfiguracji
    python scripts/download_acts.py --domain podatkowe  # tylko jedna dziedzina
    python scripts/download_acts.py --acts kodeks_pracy,ustawa_o_vat
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
import pypdf._encryption as _pypdf_encryption
from pypdf import PdfReader
from pypdf._crypt_providers._fallback import CryptRC4 as _FallbackCryptRC4

# Niektóre historyczne PDF-y Kancelarii Sejmu (np. prawo_budowlane,
# obwieszczenie DU/2000/1126) mają szyfrowanie RC4 z kluczem o długości,
# której backend `cryptography` używany domyślnie przez pypdf nie
# akceptuje (`ValueError: Invalid key size (104) for RC4`), mimo że sam
# algorytm RC4 działa z dowolną długością klucza -- to nadgorliwa
# walidacja tego konkretnego backendu, nie realne ograniczenie formatu
# PDF. Podmieniamy WYŁĄCZNIE implementację RC4 na czysto pythonową z
# `pypdf._crypt_providers._fallback` (bez tej walidacji, identyczny
# algorytm) -- szyfrowanie AES nadal korzysta z `cryptography`,
# nienaruszone. Nie wymaga instalowania/odinstalowywania żadnych
# pakietów we współdzielonym `.venv` (patrz PROGRESS.md, Krok 19).
_pypdf_encryption.CryptRC4 = _FallbackCryptRC4

API_BASE = "https://api.sejm.gov.pl/eli"
HEADERS = {"User-Agent": "kadry-plyace-asystent/0.1 (lokalny projekt edukacyjny)"}

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ACTS_CONFIG_PATH = ROOT / "data" / "acts_config.json"


def load_acts_config(domain: str | None = None, shorts: set[str] | None = None) -> list[dict]:
    """Wczytuje listę aktów z data/acts_config.json -- płaska lista wpisów
    `{short, publisher, year, position, domains}`, gdzie `domains` to lista
    dziedzin prawa, do których dany akt należy (akt może należeć do więcej
    niż jednej, np. ustawa o PIT jest jednocześnie w 'kadry_i_place' i
    'podatkowe' -- stąd lista, a nie zagnieżdżanie wpisów per dziedzina,
    żeby taki akt nie był duplikowany w konfiguracji).

    Żeby dodać kolejny akt: znajdź jego identyfikator (np. przez
    /eli/acts/search?title=...) i dopisz wpis do tego pliku.

    `domain` (opcjonalnie) filtruje do aktów, których `domains` zawiera tę
    wartość. `shorts` (opcjonalnie) filtruje do podanego zbioru skrótów.
    Oba filtry można łączyć (logiczne AND)."""
    acts = json.loads(ACTS_CONFIG_PATH.read_text(encoding="utf-8"))
    if domain is not None:
        acts = [a for a in acts if domain in a.get("domains", [])]
    if shorts is not None:
        acts = [a for a in acts if a["short"] in shorts]
    return acts


# Kompatybilność wsteczna dla modułów importujących `ACTS` bezpośrednio
# (download_acts_history.py, check_acts_freshness.py) -- pełna,
# niefiltrowana lista.
ACTS = load_acts_config()

HEADER_FOOTER_PATTERNS = [
    re.compile(r"^©Kancelaria Sejmu"),
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
    # Ten sam stempel daty wygenerowania w starszym formacie (teksty
    # ujednolicone ~2017-2021, np. "09.05.2019" co stronę) -- trafiał do
    # treści artykułów, kilkanaście tys. wystąpień w historii (PROGRESS.md Krok 65).
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

ARTICLE_SPLIT_RE = re.compile(r"(?m)^Art\.\s*(\d+[a-ząćęłńóśźż]{0,3}[⁰¹²³⁴⁵⁶⁷⁸⁹]{0,2})\.\s*")

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
        # Alternatywna notacja indeksu górnego -- niektóre tekst-jednolite
        # PDF-y kodują go zwykłymi nawiasami kwadratowymi w warstwie
        # tekstowej zamiast mniejszym rozmiarem czcionki (np. "Art. 5[1]."
        # = "Art. 5¹."), potwierdzone bezpośrednio w PDF-ach: Kodeks
        # cywilny (~230 wystąpień, np. "Art. 22[1]." = art. 22¹
        # "konsument"), Kodeks pracy (6, np. "Art. 18[3d]."), Prawo
        # budowlane (2: "Art. 5[1]."/"Art. 5[2]."). Bez tego rozpoznania
        # `strip_not_yet_in_force_text` (który normalnie zdejmuje "[ ]" z
        # zupełnie innego powodu -- stary, wciąż ważny tekst) spłaszcza je
        # do zwykłych cyfr, które potrafią kolidować z zupełnie innym,
        # prawdziwym artykułem o tym samym spłaszczonym numerze
        # (potwierdzony przypadek: Prawo budowlane art. 5[1]/5[2] kolidowały
        # z późniejszymi, osobnymi art. 51/52 -- patrz PROGRESS.md).
        bracket_super = ""
        if j < n and chars[j]["text"] == "[":
            k = j + 1
            inner = []
            while k < n and chars[k]["text"] != "]":
                inner.append(chars[k]["text"])
                k += 1
            if k < n and chars[k]["text"] == "]":
                bracket_digits = "".join(c for c in inner if c.isdigit())
                bracket_letters = "".join(c for c in inner if c.isalpha())
                if bracket_digits:
                    bracket_super = bracket_digits.translate(SUPERSCRIPT_DIGITS) + bracket_letters
                    j = k + 1
        # opcjonalny sufiks literowy (np. "30a", "18c") -- bez indeksu górnego,
        # ale MUSI trafić do wyniku, inaczej "183a"/"183b"/"183c" zlewają się w jedno
        letter_suffix = ""
        while j < n and chars[j]["text"].isalpha() and chars[j]["text"].islower():
            letter_suffix += chars[j]["text"]
            j += 1
        # Indeks górny PO sufiksie literowym (np. "Art. 5b¹." -- Prawo
        # energetyczne ma 17 takich nagłówków: 4e¹, 5a¹, 5b¹-5b⁴, 9d¹-9d³,
        # 9h¹-9h³, 43f¹...). Bez tego nagłówek nie kończy się tu kropką i
        # byłby pominięty -- zgodnie z pypdf, więc liczby się zgadzały, a
        # cały artykuł znikał wchłonięty przez poprzedni (PROGRESS.md, Krok 60).
        letter_super = ""
        if letter_suffix:
            while (
                j < n and chars[j]["text"].isdigit()
                and chars[j]["size"] < body_size * SUPERSCRIPT_SIZE_RATIO
            ):
                letter_super += chars[j]["text"]
                j += 1
        if digit_chars and j < n and chars[j]["text"] == ".":
            normal = "".join(c["text"] for c in digit_chars if c["size"] >= body_size * SUPERSCRIPT_SIZE_RATIO)
            super_ = "".join(c["text"] for c in digit_chars if c["size"] < body_size * SUPERSCRIPT_SIZE_RATIO)
            if normal:
                numbers.append(
                    normal + super_.translate(SUPERSCRIPT_DIGITS) + bracket_super + letter_suffix
                    + letter_super.translate(SUPERSCRIPT_DIGITS)
                )
            i = j + 1
        else:
            i += 1
    return numbers


# "\d{2}\.\d{2}\.\d{4}" -- stempel daty w starszym formacie, ten sam co w
# HEADER_FOOTER_PATTERNS: bez niego kontekst z pdfplumber zawierał stempel,
# którego już nie ma w clean_text, i korekta cicho przepadała (Krok 65).
FOOTER_TOKEN_RE = re.compile(r"^(©Kancelaria|Sejmu|s\.|\d+/\d+|\d{4}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{4})$")
USTEP_CONTEXT_WORDS = 4  # ile poprzedzających słów bierzemy jako "odcisk palca" do zlokalizowania miejsca w clean_text
DIFFERENT_ACT_CONTEXT_RE = re.compile(
    r"(?i)(ustawy z dnia|kodeksu|rozporz[ąa]dzenia|zmienianej|wymienionej|"
    r"odr[ęe]bnej ustawy|innej ustawy|przepis[oó]w odr[ęe]bnych|niniejszej ustawy)"
)
DIFFERENT_ACT_FOLLOWING_LOOKAHEAD = 150  # okno sprawdzane pod kątem nazwy INNEGO aktu (Krok 49)
# Skróty kończące się kropką, które NIE są końcem zdania (więc nie zaczynają nowego
# ustępu) -- odróżnia np. "godz." (godzina) od prawdziwej kropki kończącej zdanie.
# Świadomie krótka, oparta wyłącznie na potwierdzonych przypadkach (Krok 49: zapis
# godzin "godz. 600 do godz. 1800" mylnie wykryty jako ustęp "6⁰⁰"/"18⁰⁰") -- nie
# rozszerzać spekulacyjnie bez nowego potwierdzonego przypadku.
NON_SENTENCE_ENDING_ABBREVIATIONS = {"godz."}


def find_ustep_superscript_corrections_pdfplumber(pdf_bytes: bytes) -> list[tuple[list[str], str, str]]:
    """Wykrywa spłaszczone superskrypty NA POZIOMIE USTĘPU (np. "21."
    zamiast "2¹.") przez geometrię PDF -- ten sam mechanizm co
    `find_article_numbers_pdfplumber` (Krok 18), ale dla numeracji
    ustępów/paragrafów WEWNĄTRZ treści artykułu, nie numerów artykułów w
    nagłówku (Krok 20/47 -- `autorskie` art. 1 ust. 2¹, potwierdzone
    potem jako systemowy problem w 7 z 15 dziedzin, patrz PROGRESS.md).

    W przeciwieństwie do numerów artykułów (zawsze poprzedzonych stałym
    "Art. "), ustępy nie mają jednego stałego prefiksu -- w zależności od
    aktu bywają zapisane jako "§ N." albo goły "N." zaraz po kropce
    kończącej poprzedni ustęp. To z natury bardziej niejednoznaczne, więc
    funkcja świadomie ogranicza się do JEDNOZNACZNYCH przypadków:
    - wyklucza nagłówki "Art. N" (już obsłużone gdzie indziej),
    - wyklucza cytaty w prozie do numeru INNEGO artykułu (poprzedzone
      "art. "/"Art. " w środku zdania, albo zakresem "N–M" / "N-M") --
      to osobny, nienaprawiony tu problem (spłaszczony numer artykułu
      cytowany gdzie indziej w tekście, wymaga innego mechanizmu niż
      korekta pozycyjna, bo trzeba by znać poprawny numer z INNEGO
      miejsca w dokumencie),
    - wymaga, żeby cyfry o mieszanej wielkości czcionki bezpośrednio
      poprzedzał sam "§" albo kropka kończąca poprzednie zdanie --
      wszystko inne (np. przypadkowy zbieg okoliczności typu zakresu
      godzin "21:00 a 7:00" błędnie wykryty jako cyfry o mieszanej
      wielkości czcionki) zostaje świadomie pominięte.

    Do Kroku 49 funkcja wymagała też, żeby liczba kończyła się dosłowną
    kropką (bo tak wygląda nagłówek nowego ustępu) -- to niepotrzebnie
    wykluczało CYTATY do paragrafu innego artykułu w środku zdania,
    zapisane tą samą konwencją "§ N" ale kończące się przecinkiem czy
    średnikiem (potwierdzony przypadek: Kodeks pracy art. 29 cytuje
    "art. 25 § 22," -- powinno być "§ 2²," -- kropka po liczbie nigdy
    nie występuje, bo zdanie się tu nie kończy). Naprawione tym samym
    negatywnym lookahead `(?!\\d)`, którego już używa
    `apply_flattened_superscript_corrections` dla cytatów "art. N" --
    liczba musi po prostu nie być częścią dłuższej liczby, nic więcej.

    To rozluźnienie ujawniło DWA nowe źródła fałszywych trafień
    (znalezione i naprawione w tej samej sesji, Krok 49), oba dlatego, że
    "kończy się kropką" samo w sobie nie odróżnia zdania od skrótu:
    - **skróty niekończące zdania** (np. "godz." w zapisie godzin "godz.
      600 do godz. 1800") -- "godz." też kończy się kropką, więc bez
      dodatkowego wykluczenia wyglądało jak prawdziwy ustęp. Wykluczone
      przez `NON_SENTENCE_ENDING_ABBREVIATIONS` -- świadomie krótka,
      oparta wyłącznie na potwierdzonym przypadku, nie rozszerzana
      spekulacyjnie na inne skróty bez nowych dowodów.
    - **cytaty "§ N"/"ust. N" do INNEJ ustawy** (np. "art. 4779 § 31
      ustawy z dnia..." w ustawie systemowej, cytujący Kodeks pracy) --
      ten sam problem co przy cytatach "art. N"
      (`find_citation_superscript_corrections_pdfplumber`), tylko że tu
      poprzedzający sygnał to "§"/"." zamiast "art.", więc dawniejsza
      logika `is_citation` (sprawdzająca wyłącznie "art."/"Art.") tego
      nie łapała. Wykluczone tym samym `DIFFERENT_ACT_CONTEXT_RE` i tym
      samym oknem "do przodu", co w detektorze cytatów.

    Zwraca listę (słowa_kontekstu, spłaszczony_numer, poprawiony_numer) w
    KOLEJNOŚCI występowania w dokumencie -- `słowa_kontekstu` to kilka
    poprzedzających słów (bez artefaktów nagłówka/stopki strony
    Kancelarii Sejmu, które i tak znikają z `clean_text`), używane przez
    `apply_flattened_superscript_corrections` jako "odcisk palca" do
    zlokalizowania właściwego miejsca -- nie da się dopasować pozycyjnie
    jak przy artykułach (liczba ustępów na artykuł jest zmienna), a
    dosłowne dopasowanie tekstu zawodzi, bo `pdfplumber` (ten kod) i
    `pypdf` (`pdf_to_clean_text`) inaczej renderują białe znaki na
    złamaniach linii -- stąd dopasowanie ma być odporne na to w
    `apply_flattened_superscript_corrections` (dopasowanie `\\s+` zamiast
    dosłownej spacji)."""
    corrections: list[tuple[list[str], str, str]] = []
    chars: list[dict] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            chars.extend(page.chars)
    if not chars:
        return corrections
    body_size = Counter(round(c["size"], 1) for c in chars).most_common(1)[0][0]
    n = len(chars)
    i = 0
    while i < n:
        if not chars[i]["text"].isdigit():
            i += 1
            continue
        start = i
        j = i
        while j < n and chars[j]["text"].isdigit():
            j += 1
        digit_chars = chars[start:j]
        normal_digits = "".join(c["text"] for c in digit_chars if c["size"] >= body_size * SUPERSCRIPT_SIZE_RATIO)
        super_digits = "".join(c["text"] for c in digit_chars if c["size"] < body_size * SUPERSCRIPT_SIZE_RATIO)
        if normal_digits and super_digits:
            preceding = "".join(c["text"] for c in chars[max(0, start - 80): start])
            preceding_stripped = preceding.rstrip()
            preceding_words = preceding_stripped.split()
            is_art_header = preceding_stripped.endswith("Art.") or preceding_stripped.endswith("Art")
            is_citation = (
                bool(re.search(r"(?i)\bart\.?\s*$", preceding_stripped))
                or preceding_stripped.endswith("–")
                or preceding_stripped.endswith("-")
            )
            is_non_sentence_abbreviation = bool(preceding_words) and preceding_words[-1] in NON_SENTENCE_ENDING_ABBREVIATIONS
            is_true_ustep = (
                not is_art_header and not is_citation and not is_non_sentence_abbreviation
                and (preceding_stripped.endswith("§") or preceding_stripped.endswith("."))
            )
            if is_true_ustep:
                following = "".join(c["text"] for c in chars[j: j + DIFFERENT_ACT_FOLLOWING_LOOKAHEAD])
                is_different_act = bool(DIFFERENT_ACT_CONTEXT_RE.search(preceding)) or bool(
                    DIFFERENT_ACT_CONTEXT_RE.search(following)
                )
                if not is_different_act:
                    flattened = normal_digits + super_digits
                    corrected = normal_digits + super_digits.translate(SUPERSCRIPT_DIGITS)
                    words = [w for w in preceding_words if not FOOTER_TOKEN_RE.match(w)]
                    words = words[-USTEP_CONTEXT_WORDS:]
                    if words:
                        corrections.append((words, flattened, corrected))
        i = j + 1
    return corrections


CITATION_CONTEXT_WORDS = 4


def find_citation_superscript_corrections_pdfplumber(pdf_bytes: bytes) -> list[tuple[list[str], str, str]]:
    """Wykrywa spłaszczone superskrypty w CYTATACH W PROZIE do numeru
    artykułu tej SAMEJ ustawy (np. "art. 1291" zamiast "art. 129¹" w
    zdaniu "Przepisy art. 129¹ stosuje się...") -- Krok 49, następstwo
    Kroku 48. Sonda przez wszystkie 15 dziedzin (po naprawie ustępów)
    znalazła nie 47, a 642 takich przypadków po odsianiu prawdziwych
    nagłówków -- ale okazało się, że to NIE jest jednorodna kategoria:

    - część to faktycznie cytaty do artykułu TEJ SAMEJ ustawy (bezpieczne
      do naprawy -- ten sam mechanizm co ustępy w Kroku 48, bo geometria
      PDF-a bezpośrednio i niezależnie od kontekstu koduje, które cyfry
      są naprawdę superskryptem, tak samo dla cytatu jak dla nagłówka),
    - część to cytaty do artykułu INNEJ ustawy (np. "art. 775 § 2 ustawy
      z dnia 26 czerwca 1974" -- Kodeksu pracy, cytowane z zupełnie innego
      aktu) -- teoretycznie też dałoby się naprawić (mielibyśmy dokładnie
      taki sam sygnał geometrii dla tamtego aktu, GDYBY to była jedna z
      15 zaindeksowanych ustaw), ale wymaga innego mechanizmu (przeszukania
      metadanych INNEGO aktu) -- świadomie poza zakresem tej funkcji,
    - część to ZNACZNIKI PRZYPISÓW, nie numery artykułów w ogóle (np. "w
      art. 239⁷) i art. 240" -- czytane jako "art. 239, przypis 7)", nie
      jako prawdziwy "art. 239⁷") -- zaklasyfikowanie tego jako superskryptu
      artykułu i "naprawienie" wstawiłoby FAŁSZYWY numer artykułu, więc
      to NIE jest kwestia braku danych, tylko realne ryzyko błędu.

    Użytkownik świadomie wybrał najbezpieczniejszy zakres: naprawiać
    WYŁĄCZNIE cytaty do tej samej ustawy. Odsiewanie:
    - wymaga małej litery "art."/"art" bezpośrednio przed liczbą (wielka
      litera "Art." to nagłówek, już obsłużony przez
      `find_article_numbers_pdfplumber`),
    - wyklucza znaczniki przypisów: jeśli znak TUŻ PO liczbie to ")",
      pomijamy (przypisy w tych PDF-ach mają format "N)", prawdziwy
      cytat artykułu nigdy nie kończy się gołym ")"),
    - wyklucza cytaty do innej ustawy: sprawdza ok. 45 znaków PRZED "art."
      i 150 znaków PO liczbie (`DIFFERENT_ACT_FOLLOWING_LOOKAHEAD`) pod kątem
      fraz nazywających inny akt (`DIFFERENT_ACT_CONTEXT_RE`) -- szerokie okno
      "po" jest konieczne, bo nazwa aktu cytowanego zwykle pada dopiero
      pod koniec długiego zdania wyliczającego kilka artykułów (potwierdzone
      empirycznie: okno 60 znaków przepuszczało 11 fałszywych "bezpiecznych"
      przypadków, które dopiero szersze okno poprawnie wyłapało).

    To CELOWO konserwatywne -- woli pominąć prawdziwy, bezpieczny
    przypadek (np. gdy nazwa innej ustawy przypadkiem pojawia się w
    pobliżu z zupełnie innego powodu, jak wspólny przypis o historii
    zmian tej SAMEJ ustawy) niż zaryzykować naprawienie czegoś, co nie
    powinno być tknięte -- w tym kierunku błędu nic złego się nie dzieje
    (numeracja zostaje po prostu jak z pypdf, tak jak przed tą poprawką).

    Zwraca listę w tym samym kształcie co
    `find_ustep_superscript_corrections_pdfplumber`, do użycia z
    `apply_flattened_superscript_corrections`."""
    corrections: list[tuple[list[str], str, str]] = []
    chars: list[dict] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            chars.extend(page.chars)
    if not chars:
        return corrections
    body_size = Counter(round(c["size"], 1) for c in chars).most_common(1)[0][0]
    n = len(chars)
    i = 0
    while i < n:
        if not chars[i]["text"].isdigit():
            i += 1
            continue
        start = i
        j = i
        while j < n and chars[j]["text"].isdigit():
            j += 1
        digit_chars = chars[start:j]
        normal_digits = "".join(c["text"] for c in digit_chars if c["size"] >= body_size * SUPERSCRIPT_SIZE_RATIO)
        super_digits = "".join(c["text"] for c in digit_chars if c["size"] < body_size * SUPERSCRIPT_SIZE_RATIO)
        if normal_digits and super_digits:
            preceding = "".join(c["text"] for c in chars[max(0, start - 45): start])
            following = "".join(c["text"] for c in chars[j: j + DIFFERENT_ACT_FOLLOWING_LOOKAHEAD])
            preceding_stripped = preceding.rstrip()
            is_real_header = preceding_stripped.endswith("Art.") or preceding_stripped.endswith("Art")
            is_citation = not is_real_header and (
                preceding_stripped.endswith("art.") or preceding_stripped.endswith("art")
            )
            if is_citation:
                is_footnote_marker = following[:1] == ")"
                is_different_act = bool(DIFFERENT_ACT_CONTEXT_RE.search(preceding)) or bool(
                    DIFFERENT_ACT_CONTEXT_RE.search(following)
                )
                if not is_footnote_marker and not is_different_act:
                    flattened = normal_digits + super_digits
                    corrected = normal_digits + super_digits.translate(SUPERSCRIPT_DIGITS)
                    words = [w for w in preceding_stripped.split() if not FOOTER_TOKEN_RE.match(w)]
                    words = words[-CITATION_CONTEXT_WORDS:]
                    if words:
                        corrections.append((words, flattened, corrected))
        i = j + 1
    return corrections


CITED_ACT_DATE_RE = re.compile(r"ustawy z dnia (\d{1,2} [a-ząćęłńóśźż]+ \d{4}) r\.")


def _act_date_phrase(title: str) -> str | None:
    """Wyciąga i normalizuje frazę 'DD MIESIĄC RRRR' z tytułu aktu (np.
    'Ustawa z dnia 26 czerwca 1974 r. Kodeks pracy.' -> '26 czerwca 1974') --
    data uchwalenia jednoznacznie identyfikuje polską ustawę wśród wszystkich
    innych (dwa różne akty tego projektu nigdy nie dzielą tej samej daty),
    używana do rozpoznania CYTOWANEGO aktu w cytatach w prozie do INNEJ
    ustawy (Krok 50, patrz `find_cross_act_citation_corrections_pdfplumber`)."""
    m = re.search(r"z dnia (\d{1,2} [a-ząćęłńóśźż]+ \d{4}) r\.", title)
    return m.group(1) if m else None


def _known_act_dates() -> dict[str, str]:
    """Mapa {data uchwalenia -> act_short} zbudowana z tytułów zapisanych w
    `data/raw/{short}_meta.json` dla wszystkich aktów z `acts_config.json` --
    to jedyny artefakt z `data/raw/` faktycznie commitowany do repo (patrz
    .gitignore), więc ta mapa działa niezależnie od tego, które akty zostały
    pobrane w BIEŻĄCYM uruchomieniu (`--domain`/`--acts`). Akt bez zapisanych
    metadanych (nigdy jeszcze nie pobrany, świeży klon repo) po prostu nie
    zostanie rozpoznany jako cel cytatu -- bezpiecznie, żadna korekta nie
    zostanie podjęta dla cytatów do niego, tak jak dotychczas."""
    mapping: dict[str, str] = {}
    for act in load_acts_config():
        meta_path = RAW_DIR / f"{act['short']}_meta.json"
        if not meta_path.exists():
            continue
        try:
            title = json.loads(meta_path.read_text(encoding="utf-8")).get("title", "")
        except (json.JSONDecodeError, OSError):
            continue
        date_phrase = _act_date_phrase(title)
        if date_phrase:
            mapping[date_phrase] = act["short"]
    return mapping


_CITED_ACT_NUMBERS_CACHE: dict[str, set[str]] = {}


def _cited_act_article_numbers(act_short: str) -> set[str]:
    """Zwraca zbiór PRAWDZIWYCH numerów artykułów (z indeksem górnym jako
    prawdziwym znakiem Unicode) CYTOWANEGO aktu -- ostateczny bezpiecznik
    `find_cross_act_citation_corrections_pdfplumber`: korekta cytatu do
    innej ustawy stosowana jest TYLKO, gdy naprawiony numer faktycznie
    występuje na tej liście. Woli już pobrany PDF (`data/raw/{short}.pdf`);
    jeśli go brak (PDF-y nie są commitowane, patrz .gitignore), pobiera go
    jednorazowo z ELI API na podstawie wpisu w `acts_config.json`. Wynik
    cache'owany w pamięci procesu -- ten sam cytowany akt bywa cytowany
    wielokrotnie w jednym uruchomieniu (np. Kodeks pracy z kilkunastu miejsc
    ustawy o rynku pracy)."""
    if act_short in _CITED_ACT_NUMBERS_CACHE:
        return _CITED_ACT_NUMBERS_CACHE[act_short]
    pdf_path = RAW_DIR / f"{act_short}.pdf"
    if pdf_path.exists():
        pdf_bytes = pdf_path.read_bytes()
    else:
        acts = load_acts_config(shorts={act_short})
        if not acts:
            _CITED_ACT_NUMBERS_CACHE[act_short] = set()
            return set()
        meta = fetch_meta(acts[0])
        kind, file_name = pick_text_file(meta)
        pdf_bytes = fetch_pdf_bytes(acts[0], kind, file_name)
    numbers = set(find_article_numbers_pdfplumber(pdf_bytes))
    _CITED_ACT_NUMBERS_CACHE[act_short] = numbers
    return numbers


def find_cross_act_citation_corrections_pdfplumber(
    pdf_bytes: bytes, own_act_short: str
) -> list[tuple[list[str], str, str]]:
    """Wykrywa spłaszczone superskrypty w cytatach w prozie do numeru
    artykułu INNEJ ustawy (Krok 50, dokończenie Kroku 49) -- np. "art. 775
    § 2 ustawy z dnia 26 czerwca 1974 r. − Kodeks pracy" powinno być "art.
    77⁵ § 2 ...". `find_citation_superscript_corrections_pdfplumber` (Krok
    49) świadomie WYKLUCZA takie przypadki (`DIFFERENT_ACT_CONTEXT_RE`) --
    ta funkcja dopełnia to celowo pozostawione ograniczenie, ale tylko gdy
    WSZYSTKIE trzy warunki są spełnione:
    1. w kontekście cytatu (45 znaków przed "art.", 150 znaków po liczbie)
       da się jednoznacznie zidentyfikować CYTOWANY akt po jego dacie
       uchwalenia ("ustawy z dnia D MIESIĄC RRRR r.") -- ta sama fraza, którą
       akt podaje we własnym tytule (`_act_date_phrase`),
    2. ten akt jest jednym z aktów już znanych temu projektowi
       (`acts_config.json`) -- inaczej nie mamy jak sprawdzić jego
       prawdziwej numeracji,
    3. poprawiony numer (z ewentualnym sufiksem literowym tuż po cyfrach,
       np. "18³a") FAKTYCZNIE występuje w prawdziwej liście numerów
       artykułów tego aktu (`_cited_act_article_numbers`) -- ostateczny
       bezpiecznik: nawet gdyby rozpoznanie aktu z (1) było przypadkowe/
       błędne, korekta i tak nie zostanie zastosowana, chyba że naprawdę
       odpowiada istniejącemu artykułowi.

    Potwierdzone bezpośrednio na realnych danych (Krok 50): warunek 3 sam
    w sobie odsiał jeden potwierdzony fałszywy trop (`prawo_autorskie`,
    "art. 9915" -- to własny, tej samej ustawy artykuł 99¹⁵, błędnie
    wykluczony przez zbyt szeroki lookahead `DIFFERENT_ACT_CONTEXT_RE`,
    bo w tym samym zdaniu, kilka słów dalej, wspomniana jest KPA z
    zupełnie innego powodu proceduralnego -- "99¹⁵" nie istnieje w
    prawdziwej numeracji KPA, więc korekta świadomie nie została podjęta).

    Bez potwierdzenia wszystkich trzech warunków kandydat jest bezpiecznie
    pomijany -- dokładnie tak samo, jak przed tym mechanizmem. Świadomie
    NIE obejmuje cytatów na poziomie USTĘPU
    (`find_ustep_superscript_corrections_pdfplumber`) -- sonda na
    wszystkich 22 aktach (Krok 50) nie znalazła tam ANI JEDNEGO
    prawdziwego, możliwego do naprawienia przypadku: wszystkie tam
    wykluczone kandydaty to (a) cytaty do aktów spoza tego projektu (głównie
    Kodeks postępowania cywilnego, nieindeksowany) albo (b) błędne
    wykluczenia prawdziwych cytatów W TEJ SAMEJ ustawie (ten sam efekt
    uboczny szerokiego lookaheadu jak w przykładzie wyżej) -- osobny,
    nienaprawiony tu problem samego `DIFFERENT_ACT_CONTEXT_RE`, nie coś do
    rozwiązania mechanizmem cross-act.

    Świadomie NIE rozpoznaje krótkich form bez daty (np. samo "Kodeksu
    pracy" w zdaniu, gdy pełne "ustawy z dnia..." padło wcześniej w innym
    zdaniu/artykule) -- jeden taki potwierdzony przypadek pozostał
    nienaprawiony (`prawo_ochrony_srodowiska`, "art. 775 § 2 Kodeksu
    pracy"), udokumentowany jako świadomie pozostawiona luka, nie błąd.

    Zwraca listę w tym samym kształcie co inne funkcje
    `find_*_corrections_*` (bez sufiksu literowego w `flattened`/
    `corrected` -- litera zostaje nietknięta w tekście, tak jak już działa
    `apply_flattened_superscript_corrections` dla cytatów tej samej
    ustawy)."""
    corrections: list[tuple[list[str], str, str]] = []
    known_dates = _known_act_dates()
    chars: list[dict] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            chars.extend(page.chars)
    if not chars:
        return corrections
    body_size = Counter(round(c["size"], 1) for c in chars).most_common(1)[0][0]
    n = len(chars)
    i = 0
    while i < n:
        if not chars[i]["text"].isdigit():
            i += 1
            continue
        start = i
        j = i
        while j < n and chars[j]["text"].isdigit():
            j += 1
        digit_chars = chars[start:j]
        normal_digits = "".join(c["text"] for c in digit_chars if c["size"] >= body_size * SUPERSCRIPT_SIZE_RATIO)
        super_digits = "".join(c["text"] for c in digit_chars if c["size"] < body_size * SUPERSCRIPT_SIZE_RATIO)
        if normal_digits and super_digits:
            preceding = "".join(c["text"] for c in chars[max(0, start - 45): start])
            following = "".join(c["text"] for c in chars[j: j + DIFFERENT_ACT_FOLLOWING_LOOKAHEAD])
            preceding_stripped = preceding.rstrip()
            is_real_header = preceding_stripped.endswith("Art.") or preceding_stripped.endswith("Art")
            is_citation = not is_real_header and (
                preceding_stripped.endswith("art.") or preceding_stripped.endswith("art")
            )
            if is_citation:
                is_footnote_marker = following[:1] == ")"
                date_match = CITED_ACT_DATE_RE.search(preceding) or CITED_ACT_DATE_RE.search(following)
                cited_act_short = known_dates.get(date_match.group(1)) if date_match else None
                if not is_footnote_marker and cited_act_short and cited_act_short != own_act_short:
                    letter_suffix = ""
                    for ch in following:
                        if ch.isalpha() and ch.islower():
                            letter_suffix += ch
                        else:
                            break
                    corrected = normal_digits + super_digits.translate(SUPERSCRIPT_DIGITS)
                    full_corrected = corrected + letter_suffix
                    if full_corrected in _cited_act_article_numbers(cited_act_short):
                        flattened = normal_digits + super_digits
                        words = [w for w in preceding_stripped.split() if not FOOTER_TOKEN_RE.match(w)]
                        words = words[-CITATION_CONTEXT_WORDS:]
                        if words:
                            corrections.append((words, flattened, corrected))
        i = j + 1
    return corrections


def apply_flattened_superscript_corrections(
    clean_text: str, corrections: list[tuple[list[str], str, str]], label: str = "indeksu górnego"
) -> str:
    """Stosuje korekty z `find_ustep_superscript_corrections_pdfplumber`
    ALBO `find_citation_superscript_corrections_pdfplumber` na `clean_text`
    (obie zwracają ten sam kształt: lista (słowa_kontekstu, spłaszczony,
    poprawiony)). Każda korekta szuka wzorca regex (poprzedzające słowa
    połączone `\\s+` zamiast dosłownej spacji, żeby przetrwać różnice w
    białych znakach między `pdfplumber` a `pypdf` na złamaniach linii;
    liczba zakończona negatywnym lookahead `(?!\\d)` zamiast wymogu
    konkretnego znaku kończącego, bo ustępy zawsze kończą się kropką, ale
    cytaty w prozie różnie -- kropką, przecinkiem, myślnikiem zakresu,
    spacją przed "ust."/"§") i zamienia TYLKO pierwsze pozostałe
    dopasowanie -- ponieważ korekty są podawane w kolejności występowania
    w dokumencie, a każde zastosowanie "zużywa" jedno dopasowanie, kolejne
    wywołanie trafia we właściwe, następne miejsce w tekście, nawet gdyby
    krótki kontekst nie był globalnie unikalny. Ponieważ lookahead nie
    konsumuje znaku kończącego, zamiana dotyka WYŁĄCZNIE samej liczby --
    cokolwiek po niej następuje (kropka, przecinek, myślnik) zostaje
    nietknięte.

    Brak dopasowania (różnice w dzieleniu wyrazów z łącznikiem, albo
    miejsce leżące wewnątrz already-usuniętego "< >") jest CELOWO
    bezpiecznie pomijane z ostrzeżeniem, a nie traktowane jako błąd --
    zweryfikowane na 7 dziedzinach: ok. 85% zidentyfikowanych przypadków
    ustępów trafia w cel, reszta zostaje nietknięta (nie gorzej niż przed
    tą poprawką), patrz PROGRESS.md, Krok 48/49. `label` trafia tylko do
    komunikatu, żeby dziennik ekstrakcji rozróżniał ustępy od cytatów.

    WAŻNE (błąd znaleziony i naprawiony w Kroku 49, dotyczył też Kroku 48):
    wszystkie dopasowania są wyszukiwane względem NIEZMIENIONEGO,
    oryginalnego `clean_text` (przesuwany kursor `search_from` zapewnia,
    że powtarzający się identyczny krótki kontekst w różnych miejscach
    dokumentu nadal trafia we właściwą, kolejną wersję -- ta sama
    gwarancja co poprzednio), a WSZYSTKIE podmiany są stosowane dopiero
    na końcu, od najdalszej pozycji do najbliższej. Dawniejsza wersja
    mutowała `clean_text` PO KAŻDEJ korekcie z osobna -- co psuło kolejną
    korektę, jeśli jej własny "odcisk palca" (poprzedzające słowa)
    przypadkiem cytował spłaszczony numer WŁAŚNIE poprawiony przez
    poprzednią korektę. Potwierdzony rzeczywisty przypadek: "art. 1291,
    art. 1361" w Prawie własności przemysłowej -- kontekst dla "1361" to
    `['w', 'art.', '1291,', 'art.']`, więc po naprawieniu "1291" na
    "129¹" ten fragment przestawał dosłownie istnieć i druga korekta
    cicho zawodziła, mimo że w oryginalnym tekście oba dopasowania są
    całkowicie poprawne. Wyszukiwanie wszystkiego na niezmienionym
    tekście i stosowanie zamian od końca (żeby wcześniejsze pozycje się
    nie przesunęły) eliminuje tę klasę błędu w całości."""
    fixed = 0
    skipped = 0
    replacements: list[tuple[int, int, str]] = []
    search_from = 0
    for words, flattened, corrected in corrections:
        pattern = re.compile(r"\s+".join(re.escape(w) for w in words) + r"\s+" + re.escape(flattened) + r"(?!\d)")
        m = pattern.search(clean_text, search_from) or pattern.search(clean_text)
        if m:
            replacements.append((m.end() - len(flattened), m.end(), corrected))
            search_from = m.end()
            fixed += 1
        else:
            skipped += 1
    for start, end, corrected in sorted(replacements, key=lambda r: r[0], reverse=True):
        clean_text = clean_text[:start] + corrected + clean_text[end:]
    if fixed:
        print(f"poprawiono numerację {label} w {fixed} miejscach")
    if skipped:
        print(
            f"[UWAGA] nie znaleziono kontekstu dla {skipped} kandydatów na indeks górny "
            f"({label}) -- pominięto bezpiecznie (numeracja zostaje jak z pypdf)"
        )
    return clean_text


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
    linii '1)'" zostawiało jego nagłówek w art. 1 (PROGRESS.md, Krok 63).

    Kancelaria Sejmu drukuje przypisy ("15) Zmiany tekstu jednolitego
    wymienionej ustawy zostały ogłoszone w...") na dole strony, mniejszą
    czcionką (10 pt przy 12 pt tekstu), z numerem jako indeksem górnym
    (~6,5 pt). `pypdf` wypluwa je jako zwykłe linie na końcu tekstu strony,
    więc trafiały do środka artykułu łamanego przez stronę -- potwierdzone w
    co najmniej 129 artykułach 28 ustaw, a numer przypisu wyglądał jak
    kolejny punkt wyliczenia (adapter zacytował nieistniejący "art. 389 pkt
    15" Prawa wodnego, PROGRESS.md Krok 61/63).

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
    historycznych okien, ~75 zdublowanych etykiet, np. KC 2013: 28,
    PROGRESS.md Krok 64). Teraz obie sekwencje dopasowujemy lokalnie
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


def repair_flattened_superscript_labels(labels: list[str]) -> dict[int, str]:
    """Naprawa etykiet ze SPŁASZCZONYM indeksem górnym, których nie wykrył
    pdfplumber (Krok 84: Kodeks postępowania cywilnego -- 54 artykuły, bloki
    477^15-16 i 479^28-88: "47928" zamiast "479²⁸", bo nagłówki w gęstym
    bloku kolejnych indeksów nie zostały wykryte i `align_superscript_corrections`
    nie miało czego dopasować).

    Heurystyka z ciągłości numeracji (artykuły w ustawie idą rosnąco): etykieta
    złożona z samych cyfr (+ ew. litery), której początek to numer ostatniego
    zaufanego artykułu ("479" + "28"), a PÓŹNIEJ w sekwencji numeracja cofa się
    poniżej niej (po bloku wraca "478"/"479⁸⁹") -- czyli nie mogła być
    prawdziwym, wyższym artykułem -- dostaje indeks górny. Obie przesłanki
    naraz (prefiks + późniejszy spadek) praktycznie wykluczają fałszywe
    trafienia w ustawach o monotonicznej numeracji. Zwraca {indeks: nowa etykieta}."""
    def lead(label: str) -> int | None:
        m = re.match(r"\d+", label)
        return int(m.group()) if m else None

    leads = [lead(label) for label in labels]
    fixes: dict[int, str] = {}
    prev_base: int | None = None
    for i, label in enumerate(labels):
        m = re.fullmatch(r"(\d+)([a-z]{0,3})", label)
        if m and prev_base is not None:
            digits, letters = m.groups()
            prefix = str(prev_base)
            value = int(digits)
            if len(digits) > len(prefix) and digits.startswith(prefix):
                drop = next((l for l in leads[i + 1 :] if l is not None and l < value), None)
                if drop is not None and drop >= prev_base:
                    fixes[i] = prefix + digits[len(prefix) :].translate(SUPERSCRIPT_DIGITS) + letters
                    continue
        if leads[i] is not None:
            prev_base = leads[i]
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


# Krok 86: PDF-y obwieszczeń sprzed ~2014 (pliki "O"/"T" w historii) mają czcionkę
# z kodowaniem Mac Central European odczytanym jako Mac Roman -- polskie litery
# wychodzą jako "dzia∏ajà" (= "działają"), "okreÊla" (= "określa"), "wzgl´du"
# (= "względu"), "U˝yte" (= "Użyte"), "DZIA¸" (= "DZIAŁ"). Mapowanie jest stałe
# (potwierdzone na 7 oknach historii w 7 dziedzinach, ~45 tys. wystąpień).
FONT_MOJIBAKE_MAP = str.maketrans(
    {
        "∏": "ł", "à": "ą", "Ê": "ś", "´": "ę", "˝": "ż", "ç": "ć", "ƒ": "ń", "ê": "ź",
        "¸": "Ł", "Â": "Ś", "˚": "Ż", "¢": "Ę", "¡": "Ń",
    }
)
FONT_MOJIBAKE_SIGNATURE = "∏à´˝"
FONT_MOJIBAKE_MIN_HITS = 30


def fix_font_mojibake(text: str) -> str:
    """Naprawia mojibake czcionki Mac CE/Roman (patrz FONT_MOJIBAKE_MAP), ale tylko gdy
    tekst DOKUMENTU ma wyraźną sygnaturę (>= 30 znaków z "∏à´˝"). Zdrowe teksty nie
    zawierają tych znaków, więc dla nich funkcja nic nie zmienia."""
    if sum(text.count(c) for c in FONT_MOJIBAKE_SIGNATURE) < FONT_MOJIBAKE_MIN_HITS:
        return text
    return text.translate(FONT_MOJIBAKE_MAP)


_LEGIT_CHARS = frozenset(
    "aąbcćdeęfghijklłmnńoóprsśtuwyzźżAĄBCĆDEĘFGHIJKLŁMNŃOÓPRSŚTUWYZŹŻqvxQVX0123456789 \n"
    ".,;:!?()[]\"'„”«»–—-/%§°…²³¹⁰⁴⁵⁶⁷⁸⁹&*+=<>_€$№@\u00a0\u00ad"
)
SCAN_QUALITY_MAX_ODD_PER_1000 = 4.0


def odd_char_rate_per_1000(texts: list[str]) -> float:
    """Liczba znaków spoza zwykłego alfabetu/interpunkcji polskich ustaw na 1000.
    Zdrowe okna historii: <= 0,9 (maks. z 25 dziedzin); zepsuty skan OCR (Krok 86,
    KPA przed 2000): 7,7; przed naprawą mojibake'u okna sprzed 2014: 35-42 --
    próg 4,0 (jedno wywołanie po `fix_font_mojibake`)."""
    total = sum(len(t) for t in texts)
    if not total:
        return 0.0
    return 1000.0 * sum(1 for t in texts for c in t if c not in _LEGIT_CHARS) / total


# Krok 90: nowsze PDF-y tekstów ujednoliconych Kancelarii Sejmu (Kodeks pracy z 2026-10,
# także fragmenty KPC) zapisują indeks górny jako "[n]" PRZYKLEJONE do liczby: "Art. 18[4].",
# "§ 2[1].", "art. 22[1]", "pkt 5[2])", "ust. 3[3d]". `strip_not_yet_in_force_text` zdejmował
# te nawiasy jak adnotację redakcyjną, więc cyfra się spłaszczała ("§ 21", "art. 221") --
# po odświeżeniu Kodeksu pracy 498 -> 236 indeksów górnych. Nawias redakcyjny ("[stary tekst]",
# "[1. ...]") nigdy nie jest przyklejony do cyfry i nie zamyka się tuż po 1-3 cyfrach/literach.
BRACKET_SUPERSCRIPT_RE = re.compile(r"(?<=\d)\[(\d{1,3})([a-z]{0,3})\]")


_HEADER_PREFIX_RE = re.compile(r"Art\. \d+$")


def convert_bracket_superscripts(text: str) -> str:
    """"22[1]" -> "22¹", "18[3d]" -> "18³d" (cyfry w nawiasie na indeks górny, litery obok).
    NAGŁÓWKI artykułów ("Art. 18[4].", wielka litera) zostają bez zmian: mają własny
    mechanizm (geometria PDF -> poprawka etykiety), a "Art. 18³c." nie pasuje do wzorca
    podziału na artykuły -- pierwsza wersja tej funkcji wchłaniała 28 artykułów Kodeksu pracy."""
    def repl(m: re.Match) -> str:
        if _HEADER_PREFIX_RE.search(text[max(0, m.start() - 14) : m.start()]):
            return m.group(0)
        return m.group(1).translate(SUPERSCRIPT_DIGITS) + m.group(2)

    return BRACKET_SUPERSCRIPT_RE.sub(repl, text)


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
    return convert_bracket_superscripts(strip_inline_footnote_refs(fix_font_mojibake("\n".join(lines))))


def _matched_angle_bracket_positions(text: str) -> set[int]:
    """Zwraca indeksy znaków '<'/'>' które faktycznie tworzą PARĘ (do
    usunięcia razem z zawartością), przez klasyczne dopasowanie stosowe
    (jak nawiasy w wyrażeniu) zamiast niezachłannego wzorca regex
    dopasowywanego niezależnie do każdej kolejnej pary z osobna.

    Niesparowany '<' (bez ')'>' gdziekolwiek dalej w tekście) albo
    niesparowany '>' (bez wcześniejszego '<') NIE trafiają do zbioru --
    zostają jako pojedyncze osierocone znaki do usunięcia bez ruszania
    zawartości wokół nich (bezpiecznik z Krok 45: nie ryzykujemy
    pochłonięcia zbyt dużego fragmentu poprawnego tekstu)."""
    matched: set[int] = set()
    stack: list[int] = []
    for i, ch in enumerate(text):
        if ch == "<":
            stack.append(i)
        elif ch == ">" and stack:
            start = stack.pop()
            matched.add(start)
            matched.add(i)
    return matched


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
    art. 3-6 Prawa lotniczego (okno 2006), PROGRESS.md Krok 65.

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

    Implementacja to POJEDYNCZY przebieg znak-po-znaku (nie dwa
    niezależne podstawienia regex jak przed Krokiem 47), bo dwa
    niezależne, niezachłanne podstawienia (`<.*?>` potem `\\[(.*?)\\]`)
    źle radzą sobie z zagnieżdżonymi/sąsiadującymi nawiasami -- patrz
    Krok 46 (`upadlosciowe`): dla `[1. ... [stare słowo] <nowe słowo>.]`
    niezachłany `\\[(.*?)\\]` dopasowuje od ZEWNĘTRZNEGO '[' do
    PIERWSZEGO napotkanego ']' (czyli tego WEWNĘTRZNEGO), zostawiając
    osierocony wewnętrzny '[' jako dosłowny tekst i osierocone
    zewnętrzne ']' nietknięte -- a dla dwóch sąsiadujących artykułów,
    z których jeden ma taki zagnieżdżony nawias, potrafi to pochłonąć
    CAŁY nagłówek i treść NASTĘPNEGO artykułu jako rzekomą "zawartość"
    tej samej pary (potwierdzone wprost w `prawo_upadlosciowe`, art.
    241/242 dla okna @2019-02-21 -- art. 242 znikał jako osobny wpis).

    Nowe podejście: dla "< >" dopasowanie parami robione jest
    JAWNIE (`_matched_angle_bracket_positions`, jak nawiasy w wyrażeniu),
    więc zagnieżdżenie/sąsiedztwo z "[ ]" nie ma znaczenia -- zawartość
    między faktycznie sparowanym '<' i '>' znika w całości niezależnie od
    głębokości. Dla "[ ]" nie trzeba w ogóle dopasowywać par: skoro
    zawartość ZAWSZE zostaje (to tylko adnotacja edytorska), wystarczy
    usunąć KAŻDY pojedynczy znak '['/']' napotkany poza obszarem "< >"
    -- nie ma ryzyka złego sparowania, bo nigdy nie decydujemy o zasięgu
    na podstawie tych znaków.

    Niesparowany "<" (bez odpowiadającego ">" dalej w tekście) albo
    niesparowany ">" nie usuwają żadnej zawartości -- tylko sam
    pojedynczy znak znika, tekst wokół zostaje nietknięty, tak jak przy
    dwóch niezależnych regexach wcześniej -- ale wypisujemy ostrzeżenie
    do ręcznego sprawdzenia."""
    if text.count("<") != text.count(">"):
        print(
            "[UWAGA] niesparowane nawiasy < > w tekście -- część fragmentu "
            "'jeszcze nieobowiązującego' mogła nie zostać usunięta, sprawdź ręcznie"
        )
    if text.count("[") != text.count("]"):
        print("[UWAGA] niesparowane nawiasy [ ] w tekście -- sprawdź ręcznie")

    matched_angle = _matched_angle_bracket_positions(text)
    unclosed_article_ends = _unclosed_future_article_ends(text, matched_angle)
    unclosed_article_ends.update(_unclosed_future_paragraph_ends(text, matched_angle))
    output: list[str] = []
    angle_depth = 0
    skip_until = -1
    for i, ch in enumerate(text):
        if i < skip_until:
            continue
        if ch == "<":
            if i in matched_angle:
                angle_depth += 1
            elif i in unclosed_article_ends:
                skip_until = unclosed_article_ends[i]
            continue
        if ch == ">":
            if i in matched_angle:
                angle_depth -= 1
            continue
        if ch == "[" or ch == "]":
            continue
        if angle_depth == 0:
            output.append(ch)
    return "".join(output)


NEXT_ARTICLE_HEADER_RE = re.compile(r"\n[<\[]?Art\.\s*\d")


def _unclosed_future_article_ends(text: str, matched_angle: set[int]) -> dict[int, int]:
    """Kancelaria Sejmu czasem gubi w samym PDF-ie zamykające ">" przyszłej
    wersji artykułu (potwierdzone w surowym wyjściu pypdf: POŚ 2016 art.
    286, ruch drogowy 2017 art. 80b/80bh/80c/80ca). Niesparowany "<" nic
    nie wycinał, więc przyszła wersja zostawała obok obowiązującej "[Art.
    286. ...]" jako duplikat etykiety (PROGRESS.md Krok 65).

    Dla niesparowanego "<" stojącego na początku nagłówka artykułu
    ("\\n<Art. 286.") zwraca {pozycja '<': początek następnego nagłówka
    artykułu} -- wycinamy NAJMNIEJ, co się da (jeden artykuł). Blok
    przyszły bywa wieloartykułowy (80ca-80ch), ale bez ">" nie wiadomo,
    gdzie się kończy, a ten sam bezpiecznik co w Kroku 45: lepiej zostawić
    za dużo niż wyciąć obowiązujący tekst. Inne niesparowane "<" -- jak
    dotąd, tylko sam znak znika.

    Wyjątek: artykuł zamknięty "]" bez własnego "[" (Prawo energetyczne
    2024, art. 5ab: "<Art. 5ab. ... art. 6b.]", a dalej osobno właściwa
    przyszła wersja "<Art. 5ab. ...>") -- to literówka za "[", czyli
    OBOWIĄZUJĄCY tekst; zostaje."""
    ends: dict[int, int] = {}
    for i, ch in enumerate(text):
        if ch != "<" or i in matched_angle:
            continue
        if not (i == 0 or text[i - 1] == "\n") or not text.startswith("Art.", i + 1):
            continue
        nxt = NEXT_ARTICLE_HEADER_RE.search(text, i + 1)
        end = nxt.start() + 1 if nxt else len(text)
        if text.count("]", i, end) > text.count("[", i, end):
            continue
        ends[i] = end
    return ends


UNCLOSED_PARAGRAPH_OPEN_RE = re.compile(r"<(\d+)\. ")
MAX_UNCLOSED_PARAGRAPH_CHARS = 6000


def _unclosed_future_paragraph_ends(text: str, matched_angle: set[int]) -> dict[int, int]:
    """Odpowiednik `_unclosed_future_article_ends` dla USTĘPU: Kancelaria
    Sejmu zgubiła zamykające ">" przyszłego brzmienia pojedynczego ustępu
    (ustawa o systemie ubezpieczeń społecznych, stan 2026-11-09, art. 68
    ust. 2: "[2. ...]\\n<2. ..." bez ">", a dalej obowiązujące ust. 3-4).
    Niesparowany "<" nic nie wycinał, więc przyszłe brzmienie ust. 2
    trafiało do bazy obok obowiązującego (Krok 141 repo wielodziedzinowego, Krok 33 tego repo).

    Reguła jest celowo WĄSKA (ten sam bezpiecznik co w Kroku 45 --
    lepiej zostawić za dużo niż wyciąć obowiązujący tekst). Obsługujemy
    niesparowany "<" tylko gdy WSZYSTKIE warunki są spełnione:
    1. stoi na początku wiersza i otwiera ustęp "<N. ",
    2. bezpośrednio przed nim (poza białymi znakami) kończy się "]", czyli
       zamyka obowiązujące, zastępowane brzmienie TEGO ustępu,
    3. dalej jest początek ustępu N+1 ("\\nN+1. ") albo nagłówek następnego
       artykułu -- tam kończymy wycinanie (blok przyszły o kilku ustępach
       bez ">" nie jest wykrywany, zostaje jak dotąd),
    4. odcinek ma najwyżej `MAX_UNCLOSED_PARAGRAPH_CHARS` znaków i nie
       zawiera więcej "]" niż "[".
    Inne niesparowane "<" -- jak dotąd, tylko sam znak znika."""
    ends: dict[int, int] = {}
    for m in UNCLOSED_PARAGRAPH_OPEN_RE.finditer(text):
        i = m.start()
        if i in matched_angle or not (i == 0 or text[i - 1] == "\n"):
            continue
        if not text[:i].rstrip().endswith("]"):
            continue
        next_paragraph = re.compile(rf"\n{int(m.group(1)) + 1}\. ").search(text, i + 1)
        next_article = NEXT_ARTICLE_HEADER_RE.search(text, i + 1)
        candidates = [x.start() + 1 for x in (next_paragraph, next_article) if x]
        if not candidates:
            continue
        end = min(candidates)
        if end - i > MAX_UNCLOSED_PARAGRAPH_CHARS or text.count("]", i, end) > text.count("[", i, end):
            continue
        ends[i] = end
    return ends


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


LETTER_SUPERSCRIPT_HEADER_RE = re.compile(r"Art\. (\d+[a-ząćęłńóśźż]{1,3}) ?(\d{1,2})\. ")


def recover_letter_superscript_headers(clean_text: str) -> str:
    """Nagłówki z indeksem górnym PO sufiksie literowym -- np. "Art. 5b¹."
    `pypdf` spłaszcza do "Art. 5b1." albo "Art. 5a 1." (ze spacją), czego
    `ARTICLE_SPLIT_RE` nie łapie (wymaga kropki zaraz po literze), więc
    cały artykuł był wchłaniany przez poprzedni. W polskiej numeracji
    cyfra po literze w numerze artykułu występuje WYŁĄCZNIE jako indeks
    górny, więc zamiana jest jednoznaczna -- od razu na znak Unicode (a nie
    tylko sklejenie jak w `recover_midtext_superscript_headers`), żeby
    etykieta była poprawna nawet, gdy korekta pdfplumber zostanie pominięta
    przez niezgodną liczbę nagłówków. Wymuszamy też początek linii.
    Potwierdzony przypadek: Prawo energetyczne, 17 nagłówków (PROGRESS.md,
    Krok 60); w żadnym innym pobranym akcie ten wzorzec nie występuje."""
    return LETTER_SUPERSCRIPT_HEADER_RE.sub(
        lambda m: f"\nArt. {m.group(1)}{m.group(2).translate(SUPERSCRIPT_DIGITS)}. ", clean_text
    )


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


# Adnotacje Kancelarii Sejmu z marginesu tekstu ujednoliconego ("Nowe brzmienie
# § 2 w art. 18³a wejdzie w życie z dn. 5.11.2026 r. (Dz. U. z 2026 r. poz.
# 1046).") wpadają w środek zdania przepisu -- szum w embeddingach i w
# kontekście modelu (PROGRESS.md, Krok 67/74). Opis zawsze kończy się
# odwołaniem do artykułu/ustępu tuż przed "wejdzie w życie" i składa się z
# małych liter, numerów i skrótów jednostek redakcyjnych -- to odróżnia go od
# treści przepisu przed nim.
# Data bywa rozbita spacją przez pypdf ("1.01.20 30 r."), a zamiast daty bywa
# "z dniem określonym w komunikacie, o którym mowa w art. N ustawy z dn. ...".
ANNOTATION_END_RE = re.compile(
    r"\s*wejdzie w życie z (?:dn\s?\.\s*[\d\s.]{6,14}?\s*r\."
    r"|dn(?:iem)?\s?\.?\s*określonym w komunikacie, o którym mowa w art\.\s*\d+[a-z]*\s+ustawy z dn(?:ia)?\.?\s[^()]{0,200}?)"
    r"\s*\(Dz\.\s*U\.[^)]*\)\.?"
)
ANNOTATION_START_RE = re.compile(
    r"(?:Nowe brzmienie|Now[aey]\s|Dodan[a-ząęśćźżół]*|Uchylon[a-ząęśćźżół]*|Przepis uchylający|Zmiana"
    r"|Skreślon[a-ząęśćźżół]*|Ust\.|Art\.|Pkt|Lit\.|§)"
)
_ANNOTATION_TOKEN = r"(?:[a-ząęśćźżół()]+\.?|\d+[a-z]*|ust\.|art\.|pkt|lit\.|zd\.|tiret|§|[,–\-])"
ANNOTATION_DESC_RE = re.compile(rf"{_ANNOTATION_TOKEN}(?:\s+{_ANNOTATION_TOKEN})*")
ANNOTATION_TAIL_RE = re.compile(r"(?:[Aa]rt\.|ust\.)\s*\d+[a-z]*(?:\s[\da-z]{1,3})?$")
ANNOTATION_LOOKBACK = 200


def strip_entry_into_force_annotations(text: str) -> str:
    """Usuwa adnotacje "... wejdzie w życie z dn. D r. (Dz. U. ...)." z treści
    artykułu. Dla każdego końca adnotacji szuka NAJWCZEŚNIEJSZEGO słowa-klucza
    (w oknie 200 znaków), od którego do końca opis jest czysty (patrz
    ANNOTATION_DESC_RE) i kończy się odwołaniem do artykułu/ustępu; bez
    takiego startu adnotacja zostaje (lepiej szum niż ucięty przepis)."""
    out, pos = [], 0
    for m in ANNOTATION_END_RE.finditer(text):
        if m.start() < pos:
            continue
        window_start = max(pos, m.start() - ANNOTATION_LOOKBACK)
        pre = text[window_start : m.start()]
        cut = None
        for start in ANNOTATION_START_RE.finditer(pre):
            candidate = pre[start.start() :].strip()
            desc = pre[start.end() :].strip()
            if ANNOTATION_TAIL_RE.search(candidate) and (not desc or ANNOTATION_DESC_RE.fullmatch(desc)):
                cut = window_start + start.start()
                break
        if cut is None:
            continue
        out.append(text[pos:cut])
        pos = m.end()
    out.append(text[pos:])
    return re.sub(r"[ \t]{2,}", " ", "".join(out)).strip()


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
    clean_text = recover_letter_superscript_headers(clean_text)
    clean_text = fix_stray_space_before_period(clean_text)

    ustep_corrections = find_ustep_superscript_corrections_pdfplumber(pdf_bytes)
    if ustep_corrections:
        clean_text = apply_flattened_superscript_corrections(clean_text, ustep_corrections, label="poziomu ustępu")

    citation_corrections = find_citation_superscript_corrections_pdfplumber(pdf_bytes)
    if citation_corrections:
        clean_text = apply_flattened_superscript_corrections(
            clean_text, citation_corrections, label="cytatu w prozie"
        )

    cross_act_corrections = find_cross_act_citation_corrections_pdfplumber(pdf_bytes, act_short)
    if cross_act_corrections:
        clean_text = apply_flattened_superscript_corrections(
            clean_text, cross_act_corrections, label="cytatu do innej ustawy"
        )

    print(f"[{act_short}] tnę na artykuły...")
    articles = parse_articles(clean_text)
    for article in articles:
        article["text"] = strip_entry_into_force_annotations(article["text"])
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
    # Krok 84: etykiety spłaszczone, których pdfplumber w ogóle nie wykrył
    repaired = repair_flattened_superscript_labels([a["article"] for a in articles])
    for i, label in repaired.items():
        article = articles[i]
        article["text"] = article["text"].replace(f"Art. {article['article']}.", f"Art. {label}.", 1)
        article["article"] = label
    if repaired:
        print(f"[{act_short}] naprawiono spłaszczone etykiety (ciągłość numeracji) w {len(repaired)} artykułach")

    articles = split_annexes(articles)
    return articles


# Początek załączników bez nagłówka: "Załącznik nr N" po końcu zdania i przed wielką literą albo
# nawiasem ("ustawy. Załącznik nr 1 Lp. ..."). Tekst artykułu ma już sklejone linie, więc nie
# da się kotwiczyć do linii; proza ("zgodnie z Załącznik nr 1 do ustawy") nie pasuje.
ANNEX_LINE_START_RE = re.compile(r"(?<=[.;)])\s+(Za\S{1,3}cznik nr \d+[a-z]?)(?=\s+[A-ZĄĆĘŁŃÓŚŹŻ(])")
ANNEXES_HEADER_RE = re.compile(r"Za\S{1,3}\s?\S{0,2}cznik(?:i)? do ustawy z dnia [^()]{0,60}\([^)]*\)\s*")
ANNEX_NUMBER_RE = re.compile(r"(?=Za\S{1,3}cznik nr (\d+[a-z]?)(\))?)")


def _annex_number(raw: str, has_paren: bool, previous: str | None) -> str:
    """"617)" = załącznik nr 6 + sklejony odsyłacz "17)" (VAT): przy ")"
    bierzemy kolejny numer po poprzednim, jeśli cyfry od niego się zaczynają."""
    if not has_paren:
        return raw
    prev_num = int(re.match(r"\d+", previous).group()) if previous else 0
    for candidate in (str(prev_num + 1), str(prev_num)):
        if raw.startswith(candidate) and raw != candidate:
            return candidate
    return raw[:1]


def split_annexes(articles: list[dict]) -> list[dict]:
    """Załączniki ustawy (tabele stawek, wzory pouczeń, kategorie obiektów)
    stoją w PDF-ie po ostatnim artykule bez nagłówka "Art.", więc doklejały
    się do niego -- VAT art. 176 miał 49 tys. znaków, a w `konsumenckie` art.
    55 (wejście w życie + wzór pouczenia o odstąpieniu) wygrywał rankingiem
    pytań o zwrot pieniędzy (Krok 68/70). Odcinamy je od ostatniego artykułu
    po nagłówku "Załącznik(i) do ustawy z dnia ... (Dz. U. ...)" i dzielimy
    na osobne fragmenty z etykietą "załącznik nr N" (albo "załącznik", gdy
    jest jeden, nienumerowany)."""
    if not articles:
        return articles
    last = articles[-1]
    match = ANNEXES_HEADER_RE.search(last["text"])
    if match:
        annex_start, head_end = match.end(), match.start()
    else:
        # Krok 90: nowsze teksty ujednolicone (Prawo lotnicze 2026-10) nie mają nagłówka
        # "Załączniki do ustawy...", tylko od razu linię "Załącznik nr 1" -- 97 tys. znaków
        # załączników lądowało w art. 232 ("wchodzi w życie"). Linia złożona WYŁĄCZNIE z
        # "Załącznik nr N" nie występuje w zwykłej prozie.
        line_match = ANNEX_LINE_START_RE.search(last["text"])
        if not line_match:
            return articles
        annex_start, head_end = line_match.start(1), line_match.start()
    annex_text = last["text"][annex_start:].strip()
    head = last["text"][:head_end].rstrip()
    result = articles[:-1] + [{**last, "text": head}]
    starts = []
    for m in ANNEX_NUMBER_RE.finditer(annex_text):
        starts.append((m.start(), _annex_number(m.group(1), bool(m.group(2)), starts[-1][1] if starts else None)))
    if not starts:
        if annex_text:
            result.append({"article": "załącznik", "text": f"Załącznik. {annex_text}"})
        return result
    seen: set[str] = set()
    for i, (pos, number) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(annex_text)
        label = f"załącznik nr {number}"
        if label in seen:  # odwołanie "Załącznik nr N" wewnątrz innego załącznika -- doklejamy do bieżącego
            result[-1]["text"] += " " + annex_text[pos:end].strip()
            continue
        seen.add(label)
        result.append({"article": label, "text": annex_text[pos:end].strip()})
    return result


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
            "domains": act.get("domains", []),
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
    source: dict,
    act_short: str,
    base_eli: str,
    base_title: str,
    valid_from,
    valid_to,
    announcement_eli,
    domains: list[str] | None = None,
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
        source, act_short, base_eli, base_title, valid_from, valid_to, announcement_eli, domains, articles, kind, file_name
    )


def drop_future_wording_duplicates(articles: list[dict]) -> list[dict]:
    """Tekst jednolity w wersji z dnia ogłoszenia (plik "T", historia od
    Kroku 69) podaje artykuł zmieniony przepisem, który jeszcze nie wszedł w
    życie, DWA razy pod rząd: najpierw brzmienie obowiązujące (przypis "W tym
    brzmieniu obowiązuje do wejścia w życie zmiany..."), zaraz potem nowe
    ("W brzmieniu ustalonym przez..."). Sprawdzone na Ordynacji 2017, KSH
    2020, Prawie upadłościowym 2019, Prawie restrukturyzacyjnym 2021, Prawie
    o ruchu drogowym 2012. Zostawiamy pierwsze -- obowiązujące w oknie (w
    plikach "U" tę rolę pełniło wycinanie "< >"). Tylko KOLEJNE duplikaty."""
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
    domains: list[str] | None,
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
    # ustawy, np. 4 w Prawie budowlanym, powtarzający się); PROGRESS.md Krok 65.
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
            "domains": domains or [],
        }
        for a in articles
    ]


def rebuild_all_articles() -> list[dict]:
    """Odbudowuje all_articles.json ze WSZYSTKICH już pobranych aktów na
    dysku (data/processed/{short}.json), niezależnie od tego, które akty
    zostały pobrane w BIEŻĄCYM uruchomieniu `main()` -- inaczej odpalenie
    `--domain podatkowe` nadpisałoby all_articles.json samą tylko treścią
    podatkową, kasując po cichu wcześniej pobrane dziedziny. Pomija w
    ciszy akty z konfiguracji, których jeszcze nie pobrano (brak pliku
    {short}.json -- świeży klon repo albo dopiero co dodana pozycja w
    acts_config.json)."""
    all_chunks = []
    for act in load_acts_config():
        path = PROCESSED_DIR / f"{act['short']}.json"
        if not path.exists():
            continue
        all_chunks.extend(json.loads(path.read_text(encoding="utf-8")))
    return all_chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Ogranicz pobieranie do jednej dziedziny (pole 'domains' w data/acts_config.json). "
        "Domyślnie: wszystkie akty z konfiguracji.",
    )
    parser.add_argument(
        "--acts",
        type=str,
        default=None,
        help="Lista skrótów ustaw po przecinku -- ogranicza dodatkowo do podanych aktów "
        "(można łączyć z --domain).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Liczba procesów parsujących PDF-y (domyślnie: dobrane do rdzeni i RAM-u maszyny, "
        "patrz default_parse_workers; 1 = bez równoległości).",
    )
    args = parser.parse_args()

    shorts = {s.strip() for s in args.acts.split(",") if s.strip()} if args.acts else None
    acts = load_acts_config(domain=args.domain, shorts=shorts)
    if not acts:
        raise SystemExit("Brak aktów pasujących do podanych filtrów (--domain/--acts).")

    workers = args.workers or default_parse_workers(len(acts))
    print(f"[INFO] parsowanie PDF-ów: {workers} proces(ów) równolegle (pobieranie sekwencyjne)")
    with ParsePool(workers) as pool:
        pending = []
        for act in acts:
            meta, title, eli, pdf_bytes, kind, file_name = _fetch_source(act, act["short"])
            save_act_raw(act, meta, pdf_bytes)
            pending.append((act, title, eli, kind, file_name, pool.submit(pdf_bytes, act["short"])))
            time.sleep(1)  # nie bombardujemy publicznego API bez potrzeby
        for act, title, eli, kind, file_name, future in pending:
            write_act_chunks(act, title, eli, kind, file_name, future.result())

    all_chunks = rebuild_all_articles()
    out_path = PROCESSED_DIR / "all_articles.json"
    out_path.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRazem: {len(all_chunks)} fragmentów (artykułów) zapisanych w {out_path}")


if __name__ == "__main__":
    main()
