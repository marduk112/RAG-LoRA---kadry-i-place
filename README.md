# Asystent kadrowo-płacowy (RAG + lokalny model Bielik)

> **English summary.** A local, offline-capable question-answering assistant for Polish labour and social-insurance law
> (7 statutes: Labour Code, ZUS, PIT, PPK, benefits...). Retrieval-augmented generation (embeddings + reranker) over the
> official statute texts from the Sejm's ELI API, answered by a local LLM (Bielik) on Apple Silicon (MLX), NVIDIA/CUDA or
> LM Studio. The project started as a LoRA fine-tuning experiment; later measurements in the successor repo found no
> measurable gain from the adapters over the base model with RAG, so the adapter is now optional (see "Co zmierzono, a
> czego nie"). Documentation and the development log (`PROGRESS.md`) are in Polish. **Not legal advice**, not validated
> by a lawyer. Code: Apache-2.0.
>
> **Multi-domain successor:** [prawo-polskie-rag](https://github.com/marduk112/prawo-polskie-rag) -- many domains of Polish law,
> a domain router, reranking and query expansion (this repo stays single-domain: labour and payroll law).

Lokalny (offline po pierwszym pobraniu) asystent odpowiadający na pytania
z zakresu polskiego prawa pracy i ubezpieczeń społecznych, bez wysyłania
niczego do chmury. Natywnie na Apple Silicon (MLX) — plus alternatywne
ścieżki dla Linuksa/Windows z kartą NVIDIA (CUDA, patrz niżej;
przetestowane na RunPod A40) i dla dowolnego modelu już załadowanego w LM Studio.

Dziennik tego, jak to powstawało krok po kroku — z napotkanymi
pułapkami i decyzjami — jest w [PROGRESS.md](PROGRESS.md). Ten plik to
skrócona instrukcja "jak tego używać".

## Jak to działa

Dwa mechanizmy; tylko pierwszy jest niezbędny:

1. **RAG (retrieval-augmented generation)** — pytanie użytkownika jest
   embedowane i porównywane z bazą artykułów pobranych ustaw
   (`data/processed/all_articles.json`); wyniki poprawia reranker i fuzja
   RRF (sekcja "Wyszukiwanie" niżej). Najtrafniejsze fragmenty
   trafiają jako kontekst do modelu. To źródło **faktów** — aktualnych,
   źródłowych tekstów przepisów.
2. **LoRA fine-tuning (opcjonalny — pierwotny pomysł projektu)** — model
   **Bielik** (polski LLM od SpeakLeash; wariant **11B-v3.0-Instruct**,
   dostępny też szybszy **4.5B-v3.0-Instruct**) doduczony na przykładach
   pytanie/odpowiedź z tej dziedziny, żeby nauczyć go **stylu**: zwięzłe
   odpowiedzi z cytowaniem artykułu i przyznawanie się do niewiedzy
   zamiast zgadywania. **Późniejsze pomiary nie potwierdziły, że to
   daje zysk:** w projekcie następcy adapter dla Bielika v3.1 nie
   poprawił jakości względem samego modelu z RAG (22 wygrane adaptera,
   18 bazy, 29 remisów na 76 pytaniach kadrowych, różnica nieistotna
   statystycznie), a w trzech innych dziedzinach adaptery nie różniły się od
   Bielika v3.1 bez adaptera (patrz "Co zmierzono, a czego nie"). Przepis na
   trening zostaje w repo (kroki 3–4 niżej), ale nie jest potrzebny:
   wystarczy `chat.py --no-adapter` albo `chat_lmstudio.py`.

**Dlaczego RAG, a nie samo dostrajanie?** Testy pokazały, że
sam doduczony model (bez kontekstu RAG) nie jest wystarczająco
niezawodny w precyzyjnych progach liczbowych (np. dokładnie która
granica stażu pracy daje 1 miesiąc, a która 3 miesiące wypowiedzenia)
i przy pytaniach spoza wytrenowanego zakresu potrafi pewnie zmyślać
fakty. Po dodaniu kontekstu RAG te same pytania wypadały poprawnie.
Szczegóły i konkretne przykłady testów — patrz PROGRESS.md, kroki 4b–4c.
Jak działa cały pipeline w prostych słowach opisuje
[JAK_TO_DZIALA.md](https://github.com/marduk112/prawo-polskie-rag/blob/main/JAK_TO_DZIALA.md)
w projekcie następcy.

**Który model odpowiada najlepiej?** W projekcie następcy (te same
konteksty RAG, 76 pytań kadrowych) **Qwen3.8-27B** wypadł wyraźnie lepiej niż
Bielik v3.0: 65 wygranych, 7 remisów, 0 wygranych Bielika; Bielik miał
poważne błędy w 40 z 76 odpowiedzi, Qwen w 0. To pomiar okiem modelu-sędziego
(sędzią był sam Qwen), z ręcznym sprawdzeniem próbki, bez prawnika, a Qwen
pisze ponad dwa razy dłużej — traktuj go jako wskazówkę, nie dowód. **W tym
repo `chat_lmstudio.py` z Qwenem nie był testowany** (modele z "myśleniem"
wymagają obsługi, którą ma wersja wielodziedzinowa). Szczegóły: Kroki 106 i 113
w [PROGRESS.md następcy](https://github.com/marduk112/prawo-polskie-rag/blob/main/PROGRESS.md).

**Zastrzeżenie:** zestaw `data/finetune/{train,valid}.jsonl` nie został
zweryfikowany przez specjalistę kadrowo-płacowego — napisał go autor
tego repo na podstawie wiedzy ogólnej i testów działania kodu (czy RAG
trafia w dobry artykuł, czy model poprawnie cytuje i przyznaje się do
niewiedzy), nie merytorycznej poprawności prawnej każdego szczegółu.
Traktuj te przykłady jako materiał do nauki STYLU odpowiedzi, nie jako
zweryfikowane źródło faktów kadrowo-płacowych. Fakty i tak mają
pochodzić z kontekstu RAG (prawdziwy tekst ustawy z ELI API, `source_url`
w każdym wyniku pozwala zweryfikować źródło) — ale zanim ktokolwiek
oprze się na odpowiedziach tego asystenta w realnej sprawie kadrowej,
powinien je zweryfikować ze specjalistą lub bezpośrednio z tekstem
ustawy. To projekt edukacyjny/eksperymentalny, nie źródło porady prawnej.

## Wymagania

Wspólne dla wszystkich wariantów:
- Python 3.12 (system może mieć starszą wersję — patrz PROGRESS.md,
  krok 0)

Zależnie od tego, którą ścieżkę wybierzesz (patrz sekcja "Użycie"
niżej po instalacji):

| Wariant | Platforma | Status | Dodatkowe wymagania |
|---|---|---|---|
| `chat.py` (MLX) | macOS + Apple Silicon | ✅ Przetestowane: macOS 26 (Tahoe), Apple M4 Pro, 64GB RAM | Konto HF z zaakceptowaną licencją Bielika (patrz niżej) |
| `chat.py` (MLX) | macOS + Apple Silicon, starsze chipy z serii M lub inna wersja macOS | ⚠️ Nieprzetestowane, ale powinno działać — MLX wspiera cały Apple Silicon | jw. |
| `chat_cuda.py` (CUDA) | Linux/Windows + karta NVIDIA | ✅ Przetestowane na RunPod (A40, 48GB VRAM) — patrz PROGRESS.md, Krok 26 | Sterowniki CUDA, `torch` z obsługą CUDA, konto HF z licencją Bielika |
| `chat_lmstudio.py` (LM Studio) | dowolna (tam, gdzie działa LM Studio) | ✅ Przetestowane na macOS; LM Studio samo wspiera też Windows/Linux z CUDA | [LM Studio](https://lmstudio.ai/) z załadowanym modelem -- **nie wymaga** konta HF/licencji Bielika, bo model przynosisz swój |

Reszta projektu (`download_acts.py`, `build_rag_index.py`, `rag_search.py`,
`check_acts_freshness.py`) to czysty Python + `sentence-transformers`/
`requests`/`pypdf` — platformowo niezależne, powinny działać wszędzie,
gdzie działa Python 3.12, niezależnie od wybranego wariantu czatu.

**Szacunkowe minimum RAM (MLX):** nie testowane na maszynie z mniejszą
ilością pamięci niż 64GB, ale sądząc po zmierzonym `Peak memory` z
`mlx_lm` (patrz PROGRESS.md) -- ok. 5GB dla wariantu 4.5B, 8-11GB dla
11B (zależnie od długości kontekstu RAG) -- **16GB powinno spokojnie
wystarczyć na wariant 4.5B**, dla 11B bezpieczniej mieć więcej zapasu
(np. 32GB), zwłaszcza przy fine-tuningu LoRA, gdzie zużycie jest
wyższe niż przy samym czacie.

Konto Hugging Face z zaakceptowaną licencją modelu (dotyczy wariantów
MLX i CUDA -- oba warianty modelu są *gated*, licencję trzeba
zaakceptować osobno dla każdego z nich na stronie repo na HF, zanim
`hf download`/`mlx_lm.convert`/`transformers` zadziała):
- [`speakleash/Bielik-11B-v3.0-Instruct`](https://huggingface.co/speakleash/Bielik-11B-v3.0-Instruct) (domyślny)
- [`speakleash/Bielik-4.5B-v3.0-Instruct`](https://huggingface.co/speakleash/Bielik-4.5B-v3.0-Instruct) (szybszy wariant, opcjonalny)

## Instalacja

```bash
python3.12 -m venv .venv

# macOS + Apple Silicon (MLX) -- wariant domyślny:
.venv/bin/pip install -r requirements.txt

# Linux/Windows + NVIDIA (CUDA) -- zamiast powyższego, patrz też
# sekcja "Linux/Windows + karta NVIDIA (CUDA)" niżej:
# .venv/bin/pip install -r requirements-common.txt -r requirements-cuda.txt

# Tylko RAG + LM Studio (dowolna platforma, bez MLX/CUDA):
# .venv/bin/pip install -r requirements-common.txt

# Zaloguj się do Hugging Face (token z https://huggingface.co/settings/tokens)
# -- pomiń, jeśli używasz tylko chat_lmstudio.py
.venv/bin/hf auth login
```

## Odtworzenie całego pipeline'u od zera

`models/`, `adapters/`, `data/raw/` i `data/processed/` są celowo
niecommitowane (patrz `.gitignore` — duże, w pełni odtwarzalne pliki).
Po sklonowaniu repo musisz przejść poniższe kroki raz, zanim zadziała
sekcja "Użycie" niżej (chyba że korzystasz wyłącznie z
`chat_lmstudio.py` z własnym modelem w LM Studio — wtedy wystarczą
kroki 1-2, model dostarcza LM Studio).

Kroki 1-2 są wspólne dla wszystkich wariantów. Kroki 3-4 poniżej to
wariant **MLX (macOS)** — dla **CUDA** (Linux/Windows + NVIDIA) pomiń
je i użyj `scripts/train_lora_cuda.py` zamiast `mlx_lm.convert`/
`mlx_lm.lora`, opisanego w sekcji "Linux/Windows + karta NVIDIA
(CUDA)" niżej.

```bash
# 1. Pobranie aktów prawnych z ELI API Sejmu -> data/raw/, data/processed/
.venv/bin/python scripts/download_acts.py

# 2. Zbudowanie indeksu RAG (embeddingi artykułów) -> data/processed/rag_index*.
.venv/bin/python scripts/build_rag_index.py

# 2b. [opcjonalnie] Historyczne wersje ustaw -> odpowiadanie "na dany dzień
# w przeszłości" (--as-of / /data w chat.py, patrz sekcja "Użycie" niżej).
# Osobny, opt-in krok -- znacznie więcej zapytań do publicznego API ELI niż
# krok 1 (jedno obwieszczenie + jeden PDF na historyczną wersję tekstu, ~60
# wersji łącznie na 7 ustaw), więc NIE jest częścią zwykłego kroku 1.
.venv/bin/python scripts/download_acts_history.py

# 2c. [opcjonalnie] Coroczne rozporządzenia/obwieszczenia z konkretnymi
# kwotami, których same ustawy nie zawierają -- patrz PROGRESS.md, Krok
# 21-23. Niezależne od kroku 2b, ale też wymagają --include-history niżej.
.venv/bin/python scripts/download_wage_regulations.py        # min. wynagrodzenie, 2004-2027
.venv/bin/python scripts/download_zus_limit_regulations.py   # limit 30-krotności składek ZUS, 1999-2027
.venv/bin/python scripts/download_avg_wage_regulations.py    # przeciętne wynagrodzenie GUS, 2003-2025

.venv/bin/python scripts/build_rag_index.py --include-history

# 3. [macOS/MLX] Pobranie i konwersja Bielika do MLX (kwantyzacja 4-bit) -> models/
# (dla wariantu 4.5B podmień hf-path/mlx-path na Bielik-4.5B-v3.0-Instruct)
.venv/bin/hf download speakleash/Bielik-11B-v3.0-Instruct  # najpierw pełny snapshot, patrz uwaga niżej
.venv/bin/mlx_lm.convert \
  --hf-path speakleash/Bielik-11B-v3.0-Instruct \
  --mlx-path models/Bielik-11B-v3.0-Instruct-mlx \
  -q

# 4. [macOS/MLX] Fine-tuning LoRA na data/finetune/{train,valid}.jsonl -> adapters/
.venv/bin/mlx_lm.lora \
  --model models/Bielik-11B-v3.0-Instruct-mlx \
  --train \
  --data data/finetune \
  --batch-size 2 \
  --iters 200 \
  --mask-prompt \
  --adapter-path adapters/bielik11b-kadry-lora
```

**Uwaga o kroku 3:** uruchom najpierw `hf download` (pełny snapshot
repo), a dopiero potem `mlx_lm.convert` -- w innym wypadku konwersja
może się wywalić z `IncompleteSnapshotError`, bo `mlx_lm.convert` przy
ładowaniu modelu pobiera tylko pliki potrzebne do inferencji, pomijając
np. `.gitattributes`/`README.md`, a etap zapisu wymaga kompletnego
zrzutu repo. Szczegóły w PROGRESS.md, krok 4a.

**Uwaga o kroku 4:** monitoruj `Val loss` w trakcie treningu. W naszych
przebiegach najlepszy wynik walidacyjny wypadał bardzo wcześnie
(iteracja 25 dla obu wariantów przy obecnym, 50-elementowym zbiorze —
patrz PROGRESS.md, Krok 25) — dalszy trening tylko przeuczał model.
Zalecane jest
ręczne sprawdzenie kilku zapisanych checkpointów
(`adapters/*/0000XXX_adapters.safetensors`) na pytaniach spoza
dosłownej treści zbioru treningowego i wybranie najlepszego, a nie
automatyczne poleganie na finalnej iteracji. Szczegóły w PROGRESS.md,
kroki 4b i 6.

## Użycie

```bash
# Tryb interaktywny
.venv/bin/python scripts/chat.py

# Pojedyncze pytanie
.venv/bin/python scripts/chat.py --prompt "Ile dni urlopu przysługuje po 10 latach pracy?"

# Dla porównania: bez LoRA (sam bazowy model + RAG)
.venv/bin/python scripts/chat.py --no-adapter

# Jeszcze więcej fragmentów kontekstu niż domyślne 8 (np. przy pytaniach
# o kwoty konkurujących z wieloma podobnymi artykułami, patrz niżej)
.venv/bin/python scripts/chat.py --top-k 12

# Szybszy wariant 4.5B zamiast domyślnego 11B
.venv/bin/python scripts/chat.py \
  --model models/Bielik-4.5B-v3.0-Instruct-mlx \
  --adapter-path adapters/bielik-kadry-lora-v3-iter25
```

**4.5B vs 11B:** oba warianty są wytrenowane i dostępne. Domyślny jest
**11B** kosztem szybkości (~26 tok/s vs ~50+ tok/s) i pamięci (~8.3GB
vs ~4-5GB peak). Do kroku 6/16 4.5B potrafił przesłonić poprawny
kontekst z RAG błędną liczbą z wag przy pytaniach z kilkoma blisko
siebie leżącymi wartościami w tym samym akcie (np. PPK: 1,5% vs 2,5%) —
**Krok 25 naprawił oba znane przypadki tego typu** kontrastującymi
przykładami treningowymi, ale nie gwarantuje niezawodności na
dowolnych, nieprzewidzianych kombinacjach takich pytań (patrz tam).
Użyj `--model` + `--adapter-path` jak wyżej, żeby przełączyć się na
szybszy 4.5B.

**Kontekst rozmowy:** tryb interaktywny (we wszystkich trzech
wariantach — `chat.py`, `chat_lmstudio.py`, `chat_cuda.py`) pamięta
poprzednie pytania i odpowiedzi w ramach jednej sesji. Wpisz `/nowy`,
żeby zacząć nowy wątek i wyczyścić historię, albo ogranicz jej długość
flagą `--max-turns` (domyślnie 6 par pytanie/odpowiedź) — dłuższa
historia + fragmenty RAG doklejane w każdej turze mogą zapełnić okno
kontekstu modelu.

**Pytania o przebieg rozmowy** (np. "przypomnij, o czym mówiliśmy",
"zacytuj to z pierwszego pytania") są rozpoznawane heurystycznie i
odpowiadane na podstawie historii rozmowy, bez wstrzykiwania fragmentów
RAG — w testach okazało się, że doklejanie fragmentów wyszukanych od
nowa (i przez to często niezwiązanych z takim pytaniem) potrafiło
skutecznie zablokować modelowi dostęp do własnej historii, szczególnie
po wcześniejszej odmowie w tej samej rozmowie (patrz PROGRESS.md, krok
10 — tam też opis nieudanych prób naprawy samą redakcją system promptu
i większym modelem). Rozpoznawanie nie jest wyczerpujące — przy
nietypowym sformułowaniu takiego pytania może nadal paść odmowa; w
takiej sytuacji pomaga `/nowy`.

**Pytania uzupełniające** (np. "a po 15 latach?" jako kontynuacja
pytania o urlop) same w sobie bywają zbyt krótkie, żeby RAG trafił na
właściwy przepis — w takim wypadku wyszukiwanie próbuje doklejać po
kolei ostatnie kilka pytań z historii i zostaje przy tym, które dało
najlepsze dopasowanie (patrz PROGRESS.md, krok 11). Bez tego model
potrafił "dopowiadać" nieistniejące szczegóły (np. zmyślony numer
punktu w artykule), zamiast wprost przyznać brak groundingu.

**Upload własnego pliku PDF** (we wszystkich trzech wariantach): wpisz
`/plik <ścieżka>` w trybie interaktywnym albo podaj `--file <ścieżka>`
przy starcie, żeby dołączyć np. własną umowę o pracę czy regulamin jako
dodatkowy kontekst obok RAG-a. Obsługiwane są na razie tylko PDF-y
tekstowe (tekst wyciągany wprost z pliku) — skany/obrazy (OCR) są poza
obecnym zakresem i zwrócą czytelny błąd zamiast po cichu zignorować
plik. Fragmenty z wgranego pliku są wyraźnie oznaczane w promptcie jako
"treść pliku użytkownika" i model jest instruowany, żeby nie mylić ich
z obowiązującym prawem — w testach poprawnie rozróżniał zapis w
przykładowej umowie od przepisów Kodeksu pracy i zwracał uwagę na
rozbieżność między nimi. Plik jest indeksowany tylko w pamięci na czas
sesji (nigdy nie trafia do stałej bazy ustaw) i można go podmienić w
dowolnym momencie kolejnym `/plik`.

```bash
.venv/bin/python scripts/chat.py --file umowa.pdf
# albo w trakcie rozmowy:
# Ty: /plik umowa.pdf
```

**Prawo obowiązujące na dany dzień w przeszłości** (we wszystkich trzech
wariantach): wpisz `/data RRRR-MM-DD` w trybie interaktywnym albo podaj
`--as-of RRRR-MM-DD` przy starcie, żeby dostać odpowiedź opartą o wersję
przepisów obowiązującą na tę konkretną datę, nie stan bieżący -- przydatne
przy przeliczaniu zaległego urlopu/wynagrodzenia, korektach list płac za
miniony okres czy sporach o zdarzenie sprzed lat (patrz PROGRESS.md, Krok
18-20). Wymaga wcześniejszego uruchomienia `download_acts_history.py` +
`build_rag_index.py --include-history` (krok 2b wyżej) -- bez tego `--as-of`
nie ma czego przeszukiwać i pytanie efektywnie wraca do stanu bieżącego. Samo
`/data` (bez daty) czyści ustawioną datę z powrotem na stan bieżący; `/nowy`
świadomie NIE czyści ustawionej daty (tak jak nie czyści wgranego pliku) --
data utrzymuje się między wątkami rozmowy, dopóki nie wyczyścisz jej jawnie.

```bash
.venv/bin/python scripts/chat.py --as-of 2018-01-15 --prompt "Ile wynosi minimalne wynagrodzenie za pracę?"
# albo w trakcie rozmowy:
# Ty: /data 2018-01-15
# Ty: /data            (powrót do stanu bieżącego)
```

**Ograniczenie warte znajomości (patrz PROGRESS.md, Krok 18-24):** dane
historyczne są dostępne tylko od pierwszego ogłoszonego tekstu jednolitego
danej ustawy (dla części ustaw to prawie cały okres ich obowiązywania, dla
innych -- np. ustawy o minimalnym wynagrodzeniu -- dopiero od ok. 2015, mimo
że ustawa obowiązuje od 2003). Dla dat wcześniejszych model ma wprost
przyznać się do braku danych zamiast zgadywać, ale to zachowanie nie jest
niezawodne, gdy jakiś fragment TECHNICZNIE pokrywa żądaną datę, lecz nie
zawiera odpowiedzi na pytanie -- w takim przypadku model bywał
niekonsekwentny (raz wiernie cytował nietrafny fragment, raz podawał
wiarygodnie brzmiącą, ale niepodpartą fragmentem liczbę z pamięci). Trzy
konkretne przypadki, które to ujawniły -- kwota minimalnego wynagrodzenia,
limit 30-krotności składek ZUS i przeciętne wynagrodzenie GUS, których same
ustawy nie zawierają -- są już naprawione (krok 2c wyżej, Krok 21-23).
Traktuj takie, wąskie przypadki z ostrożnością i zawsze
zweryfikuj konkretną kwotę w oficjalnym źródle.

### Alternatywa: RAG + model z LM Studio (`scripts/chat_lmstudio.py`)

Jeśli masz już jakiś model załadowany w [LM Studio](https://lmstudio.ai/)
(np. Bielik w formacie GGUF) i nie chcesz go dublować w MLX, ten
skrypt daje dokładnie ten sam RAG i ten sam prompt systemowy co
`chat.py`, ale generację odsyła do lokalnego serwera LM Studio zamiast
ładować model przez MLX:

```bash
# W LM Studio: zakładka "Developer" -> "Start Server" (domyślnie port 1234)

.venv/bin/python scripts/chat_lmstudio.py
.venv/bin/python scripts/chat_lmstudio.py --prompt "Ile wynosi zasiłek dla bezrobotnych?"

# Jeśli LM Studio ma załadowanych kilka modeli naraz, wskaż który:
.venv/bin/python scripts/chat_lmstudio.py --model bielik-11b-v3.0-instruct

# Inny port/host serwera LM Studio:
.venv/bin/python scripts/chat_lmstudio.py --url http://localhost:1234/v1
```

Bez `--model` skrypt sam wypisze listę załadowanych w LM Studio modeli
i użyje pierwszego z nich.

**Różnica względem `chat.py`:** ten wariant **nie korzysta z naszego
douczonego LoRA** — adaptery w `adapters/` są w formacie MLX i nie da
się ich doczepić do modelu GGUF załadowanego w LM Studio. Dostajesz
sam RAG (7 ustaw) + Twój model z LM Studio, bez wyuczonego stylu/
kalibracji z kroku 4b–4c. W testach samo grounding przez RAG
wystarczyło jednak do trafnych, precyzyjnie cytowanych odpowiedzi
(patrz PROGRESS.md, krok 7), a późniejsze pomiary nie wykazały, żeby
adapter coś dodawał (patrz "Jak to działa" i "Co zmierzono, a czego nie").

### Linux/Windows + karta NVIDIA (CUDA)

MLX (a więc `chat.py` i `mlx_lm.lora`) działa tylko na Apple Silicon.
Dla maszyn z kartą NVIDIA jest równoległy stos oparty o `transformers`
+ `peft` + `bitsandbytes` (kwantyzacja 4-bit) + `trl` (`SFTTrainer`)
zamiast `mlx-lm` — te same dane treningowe (`data/finetune/`), zbliżone
hiperparametry co w PROGRESS.md (krok 4b/6, ale inny domyślny
scheduler LR niż `mlx_lm.lora` — patrz Krok 26), ten sam RAG i prompt
systemowy (`scripts/prompt.py`).

**Przetestowane end-to-end na RunPod (instancja A40, 48GB VRAM)** —
patrz PROGRESS.md, Krok 26, po pełny opis: dwa bugi w API znalezione i
naprawione (`SFTTrainer` bez `processing_class`, chat_template Bielika
niekompatybilny z `assistant_only_loss`), analiza różnicy w
zachowaniu treningu względem MLX (scheduler LR), oraz porównanie
4.5B/11B — **zalecany jest model 11B**, 4.5B ma udokumentowaną
tendencję do przesłaniania niepełnego kontekstu z RAG zmyśloną, ale
pewnie brzmiącą liczbą z wag.

```bash
# Torch z obsługą CUDA zainstaluj osobno wg https://pytorch.org/get-started/locally/
# (NIE używaj requirements.txt -- ten ciągnie za sobą mlx-lm, Apple-only)
.venv/bin/pip install -r requirements-common.txt -r requirements-cuda.txt

# Fine-tuning LoRA (odpowiednik kroku 4 wyżej, ale przez transformers+peft+trl)
# -- domyślnie 11B, patrz wyżej dlaczego
.venv/bin/python scripts/train_lora_cuda.py

# Czat (RAG + doduczony model) -- transformers+peft zamiast MLX
.venv/bin/python scripts/chat_cuda.py
.venv/bin/python scripts/chat_cuda.py --prompt "Ile dni urlopu przysługuje po 10 latach pracy?"
```

Konto Hugging Face z zaakceptowaną licencją Bielika jest wymagane tak
samo jak w wariancie MLX (patrz "Wymagania" wyżej) — tu nie ma
osobnego kroku konwersji, `transformers` ładuje i kwantyzuje model
bezpośrednio przy starcie.

Samo wyszukiwanie RAG (bez odpalania modelu językowego) można
przetestować szybciej przez:

```bash
.venv/bin/python scripts/rag_search.py "okres wypowiedzenia po 5 latach"
```

## Zakres wiedzy i ograniczenia

Baza wiedzy obejmuje obecnie siedem ustaw (1661 artykułów, stan pobrania z 2026-11-17):
- Kodeks pracy
- Ustawa o systemie ubezpieczeń społecznych
- Ustawa o minimalnym wynagrodzeniu za pracę
- Ustawa o świadczeniach pieniężnych z ubezpieczenia społecznego w
  razie choroby i macierzyństwa ("ustawa zasiłkowa" — L4, zasiłek
  macierzyński/opiekuńczy)
- Ustawa o podatku dochodowym od osób fizycznych (PIT)
- Ustawa o pracowniczych planach kapitałowych (PPK)
- Ustawa o rynku pracy i służbach zatrudnienia (zasiłek dla
  bezrobotnych i pokrewne świadczenia — następczyni uchylonej z dniem
  2025-06-01 ustawy o promocji zatrudnienia i instytucjach rynku pracy)

Poza tym zakresem (np. VAT, ZUS dla działalności gospodarczej,
prawo spółek) model jest trenowany tak, by przyznać się do niewiedzy
zamiast zgadywać — ale to nie jest gwarancja stuprocentowa, zwłaszcza
dla pytań mieszających kilka tematów naraz.

**Konkretne kwoty ustalane osobnymi rozporządzeniami, nie samymi
ustawami.** Część aktualnych, "twardych" liczb nie jest zapisana wprost w
ustawach z listy wyżej — ustala je osobne, coroczne rozporządzenie/
obwieszczenie/komunikat. **Wysokość minimalnego wynagrodzenia i minimalnej
stawki godzinowej** (2004-2027, `download_wage_regulations.py`), **roczny
limit 30-krotności** podstawy wymiaru składek emerytalno-rentowych
(1999-2027, `download_zus_limit_regulations.py`) i **przeciętne
wynagrodzenie w gospodarce narodowej** (2003-2025,
`download_avg_wage_regulations.py`) są już dodane osobno (krok 2c wyżej) i
działają razem z `--as-of`/`/data` — dla przeciętnego wynagrodzenia zwróć
uwagę, że pytanie o rok RRRR wymaga `--as-of` z datą w RRRR+1 (to wtedy
fakt za rok RRRR został ogłoszony, patrz PROGRESS.md Krok 23). Dla pytań o
kwoty spoza `ACTS`/dodanych rozporządzeń RAG trafia w powiązany, ale
niewystarczający fragment ustawy, a model bywa niekonsekwentny (patrz
PROGRESS.md, Krok 20) — zawsze zweryfikuj tego typu kwoty w oficjalnym
źródle. Domyślne `--top-k` zostało podniesione do 8 właśnie dlatego, że
nawet dla już dodanych rozporządzeń trafny fragment czasem nie mieścił się
w dawnym domyślnym `--top-k 5` (konkurował z tematycznie bliskimi
artykułami tej samej ustawy bazowej, sprawdzone dwukrotnie — Krok 21 i 22,
domknięte w Kroku 24) — jeśli mimo to odpowiedź na pytanie o konkretną
kwotę wygląda na nietrafną, spróbuj jeszcze wyższego `--top-k`.

**Przepisy na 2027 rok.** Rozporządzenie o płacy minimalnej na 2027 r. (z 14 września
2026, DU/2026/1213) i obwieszczenie o limicie 30-krotności składek na 2027 r. (z 3 listopada
2026, MP/2026/1098) są w bazie, ale obowiązują od 2027-01-01, więc zwykłe pytanie widzi
stan na dziś, czyli stawki z 2026 roku. Żeby zapytać o 2027, użyj `--as-of 2027-01-02`
(albo `/data 2027-01-02`). Tak samo zachowują się inne przepisy, które wejdą w życie
później — wtedy ustawa ma w tekście ujednoliconym osobne, jeszcze nieobowiązujące brzmienie.

**Znany problem w ustawie o systemie ubezpieczeń społecznych.** W PDF-ie tekstu
ujednoliconego (stan 2026-11-09) jest 16 znaków `<` i tylko 15 `>`. Kod usuwa przyszłe
fragmenty (wchodzące w życie od 1.01.2027) zapisane w nawiasach `< >`, więc ten brak sprawia,
że art. 68 ust. 2 trafia do bazy w obu wersjach, obecnej i przyszłej. Skrypt wypisuje wtedy
ostrzeżenie o niesparowanych nawiasach. Nienaprawione; przy pytaniu o art. 68 sprawdź tekst w
oficjalnym źródle.

**Poprawna numeracja artykułów z indeksem górnym (np. Art. 11¹).**
Kodeks pracy od lat jest nowelizowany przez wstawianie nowych artykułów
z indeksem górnym między istniejące numery, zamiast przenumerowania
całego kodeksu — `pypdf` (podstawowa biblioteka do ekstrakcji PDF)
gubi to formatowanie, więc np. "Art. 11¹" i prawdziwy, trzycyfrowy
"Art. 111" stawały się nierozróżnialne w wydobytym tekście. Naprawione
przez dodatkową analizę geometrii znaków w PDF-ie (`pdfplumber`:
rozmiar i pozycja czcionki wykrywają indeks górny) — dotyczyło 204
artykułów w Kodeksie pracy. Przy okazji naprawiono też dwa pokrewne
problemy: (1) kilka artykułów było całkowicie pomijanych, bo ich
nagłówek trafiał się w środku akapitu bez złamania linii (np. Art. 22³
— monitoring poczty elektronicznej pracownika — był wtopiony w treść
Art. 22²); (2) fragmenty prawa uchwalonego, ale jeszcze nieobowiązującego
(oznaczane w tekście ujednoliconym nawiasami "< >") były błędnie
włączane do indeksu, jakby już obowiązywały — 16 takich artykułów
usunięto z ustawy o systemie ubezpieczeń społecznych (Art. 85c-85j,
przyszłe zmiany w orzecznictwie lekarskim ZUS). Szczegóły w PROGRESS.md,
krok 12-13 — po pełnym audycie wszystkie siedem ustaw ma teraz w 100%
zweryfikowaną numerację artykułów (liczba nagłówków z niezależnej analizy
geometrii PDF-a zgadza się z liczbą wyodrębnionych artykułów w każdym akcie).

**Uwaga przy dodawaniu kolejnych aktów:** zawsze zweryfikuj status
aktu przez `GET https://api.sejm.gov.pl/eli/acts/{publisher}/{year}/{position}`
(pola `status` i `inForce`) przed dopisaniem go do `ACTS` — sama nazwa
ustawy, nawet poprawna historycznie, może wskazywać na akt już
uchylony i zastąpiony nowszym (tak było z ustawą o promocji
zatrudnienia, patrz PROGRESS.md krok 5).

Aby dodać kolejny akt prawny, dopisz wpis (`short`, `publisher`, `year`,
`position`, `domains`) do `data/acts_config.json`, uruchom
`scripts/download_acts.py` ponownie, a następnie przebuduj indeks RAG
(krok 2 wyżej) — model nie wymaga wtedy zmian, bo to RAG dostarcza faktów.

### Sprawdzanie aktualności bazy (`scripts/check_acts_freshness.py`)

Ustawy się zmieniają — bywają nowelizowane albo, jak pokazał przypadek
ustawy o promocji zatrudnienia (patrz wyżej), całkowicie uchylane i
zastępowane nowymi aktami. Ten skrypt automatyzuje kontrolę, którą
wcześniej robiliśmy ręcznie: dla każdej ustawy w bazie pyta ELI API,
czy status się nie zmienił, czy akt nie został uchylony, i czy nie
pojawił się nowszy tekst ujednolicony niż ten, który mamy pobrany.
Nic nie modyfikuje, tylko raportuje (kod wyjścia 1, jeśli coś wymaga
uwagi):

```bash
.venv/bin/python scripts/check_acts_freshness.py
```

Warto uruchamiać okresowo (np. raz na kwartał) i zawsze przed
dodaniem kolejnego aktu do `ACTS`. Uruchamia się też automatycznie co
tydzień przez GitHub Actions (`.github/workflows/acts-freshness.yml`) --
porównuje na bieżąco commitowane `data/raw/*_meta.json` (jedyne pliki z
`data/raw/`, które NIE są gitignored -- same metadane, kilkadziesiąt KB
na akt, bez treści PDF) ze świeżym stanem ELI API; failed run = mail z
GitHub o wykrytej rozbieżności.

## Testy

```bash
.venv/bin/pip install -r requirements-common.txt -r requirements-dev.txt
.venv/bin/pytest
```

Uruchamiają się też automatycznie przy każdym push/PR do `main`
(`.github/workflows/tests.yml`, GitHub Actions) -- na świeżym klonie
repo bez lokalnie pobranej bazy (`data/raw/`, `data/processed/`), więc
testy oznaczone `skipif` (patrz niżej) są tam pomijane.

Testy w `tests/` pokrywają logikę kontekstu rozmowy i ekstrakcji
artykułów z PDF-ów (`scripts/prompt.py`, `scripts/download_acts.py`) --
głównie na małych, ręcznie skonstruowanych fragmentach tekstu, więc
większość działa bez pobranej bazy. Kilka testów to regresje konkretnych
błędów znalezionych w audycie (patrz PROGRESS.md, krok 12-13, np. "czy
Art. 22³ Kodeksu pracy nie jest wciąż wtopiony w Art. 22²") — te
wymagają lokalnie pobranej bazy (`data/raw/`, `data/processed/`) i są
automatycznie pomijane (`skipped`), jeśli jej nie ma.

## Co zmierzono, a czego nie

Dla uczciwego obrazu jakości. Wszystkie zbiory pytań ułożył autor (z pomocą AI), są małe, a odpowiedzi
modelu oceniał głównie inny model, nie prawnik ani kadrowa.

| Co | Wynik | Zastrzeżenie |
|---|---|---|
| Wyszukiwanie (reranker + RRF), 26 pytań kadrowych | top-1/3/8 = 20/26/26 po odświeżeniu bazy 2026-11-17 (Krok 32); w Kroku 28: 18/25/26, sam kosinus 17/22/25 | etykiety sprawdzone po treści artykułów, ale to wynik retrievalu, nie jakości odpowiedzi |
| Pytania uzupełniające ("A po 15 latach?"), `chat_lmstudio.py` | właściwy artykuł w top-8: 14/14 (wcześniej 5/14) | 14 + 26 pytań, jeden model przepisujący |
| Adapter LoRA a sam model z RAG (Bielik v3.1, 76 pytań kadrowych, sędzia Qwen w obu kolejnościach) | 22 : 18 na korzyść adaptera, 29 remisów, różnica nieistotna (p = 0,32) | sędzia = model, jedna dziedzina |
| Adaptery produkcyjne a Bielik v3.1 bez adaptera (3 inne dziedziny, 70 pytań) | 18 : 20, bez różnicy | j.w. |
| Qwen3.8-27B a Bielik v3.0 (76 pytań kadrowych) | 65 : 0 przy 7 remisach | sędzią był Qwen, odpowiedzi Qwena są 2,4 razy dłuższe; w tym repo Qwen nietestowany |
| Poprawność merytoryczna odpowiedzi przez prawnika / kadrową | **nie wykonano** | to największa luka |

Pełne opisy: Kroki 106, 113, 118, 121 i 122 w
[PROGRESS.md projektu następcy](https://github.com/marduk112/prawo-polskie-rag/blob/main/PROGRESS.md) oraz
Kroki 28–32 w [PROGRESS.md](PROGRESS.md) tego repo.

## Zastrzeżenie prawne

To narzędzie ma charakter pomocniczy i edukacyjny. Odpowiedzi opierają
się na tekście ustaw pobranym w dniu wykonania kroku 2 (patrz
PROGRESS.md) i mogą nie uwzględniać późniejszych nowelizacji ani
najnowszego orzecznictwa. Narzędzie **nie zastępuje** porady prawnej
ani konsultacji z księgową/księgowym czy doradcą podatkowym — w
sprawach spornych, o wysokiej stawce finansowej lub niejednoznacznych
skonsultuj się ze specjalistą.

## Zbudowane z pomocą AI

Ten projekt (kod, skrypty, dokumentacja) powstał we współpracy z
[Claude Code](https://claude.com/claude-code) (Anthropic) jako
asystentem programistycznym. Decyzje architektoniczne, wybór ustaw,
testy jakości i weryfikacja faktów prawnych (patrz PROGRESS.md) były
wykonywane i sprawdzane na bieżąco, ale duża część implementacji i
research'u (np. weryfikacja identyfikatorów ELI, testy fine-tuningu)
została wykonana przez model.

## Wyszukiwanie: reranker i fuzja RRF

Od Kroku 28 wyszukiwanie ma dwa etapy: podobieństwo kosinusowe (embeddingi `sdadas/mmlw-retrieval-roberta-large`) wybiera
30 kandydatów, a cross-encoder `sdadas/polish-reranker-base-ranknet` ocenia je jeszcze raz; oba rankingi łączy fuzja RRF
(`scripts/rag_search.py`). Model rerankera (Apache-2.0 wg karty na HF) pobiera się sam przy pierwszym uruchomieniu --
konto HF nie jest potrzebne. Na 26 pytaniach kadrowych z `data/retrieval_battery.json` (`scripts/check_retrieval_regressions.py`):

| wariant | top-1 | top-3 | top-8 |
|---|---|---|---|
| tylko kosinus (`--no-rerank`) | 17 | 22 | 25 |
| kosinus + reranker + RRF (domyślnie) | 18 | 25 | 26 |

Próba jest mała, a pytania ułożył autor, więc to wskazówka, nie dowód (szczegóły w PROGRESS.md, Krok 28; po odświeżeniu bazy i przeniesieniu aktualnego ekstraktora ustaw, Krok 32, ta sama bateria dała 20/26/26). Pomysł i
implementacja pochodzą z repozytorium wielodziedzinowym [prawo-polskie-rag](https://github.com/marduk112/prawo-polskie-rag).

## Pytania uzupełniające w rozmowie ("A po 15 latach?")

Krótkie pytanie uzupełniające nie zawiera słów kluczowych, więc samo wyszukiwanie często pomija właściwy artykuł. W
`chat_lmstudio.py` (tryb rozmowy) model przed wyszukaniem przepisuje takie pytanie na samodzielne -- albo zwraca je bez
zmian, jeśli już jest samodzielne. Na 14 pytaniach uzupełniających właściwy artykuł jest w top-8 w 14/14 przypadków
(poprzednio 5/14), po obcym wątku w środku rozmowy w 6/6 (poprzednio 3/6), a 26 pytań samodzielnych nie pogorszyło się.
Przetestowane z Bielikiem v3.1 8-bit w LM Studio; wyłączane flagą `--no-rewrite-followups`. W `chat.py` (MLX) funkcja jest
**opcjonalna** (`--rewrite-followups`, domyślnie wyłączona): domyślny Bielik v3.0 zamiast przepisywać pytanie odpowiadał na nie,
a z Bielikiem v3.1 8-bit działa poprawnie (`--model models/Bielik-11B-v3.1-Instruct-mlx-8bit --no-adapter --rewrite-followups`).
`chat_cuda.py` bez zmian (nieprzetestowane). Szczegóły: PROGRESS.md, Krok 29 i 30.

## Licencja

Kod tego repozytorium jest dostępny na licencji [Apache 2.0](LICENSE).

Modele Bielika (`speakleash/Bielik-*`) są udostępniane przez SpeakLeash
osobno, również na licencji Apache 2.0, ale nie są częścią tego
repozytorium (pobierane oddzielnie, patrz sekcja "Instalacja"). Teksty
ustaw pobierane przez `scripts/download_acts.py` to oficjalne akty
normatywne — w Polsce nieobjęte prawem autorskim (art. 4 ustawy o
prawie autorskim i prawach pokrewnych).
