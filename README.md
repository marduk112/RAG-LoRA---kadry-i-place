# Asystent kadrowo-płacowy (Bielik + RAG + LoRA)

Lokalny (offline po pierwszym pobraniu) asystent odpowiadający na pytania
z zakresu polskiego prawa pracy i ubezpieczeń społecznych, bez wysyłania
niczego do chmury. Natywnie na Apple Silicon (MLX) — plus alternatywne
ścieżki dla Linuksa/Windows z kartą NVIDIA (CUDA, patrz niżej;
nieprzetestowane) i dla dowolnego modelu już załadowanego w LM Studio.

Dziennik tego, jak to powstawało krok po kroku — z napotkanymi
pułapkami i decyzjami — jest w [PROGRESS.md](PROGRESS.md). Ten plik to
skrócona instrukcja "jak tego używać".

## Jak to działa

Dwa niezależne mechanizmy działają razem, celowo:

1. **RAG (retrieval-augmented generation)** — pytanie użytkownika jest
   embedowane i porównywane z bazą artykułów pobranych ustaw
   (`data/processed/all_articles.json`). Najtrafniejsze fragmenty
   trafiają jako kontekst do modelu. To źródło **faktów** — aktualnych,
   źródłowych tekstów przepisów.
2. **LoRA fine-tuning** — model **Bielik** (polski LLM od SpeakLeash;
   domyślnie wariant **11B-v3.0-Instruct**, dostępny też szybszy
   **4.5B-v3.0-Instruct**) jest doduczony na przykładach pytanie/
   odpowiedź z tej dziedziny. To źródło **stylu**: zwięzłe odpowiedzi z
   cytowaniem artykułu, oraz — co ważne — nawyk przyznawania się do
   niewiedzy zamiast zgadywania, gdy kontekst nie wystarcza.

**Dlaczego oba naraz, a nie tylko fine-tuning?** Testy pokazały, że
sam doduczony model (bez kontekstu RAG) nie jest wystarczająco
niezawodny w precyzyjnych progach liczbowych (np. dokładnie która
granica stażu pracy daje 1 miesiąc, a która 3 miesiące wypowiedzenia)
i przy pytaniach spoza wytrenowanego zakresu potrafi pewnie zmyślać
fakty. Po dodaniu kontekstu RAG te same pytania wypadały poprawnie.
Szczegóły i konkretne przykłady testów — patrz PROGRESS.md, kroki 4b–4c.

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
| `chat_cuda.py` (CUDA) | Linux/Windows + karta NVIDIA | ⚠️ Nieprzetestowane (brak dostępu do sprzętu NVIDIA) | Sterowniki CUDA, `torch` z obsługą CUDA, konto HF z licencją Bielika |
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
(iteracja 50 dla 4.5B, iteracja 25 dla 11B) — dalszy trening tylko
przeuczał model na 35-elementowym zbiorze treningowym. Zalecane jest
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

# Większa liczba fragmentów kontekstu
.venv/bin/python scripts/chat.py --top-k 8

# Szybszy wariant 4.5B zamiast domyślnego 11B
.venv/bin/python scripts/chat.py \
  --model models/Bielik-4.5B-v3.0-Instruct-mlx \
  --adapter-path adapters/bielik-kadry-lora-iter50
```

**4.5B vs 11B:** oba warianty są wytrenowane i dostępne. Domyślny jest
**11B** — daje zauważalnie dokładniejsze odpowiedzi na pytania
graniczne (patrz PROGRESS.md, krok 6) kosztem szybkości (~26 tok/s vs
~50+ tok/s) i pamięci (~8.3GB vs ~4-5GB peak). Użyj `--model` +
`--adapter-path` jak wyżej, żeby przełączyć się na szybszy 4.5B.

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
(patrz PROGRESS.md, krok 7).

### Linux/Windows + karta NVIDIA (CUDA) — ⚠️ NIEPRZETESTOWANE

MLX (a więc `chat.py` i `mlx_lm.lora`) działa tylko na Apple Silicon.
Dla maszyn z kartą NVIDIA jest równoległy stos oparty o `transformers`
+ `peft` + `bitsandbytes` (kwantyzacja 4-bit) + `trl` (`SFTTrainer`)
zamiast `mlx-lm` — te same dane treningowe (`data/finetune/`), te same
hiperparametry co w PROGRESS.md (krok 4b/6), ten sam RAG i prompt
systemowy (`scripts/prompt.py`).

**Ten wariant nie został uruchomiony na żadnej karcie NVIDIA** — środowisko,
w którym powstał ten projekt, to Mac bez GPU CUDA. API zostało
zweryfikowane względem aktualnej dokumentacji `peft`/`trl`, ale nie
przetestowane end-to-end. Jeśli coś nie zadziała, to prawdopodobnie
drobna niezgodność wersji bibliotek — daj znać (issue) albo od razu
wyślij PR z poprawką i aktualizacją tego pliku oraz PROGRESS.md.

```bash
# Torch z obsługą CUDA zainstaluj osobno wg https://pytorch.org/get-started/locally/
# (NIE używaj requirements.txt -- ten ciągnie za sobą mlx-lm, Apple-only)
.venv/bin/pip install -r requirements-common.txt -r requirements-cuda.txt

# Fine-tuning LoRA (odpowiednik kroku 4 wyżej, ale przez transformers+peft+trl)
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

Baza wiedzy obejmuje obecnie siedem ustaw (1641 artykułów):
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

Aby dodać kolejny akt prawny, dopisz wpis do listy `ACTS` w
`scripts/download_acts.py`, uruchom go ponownie, a następnie przebuduj
indeks RAG (krok 2 wyżej) — fine-tuning LoRA nie wymaga wtedy zmian,
bo to RAG dostarcza faktów.

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

## Licencja

Kod tego repozytorium jest dostępny na licencji [Apache 2.0](LICENSE).

Modele Bielika (`speakleash/Bielik-*`) są udostępniane przez SpeakLeash
osobno, również na licencji Apache 2.0, ale nie są częścią tego
repozytorium (pobierane oddzielnie, patrz sekcja "Instalacja"). Teksty
ustaw pobierane przez `scripts/download_acts.py` to oficjalne akty
normatywne — w Polsce nieobjęte prawem autorskim (art. 4 ustawy o
prawie autorskim i prawach pokrewnych).
