# Asystent kadrowo-płacowy (Bielik + RAG + LoRA)

Lokalny (offline po pierwszym pobraniu) asystent odpowiadający na pytania
z zakresu polskiego prawa pracy i ubezpieczeń społecznych. Działa
natywnie na Apple Silicon (MLX), bez wysyłania niczego do chmury.

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

- macOS na Apple Silicon (MLX korzysta z GPU przez Metal)
- Python 3.12 (system może mieć starszą wersję — patrz PROGRESS.md,
  krok 0)
- Konto Hugging Face z zaakceptowaną licencją modelu (oba warianty są
  *gated* -- licencję trzeba zaakceptować osobno dla każdego z nich,
  na stronie repo na HF, zanim `hf download`/`mlx_lm.convert` zadziała):
  - [`speakleash/Bielik-11B-v3.0-Instruct`](https://huggingface.co/speakleash/Bielik-11B-v3.0-Instruct) (domyślny)
  - [`speakleash/Bielik-4.5B-v3.0-Instruct`](https://huggingface.co/speakleash/Bielik-4.5B-v3.0-Instruct) (szybszy wariant, opcjonalny)

## Instalacja

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Zaloguj się do Hugging Face (token z https://huggingface.co/settings/tokens)
.venv/bin/hf auth login
```

## Odtworzenie całego pipeline'u od zera

Repozytorium zawiera już gotowe dane (`data/`), model (`models/`) i
adapter LoRA (`adapters/`) — więc do samego *używania* asystenta
wystarczy sekcja "Użycie" niżej. Poniższe kroki są potrzebne tylko,
jeśli chcesz odtworzyć pipeline od podstaw (np. dodać nową ustawę albo
przetrenować LoRA od nowa).

```bash
# 1. Pobranie aktów prawnych z ELI API Sejmu -> data/raw/, data/processed/
.venv/bin/python scripts/download_acts.py

# 2. Zbudowanie indeksu RAG (embeddingi artykułów) -> data/processed/rag_index*.
.venv/bin/python scripts/build_rag_index.py

# 3. Pobranie i konwersja Bielika do MLX (kwantyzacja 4-bit) -> models/
# (dla wariantu 4.5B podmień hf-path/mlx-path na Bielik-4.5B-v3.0-Instruct)
.venv/bin/hf download speakleash/Bielik-11B-v3.0-Instruct  # najpierw pełny snapshot, patrz uwaga niżej
.venv/bin/mlx_lm.convert \
  --hf-path speakleash/Bielik-11B-v3.0-Instruct \
  --mlx-path models/Bielik-11B-v3.0-Instruct-mlx \
  -q

# 4. Fine-tuning LoRA na data/finetune/{train,valid}.jsonl -> adapters/
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

Samo wyszukiwanie RAG (bez odpalania modelu językowego) można
przetestować szybciej przez:

```bash
.venv/bin/python scripts/rag_search.py "okres wypowiedzenia po 5 latach"
```

## Zakres wiedzy i ograniczenia

Baza wiedzy obejmuje obecnie siedem ustaw (1650 artykułów):
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
dodaniem kolejnego aktu do `ACTS`.

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
