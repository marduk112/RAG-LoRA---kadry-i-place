# Postęp prac — asystent kadrowo-płacowy (Bielik + RAG + LoRA)

Ten plik to dziennik tego, co zostało zrobione krok po kroku, żebyś mógł
się z tym zapoznać. Docelowa dokumentacja "jak tego używać" trafi do
README.md na końcu projektu.

## Krok 0-1: Środowisko

- Środowisko testowe: macOS 26 (Tahoe, build 25G76), Apple M4 Pro,
  64GB pamięci zunifikowanej (`sw_vers`, `sysctl -n machdep.cpu.brand_string`,
  `sysctl -n hw.memsize`). Cały projekt (RAG, konwersja modeli,
  fine-tuning LoRA, oba warianty Bielika) był rozwijany i testowany
  wyłącznie na tej maszynie -- inne wersje macOS/chipy Apple Silicon
  (M1/M2/M3) powinny działać (MLX wspiera cały Apple Silicon), ale nie
  zostały zweryfikowane. Wariant CUDA (krok 9) nie był testowany w
  ogóle, z braku dostępu do sprzętu NVIDIA.
- System miał tylko Pythona 3.9 (za stary dla nowoczesnych bibliotek ML).
  Doinstalowano przez Homebrew **Python 3.12** i założono wirtualne
  środowisko `.venv/` w katalogu projektu (żeby nic nie zaśmiecać w systemie).
- Zainstalowano `mlx-lm` (biblioteka Apple MLX do uruchamiania i
  fine-tuningu modeli językowych natywnie na krzemie Apple, bez CUDA —
  CPU i GPU M4 Pro dzielą tę samą pamięć 64GB, więc nie ma kosztownego
  kopiowania danych jak na PC z osobną kartą graficzną).
- Sprawdzono realną, aktualną składnię komend `mlx_lm.convert` i
  `mlx_lm.lora` (`--help` zainstalowanej wersji 0.31.3), żeby nie
  opierać się na nieaktualnej wiedzy z pamięci.

## Krok 2: Pobieranie aktów prawnych (`scripts/download_acts.py`)

Źródło: oficjalne, darmowe **ELI API Sejmu** (`api.sejm.gov.pl/eli`).

Pobrane na start 3 ustawy:
- Kodeks pracy (Dz.U. 1974 nr 24 poz. 141) — 475 artykułów
- Ustawa o systemie ubezpieczeń społecznych (Dz.U. 1998 nr 137 poz. 887) — 207 artykułów
- Ustawa o minimalnym wynagrodzeniu za pracę (Dz.U. 2002 nr 200 poz. 1679) — 17 artykułów

**Ważna pułapka, na którą trafiliśmy i naprawiliśmy:** endpoint
`/text.html` tego API zwraca tekst PIERWOTNY ustawy (z dnia uchwalenia),
a NIE tekst uwzględniający późniejsze nowelizacje. Wykryto to sprawdzając
art. 154 Kodeksu pracy (wymiar urlopu) — `/text.html` zwracał formułę
z 1974 r. ("14/17/20/26 dni roboczych" zależnie od stażu), podczas gdy
faktycznie obowiązująca od dawna treść to "20/26 dni" zależnie od tego,
czy staż wynosi mniej czy co najmniej 10 lat. Zweryfikowano to
wyszukiwaniem w sieci i porównaniem z niezależnymi źródłami prawnymi.

Rozwiązanie: skrypt korzysta zamiast tego z pliku oznaczonego w
metadanych aktu jako typ **"U"** (tekst ujednolicony roboczy,
przygotowywany i aktualizowany przez Kancelarię Sejmu po każdej
nowelizacji) — to PDF, więc trzeba go było parsować (`pypdf`), czyścić
z nagłówków/stopek Kancelarii Sejmu i dzielić na artykuły wzorcem
"Art. N." na początku linii (wielka litera zawsze zaczyna nagłówek
artykułu, małą literą "art." pisze się tylko odniesienia wewnątrz tekstu).

Efekt: `data/processed/all_articles.json` — 699 fragmentów (jeden
artykuł = jeden rekord JSON) z polami: `id`, `act_short`, `act_title`,
`eli`, `article`, `text`, `source_url`.

**Do rozważenia na później:** trzy ustawy to dobry rdzeń, ale do
realnej pracy kadrowej warto dodać m.in. ustawę zasiłkową (L4, zasiłek
macierzyński), ustawę o PIT (zaliczki na podatek), ustawę o PPK.
Dodanie kolejnego aktu = dopisanie wpisu do listy `ACTS` w
`scripts/download_acts.py` i ponowne uruchomienie skryptu.

## Krok 3: Prosty RAG (`scripts/build_rag_index.py`, `scripts/rag_search.py`)

Model embeddingowy: **`sdadas/mmlw-retrieval-roberta-large`** (polski
model stworzony specjalnie do wyszukiwania/retrieval, autorstwa
Sławomira Dadasa / CLARIN-PL). Wymaga prefiksu `"zapytanie: "` dla
pytań (nie dla fragmentów ustaw).

Jak to działa:
1. Długie artykuły (np. w ustawie o ubezpieczeniach społecznych są
   artykuły po kilka tysięcy znaków) są dodatkowo dzielone na
   nakładające się okna ~900 znaków, żeby sensownie się embedowały.
2. Każdy fragment zamieniany jest na wektor 1024 liczb (embedding).
3. Pytanie użytkownika embedowane jest tym samym modelem (z prefiksem).
4. Liczone jest podobieństwo kosinusowe pytania do wszystkich
   fragmentów; wyniki grupowane są z powrotem na poziom całego
   artykułu (bo inaczej długie, pocięte na kilka części artykuły
   miałyby nieuczciwie więcej "szans" na przypadkowo wysoki wynik niż
   krótkie, jednofragmentowe artykuły — to realnie zaobserwowany i
   naprawiony błąd podczas testów).

**Testy jakości wyszukiwania** (`python scripts/rag_search.py "pytanie"`):
- "okres wypowiedzenia po 5 latach" → trafnie art. 36 KP na 1. miejscu
- "minimalne wynagrodzenie za pracę" → trafnie artykuły 25/2/8/1/4 ustawy o min. wynagrodzeniu
- "kto podlega ubezpieczeniu emerytalnemu i rentowemu" → trafnie art. 9, 6b, 12, 11, 6c
- "jakie są rodzaje umów o pracę" → **słabiej** — właściwy art. 25 nie
  wpadł w top 5. To pokazuje uczciwe ograniczenie prostego wyszukiwania
  semantycznego: bardzo ogólne, abstrakcyjne pytania ("rodzaje X") mogą
  gorzej trafiać niż pytania z konkretnymi terminami z ustawy. Nie jest
  to błąd w kodzie, tylko właściwość tej metody — zostanie to opisane
  jako znane ograniczenie w README.

Pierwsze uruchomienie modelu embeddingowego pobiera go z Hugging Face
(darmowy, niegated model, ~1.4GB).

## Krok 4a: Pobranie i konwersja Bielika do MLX

Zdecydowano zacząć od mniejszego modelu **`speakleash/Bielik-4.5B-v3.0-Instruct`**
(szybszy trening/inferencja do iteracji), z opcją pobrania 11B później,
jeśli podejście się sprawdzi.

Logowanie do HF: `hf auth login --token ...` (token zapisany w
`~/.cache/huggingface/token`).

**Napotkana i naprawiona pułapka:** pierwsze uruchomienie
`mlx_lm.convert --hf-path speakleash/Bielik-4.5B-v3.0-Instruct --mlx-path
models/Bielik-4.5B-v3.0-Instruct-mlx -q` pobrało wagi i poprawnie je
zakwantyzowało (4.505 bitów/wagę), ale padło na etapie zapisu z
`IncompleteSnapshotError` — biblioteka `huggingface_hub` w kroku ładowania
modelu pobiera tylko pliki potrzebne do inferencji (safetensors, config,
tokenizer), pomijając `.gitattributes` i `README.md`; sam `mlx_lm.convert`
w kroku zapisu woła jednak `snapshot_download(..., local_files_only=True)`,
który wymaga *kompletnego* zrzutu repo i się wywala. Rozwiązanie: ręcznie
dociągnąć pełny snapshot przed konwersją: `hf download
speakleash/Bielik-4.5B-v3.0-Instruct` (bez żadnych filtrów plików), a
potem ponownie uruchomić `mlx_lm.convert` — wagi są już w cache, więc to
szybkie.

Efekt: `models/Bielik-4.5B-v3.0-Instruct-mlx/` (~2.5GB, bfloat16→4.5bit
quant).

**Szybki test bez fine-tuningu** (`mlx_lm.generate --model
models/Bielik-4.5B-v3.0-Instruct-mlx --prompt "..."`): pytanie o wymiar
urlopu przy 12 latach stażu → poprawna odpowiedź (26 dni), model już
"z pudełka" ma niezłą wiedzę ogólną o polskim prawie pracy. ~76 tok/s
generacji, peak memory ~2.8GB — dużo zapasu na LoRA.

## Krok 4b: fine-tuning LoRA

Dane treningowe: `data/finetune/train.jsonl` (35 przykładów) i
`valid.jsonl` (5 przykładów), format czatu `{"messages": [...]}`
zgodny z `mlx_lm.tuner.datasets.ChatDataset`. Przykłady są oparte na
faktycznej treści artykułów z `all_articles.json` (nie wymyślone) i
obejmują: rodzaje/treść/rozwiązanie umów o pracę, okresy wypowiedzenia,
wymiar i zasady nabywania urlopu, czas pracy, wynagrodzenie chorobowe,
zakaz obchodzenia umowy o pracę umową cywilnoprawną, niedyskryminację,
minimalne wynagrodzenie, ubezpieczenia społeczne — a także kilka
przykładów uczących model **przyznawania się do niewiedzy** zamiast
zgadywania (np. pytania o zasiłek macierzyński czy PIT, których nie ma
w bazie 3 pobranych ustaw).

Trening: `mlx_lm.lora --model models/Bielik-4.5B-v3.0-Instruct-mlx
--train --data data/finetune --batch-size 2 --iters 200 --mask-prompt
--adapter-path adapters/bielik-kadry-lora` (LoRA na ostatnich 16 z 60
warstw, rank 8 — domyślne parametry `mlx_lm.lora`).

**Ważna obserwacja z testów (przeoverfitowanie):** val loss osiągnął
minimum (1.084) już przy iter 50, a potem rósł mimo że train loss spadł
niemal do zera (0.000 przy iter 200) — klasyczny obraz przeuczenia na
tak małym zbiorze (35 przykładów, ~11+ epok przy 200 iteracjach).
Przetestowano oba checkpointy (`0000050_adapters.safetensors` vs
finalny `adapters.safetensors` z iter 200) na pytaniach spoza dosłownej
treści treningu (np. "pracownik zatrudniony dokładnie 3 lata" / "2
lata" zamiast wytrenowanych "4 lata" / "1,5 roku" / "5 lat") — **żaden
z samych checkpointów LoRA (bez kontekstu RAG) nie radził sobie
niezawodnie z granicznymi przypadkami progu 3-letniego stażu**, a przy
pytaniu spoza zakresu ustaw ("ile wynosi zasiłek dla bezrobotnych?")
finalny (iter 200) checkpoint pewnie halucynował konkretne kwoty i
przepisy (podobnie jak model bazowy bez LoRA) — utracił wyuczone
zachowanie "przyznaj się do niewiedzy" przez przeuczenie na
dosłownych przykładach. Checkpoint z iter 50 zachował to zachowanie
(odmówił zgadywania kwoty), więc **jako domyślny adapter przyjęto
`adapters/bielik-kadry-lora-iter50`** (kopia checkpointu z najlepszym
val loss). Pełna historia checkpointów została w
`adapters/bielik-kadry-lora/`.

Wniosek: 35 przykładów LoRA wystarcza, by nauczyć model *stylu*
odpowiedzi (cytowanie artykułów, zwięzłość, hedging przy niewiedzy),
ale nie wystarcza, by niezawodnie "wszyć" w wagi precyzyjne progi
liczbowe z ustaw. To nie jest awaria — to oczekiwany argument za tym,
żeby fine-tuning **nie był** jedynym źródłem faktów w tym projekcie,
tylko działał razem z RAG (patrz krok 4c niżej).

## Krok 4c: połączenie RAG + LoRA w jeden CLI (`scripts/chat.py`)

`scripts/chat.py` dla każdego pytania: (1) wyszukuje top-k fragmentów
przez `RagIndex.search()`, (2) wstrzykuje je jako kontekst do system +
user prompta z jasną instrukcją "odpowiadaj wyłącznie na podstawie
dostarczonych fragmentów, przyznaj się do niewiedzy gdy fragmenty nie
wystarczą, cytuj artykuł", (3) generuje odpowiedź douczonym modelem
(`mlx_lm.load(..., adapter_path=...)` + `mlx_lm.generate`).

**Efekt — te same graniczne pytania, które zawodziły bez kontekstu
RAG, po dodaniu RAG wypadły poprawnie:**
- "zatrudniony dokładnie 3 lata" → poprawnie "3 miesiące", art. 36 § 1
  pkt 3.
- "ile wynosi zasiłek dla bezrobotnych?" (poza zakresem pobranych
  ustaw) → model odmówił podania kwoty i odesłał do ZUS/urzędu pracy,
  zamiast (jak wcześniej) zmyślać konkretną sumę w złotówkach.

To potwierdza pierwotny zamysł projektu: LoRA odpowiada za styl i
kalibrację ("nie zgaduj"), a RAG dostarcza aktualne, źródłowe fakty do
kontekstu w momencie odpowiedzi.

Użycie:
```
python scripts/chat.py                          # tryb interaktywny
python scripts/chat.py --prompt "pytanie..."     # pojedyncze pytanie
python scripts/chat.py --no-adapter              # do porównania: bez LoRA
```

## Krok 5: rozszerzenie bazy ustaw

Dodano do `ACTS` w `scripts/download_acts.py` cztery kolejne akty i
przebudowano indeks RAG (`scripts/build_rag_index.py`) -- teraz **7
ustaw, 1650 artykułów, 4425 fragmentów** (wcześniej 3 ustawy / 699
artykułów):

- **Ustawa zasiłkowa** (DU/1999/636) -- zasiłek chorobowy, macierzyński,
  opiekuńczy.
- **Ustawa o PIT** (DU/1991/350) -- zaliczki na podatek dochodowy od
  wynagrodzeń.
- **Ustawa o PPK** (DU/2018/2215) -- pracownicze plany kapitałowe.
- **Ustawa o rynku pracy i służbach zatrudnienia** (DU/2025/620) --
  zasiłek dla bezrobotnych i pokrewne świadczenia.

**Ważna pułapka wykryta i naprawiona przy tym kroku:** pierwotnie
planowana do dodania "ustawa o promocji zatrudnienia i instytucjach
rynku pracy" (DU/2004/1001, ta sama nazwa, którą podpowiada pamięć/
wiedza ogólna) okazała się być **uchylona z dniem 2025-06-01** --
wykryto to sprawdzając pole `references.["Akty uchylające"]` w
odpowiedzi ELI API dla tego aktu, które wskazywało na
`DU/2025/620` jako akt uchylający. Dodanie starej, uchylonej ustawy do
bazy RAG wprowadziłoby nieaktualne informacje o dokładnie tym typie
pytań (zasiłek dla bezrobotnych), które ten projekt ma obsługiwać
rzetelnie. Zweryfikowano więc każdy nowo dodawany akt przez zapytanie
`GET /eli/acts/{publisher}/{year}/{position}` i sprawdzenie pól
`status` oraz `inForce` przed dopisaniem go do listy, zamiast polegać
na identyfikatorze z pamięci.

Test po przebudowie indeksu: pytanie "ile wynosi zasiłek dla
bezrobotnych" trafnie zwraca art. 224 nowej ustawy z konkretną,
aktualną kwotą (1662,00 zł / 1305,20 zł) -- czyli dokładnie ten
przypadek, który wcześniej (krok 4c) model musiał grzecznie odrzucać
z braku danych w bazie.

LoRA nie wymagało ponownego treningu przy tej zmianie -- to RAG
dostarcza faktów, adapter dalej odpowiada tylko za styl.

## Krok 6: fine-tuning na Bielik-11B-v3.0-Instruct

Pobrano i skonwertowano `speakleash/Bielik-11B-v3.0-Instruct` do MLX
(`models/Bielik-11B-v3.0-Instruct-mlx`, ~6.3GB, 4.5-bit) -- tym razem
bez pułapki z krokiem 4a, bo pełny snapshot (`hf download`) pobrano
przed konwersją. Wytrenowano LoRA na tym samym zbiorze
(`data/finetune/`), tymi samymi hiperparametrami co dla 4.5B (batch
2, 200 iteracji), ale z gęstszą ewaluacją (co 25 iteracji zamiast co
50) po doświadczeniu z krokiem 4b.

**Ten sam wzorzec przeuczenia, tylko szybszy:** początkowy val loss
11B (1.067) jest już wyraźnie niższy niż dla 4.5B (2.422) -- większy
model ma lepsze pierwotne wyczucie dziedziny. Najlepszy val loss
(0.511) wypadł już przy iter 25, po czym monotonicznie rósł do 0.807
przy iter 200, mimo że train loss spadł do ~0 już koło iter 80. Jako
domyślny adapter przyjęto checkpoint z iter 25:
`adapters/bielik11b-kadry-lora-iter25` (analogicznie do
`bielik-kadry-lora-iter50` dla 4.5B).

**Test na tych samych pytaniach granicznych co w kroku 4b/4c, tym
razem z kontekstem RAG (rozszerzonym o 7 ustaw z kroku 5):**
- "zatrudniony dokładnie 3 lata" -> poprawnie 3 miesiące, z trafnym
  dodatkowym zastrzeżeniem o wliczaniu stażu u poprzedniego
  pracodawcy przy przejściu zakładu pracy (art. 36 § 11) -- head i
  ramy odpowiedzi bogatsze niż u 4.5B.
- "ile wynosi zasiłek dla bezrobotnych" -> precyzyjnie zacytowane
  kwoty z art. 224 (1662,00 zł / 1305,20 zł) wraz z dodatkową klauzulą
  o podwyższonym zasiłku przy stażu >= 20 lat (art. 224 ust. 2) --
  szczegół, którego 4.5B nie wspomniał.

**Koszt:** ~26 tok/s generacji i ~8.3GB peak memory dla 11B+RAG+LoRA
vs ~50+ tok/s i ~4-5GB dla 4.5B -- wolniej i więcej pamięci, ale wciąż
daleko poniżej 64GB unified memory na M4 Pro.

`scripts/chat.py` dostał flagę `--model`, żeby dało się przełączać
między wariantami modelu bez edycji kodu:
```
python scripts/chat.py --model models/Bielik-11B-v3.0-Instruct-mlx \
  --adapter-path adapters/bielik11b-kadry-lora-iter25
```
Po decyzji użytkownika **11B jest teraz domyślnym modelem** w
`scripts/chat.py` (`MODEL_PATH` / `DEFAULT_ADAPTER_PATH`
zaktualizowane). Wariant 4.5B nadal dostępny przez `--model` +
`--adapter-path` dla szybszej pracy.

## Krok 7: RAG + model z LM Studio (`scripts/chat_lmstudio.py`)

Użytkownik ma już Bielika-11B (GGUF) załadowanego w LM Studio i nie
chciał go dublować w MLX. Zamiast przepisywać `chat.py`, dodano osobny
skrypt `scripts/chat_lmstudio.py`, który importuje i ponownie
wykorzystuje `SYSTEM_PROMPT` oraz `build_context()` z `chat.py` (ten
sam prompt systemowy i sposób budowania kontekstu RAG), ale zamiast
`mlx_lm.load`/`generate` woła lokalny serwer LM Studio przez REST API
zgodne z OpenAI (`POST /v1/chat/completions`).

Funkcjonalność:
- Auto-wykrywanie załadowanego modelu przez `GET /v1/models`, jeśli
  `--model` nie podano (przydatne, bo LM Studio może mieć kilka modeli
  załadowanych naraz -- w testach użytkownika: `bielik-11b-v3.0-instruct`,
  `qwen/qwen3.6-35b-a3b`, `text-embedding-nomic-embed-text-v1.5`).
- Czytelny komunikat błędu (bez surowego tracebacku), jeśli serwer LM
  Studio nie jest uruchomiony.

**Ograniczenie:** to podejście **nie korzysta z douczonego LoRA** --
adaptery w `adapters/` są w formacie MLX i nie da się ich nałożyć na
model GGUF w LM Studio. To sam RAG + model z LM Studio, bez wyuczonego
stylu/kalibracji z kroku 4b-4c.

**Test:** te same pytania graniczne co w krokach 4c/6 ("ile wynosi
zasiłek dla bezrobotnych", "urlop po 10 latach pracy") przez
`chat_lmstudio.py` z Bielikiem-11B w LM Studio dały równie trafne,
dobrze ugruntowane i precyzyjnie cytowane odpowiedzi jak przez lokalny
MLX -- co potwierdza, że w tym projekcie ciężar poprawności faktycznej
niesie głównie RAG, a nie LoRA (LoRA odpowiada za styl, patrz krok 4b).

## Krok 8: automatyczna kontrola aktualności ustaw (`scripts/check_acts_freshness.py`)

Ręczna kontrola z kroku 5 (sprawdzanie w ELI API, czy akt nie został
uchylony) została zautomatyzowana jako osobny, powtarzalny skrypt. Dla
każdego aktu z `ACTS` (import z `download_acts.py`, więc lista nie jest
duplikowana) porównuje lokalnie zapisane metadane
(`data/raw/{skrot}_meta.json`) ze świeżym zapytaniem do ELI API i
zgłasza:
1. status/`inForce` inny niż obowiązujący,
2. wpis w `references."Akty uchylające"` (akt uchylony -- z ID
   następcy),
3. zmianę nazwy pliku tekstu ujednoliconego typu "U" (sygnał
   nowelizacji, której jeszcze nie mamy pobranej),
4. zmianę pola `changeDate` w metadanych aktu od ostatniego pobrania
   (miękki sygnał "coś się ruszyło, zweryfikuj ręcznie").

Nic nie modyfikuje -- czysto diagnostyczny, kod wyjścia 1 przy
znalezionych problemach (nadaje się do uruchamiania okresowo albo
przed każdym dodaniem nowego aktu). Test na 7 obecnych ustawach:
wszystkie aktualne.

Użycie:
```
python scripts/check_acts_freshness.py
```

## Krok 9: wariant CUDA (Linux/Windows + NVIDIA) — ⚠️ nieprzetestowane

Cel: projekt do tej pory działał tylko na Apple Silicon, bo MLX to
biblioteka specyficzna dla Apple/Metal. Żeby ktokolwiek z kartą NVIDIA
mógł to odpalić bez Maca, dodano równoległy stos:

**Poprawka przy okazji (drobny, ale realny bug):** `chat_lmstudio.py`
importował `SYSTEM_PROMPT`/`build_context` z `chat.py`, a `chat.py` na
poziomie modułu robi `from mlx_lm import ...` -- czyli `chat_lmstudio.py`
w praktyce NIE dałoby się uruchomić na maszynie bez zainstalowanego
`mlx_lm` (czyli poza macOS), mimo że sam w sobie nie potrzebuje MLX
(łączy się tylko przez HTTP z LM Studio). Wydzielono `scripts/prompt.py`
(bez ciężkich zależności: `SYSTEM_PROMPT`, `build_context`,
`build_user_message`) i przepisano `chat.py`/`chat_lmstudio.py`, żeby
z niego korzystały. Zweryfikowano przez `ast`-analizę importów, że
`chat_lmstudio.py` już nie ciągnie za sobą `mlx`/`mlx_lm` w ogóle.

**Nowe pliki:**
- `requirements-cuda.txt` -- `transformers`, `peft`, `bitsandbytes`,
  `trl`, `accelerate`, `datasets` (do doinstalowania obok
  `requirements.txt`; torch z CUDA instaluje się osobno wg instrukcji
  z pytorch.org, bo zależy od wersji sterowników).
- `scripts/train_lora_cuda.py` -- odpowiednik `mlx_lm.lora`:
  `BitsAndBytesConfig` (4-bit, nf4, double quant) +
  `prepare_model_for_kbit_training` + `peft.LoraConfig` (rank 8,
  target_modules `q_proj`/`v_proj` -- jak domyślne `mlx_lm.lora`,
  tylko attention) + `trl.SFTTrainer` z `assistant_only_loss=True`
  (odpowiednik `--mask-prompt`). Te same domyślne hiperparametry co w
  przebiegach z kroku 4b/6 (batch 2, 200 kroków, ewaluacja/zapis co
  25) -- z zastrzeżeniem w docstringu, że punkt najlepszego val loss
  może wypaść inaczej niż na MLX (inny optymalizator/precyzja), więc
  i tak trzeba ręcznie sprawdzić checkpointy, nie ufać automatycznie
  ostatniemu.
- `scripts/chat_cuda.py` -- odpowiednik `chat.py`: ładowanie modelu
  przez `transformers` (4-bit) + `peft.PeftModel.from_pretrained` na
  adapter, `tokenizer.apply_chat_template` + `model.generate`. Ten sam
  `SYSTEM_PROMPT`/RAG co pozostałe warianty.

**Jak API zostało zweryfikowane:** przez `find-docs`/context7
(`/huggingface/peft`, `/huggingface/trl`) zamiast z pamięci -- w
szczególności nazwa parametru `assistant_only_loss` (nowsza,
zastąpiła starsze podejścia do maskowania promptu w TRL) i wzorzec
ładowania modelu w 4-bit przez `BitsAndBytesConfig` +
`quantization_config=` w `AutoModelForCausalLM.from_pretrained`.

**Czego NIE zrobiono:** żadnego uruchomienia end-to-end. Środowisko
deweloperskie tego projektu to Mac bez karty NVIDIA/CUDA -- nie ma
tu jak przetestować `train_lora_cuda.py` ani `chat_cuda.py`.
Zweryfikowano tylko: składnię (`py_compile`), zgodność z aktualną
dokumentacją bibliotek, oraz że `target_modules` (`q_proj`, `v_proj`)
pasują do rzeczywistej architektury Bielika (`LlamaForCausalLM`,
sprawdzone w `config.json` pobranego modelu). Upublicznione mimo
braku testu na wyraźną prośbę użytkownika, z jasnym oznaczeniem
"NIEPRZETESTOWANE" w kodzie i README.

## Krok 10: kontekst rozmowy (multi-turn) we wszystkich wariantach czatu

Cel: do tej pory każde pytanie w trybie interaktywnym było niezależne
-- `answer()` budowało prompt tylko z systemowej wiadomości i
aktualnego pytania, więc model nie pamiętał poprzednich tur (np.
pytanie "a po 15 latach?" bez powtórzenia kontekstu nie miało do
czego się odnieść).

**Zmiana:** `answer()` w `chat.py`, `chat_lmstudio.py` i `chat_cuda.py`
przyjmuje teraz `history: list[dict]` -- listę wiadomości `user`/
`assistant` z poprzednich tur, mutowaną in-place (dopisywane pytanie
z wstrzykniętym kontekstem RAG przed wywołaniem modelu, odpowiedź po).
RAG wyszukuje fragmenty tylko na podstawie bieżącego pytania (nie
całej historii) -- każda tura i tak dokłada własne fragmenty do
promptu, więc historia rośnie szybciej niż liczba wiadomości.

Żeby nie przepełnić okna kontekstu modelu (istotne zwłaszcza przy
lokalnych 4.5B/11B), dodano:
- `trim_history()` w `scripts/prompt.py` -- ucina historię do
  ostatnich `--max-turns` par pytanie/odpowiedź (domyślnie 6),
  wywoływane po każdej turze w pętli interaktywnej.
- komendę `/nowy` w trybie interaktywnym -- czyści historię ręcznie,
  przydatne przy zmianie tematu (RAG i tak przeszukuje tylko bieżące
  pytanie, ale stara historia w promptcie mogłaby zdezorientować model
  przy całkowicie innym temacie).

Tryb `--prompt` (pojedyncze pytanie) używa pustej historii -- zachowanie
bez zmian.

Zweryfikowano: `py_compile` na wszystkich czterech plikach, unit-test
`trim_history` (okno przesuwne, sprawdzone na 10 turach z max_turns=3),
oraz test `chat_lmstudio.answer()` z podmienionym `requests.post` --
potwierdzono, że historia poprawnie akumuluje się między wywołaniami i
że kolejne zapytanie do serwera zawiera pełną dotychczasową historię.

**Test end-to-end na żywym serwerze LM Studio (Bielik-11B) znalazł dwa
realne problemy:**

1. **Przepełnienie kontekstu przez pojedynczy, bardzo długi artykuł.**
   Art. 50 ustawy systemowej ma w źródłowym PDF-ie ok. 51k znaków
   między nagłówkiem a Art. 51 (sprawdzone bezpośrednio w PDF, nie
   błąd naszego parsera -- to efekt wielu nowelizacji "informacji o
   stanie konta" w ZUS, ust. 1a-1f, 2a... aż do 32). Sam ten jeden
   wynik RAG w top-k=5 potrafił wygenerować >44k tokenów promptu i
   przekroczyć limit 32k tokenów serwera LM Studio -- już przy drugiej
   turze rozmowy. **Naprawiono:** `build_context()` w `scripts/prompt.py`
   ucina tekst artykułu powyżej `MAX_ARTICLE_CHARS = 6000` znaków i
   dopisuje link do `source_url` z pełną treścią. Zweryfikowano, że ten
   sam wcześniej awaryjny scenariusz (pytanie o urlop, potem "a po 15
   latach?") po poprawce przechodzi end-to-end bez błędu 400, a model
   poprawnie rozpoznaje kontynuację tematu z historii.

2. **Model bywa nadmiernie asekuracyjny po jednej odmowie w tej samej
   rozmowie -- naprawione, ale nie przez prompt.** Gdy któraś tura w
   rozmowie kończy się odpowiedzią "nie znalazłem w dostarczonych
   fragmentach", kolejne pytanie -- nawet niezwiązane z poprzednim i
   niewymagające RAG (np. prośba o zacytowanie czegoś z pierwszej tury
   rozmowy) -- ma podwyższoną szansę też skończyć się odmową, mimo że
   odpowiedź jest wprost obecna w historii. Odtworzone powtarzalnie w
   testach.

   Kolejność prób i co nie zadziałało:
   - Dopisanie do `SYSTEM_PROMPT` reguły zezwalającej na korzystanie z
     historii przy pytaniach o przebieg rozmowy -- **nie pomogło**
     (odpowiedź była nawet gorsza).
   - Mocniejsza, bardziej jednoznaczna wersja tej reguły ("zignoruj
     fragmenty poniżej, jeśli pytanie dotyczy tej rozmowy") -- **też
     nie pomogło**.
   - Większy, architektonicznie odmienny model (`qwen/qwen3.6-35b-a3b`,
     MoE, tryb "thinking") załadowany w tym samym LM Studio -- **ten
     sam wzorzec błędu**, mimo zadeklarowanego okna kontekstowego
     262144 tokenów i realnego zużycia rzędu kilku-kilkunastu tysięcy
     tokenów. Wyklucza to rozmiar modelu/okna kontekstowego jako
     przyczynę. (Przy okazji: modele "thinking" zużywają budżet
     `max_tokens` na wewnętrzne rozumowanie (`reasoning_content`) zanim
     wygenerują właściwą odpowiedź w `content` -- sztywne
     `max_tokens=500` w `chat_lmstudio.py` wystarcza dla Bielika, ale
     nie dla takich modeli; nieistotne dla obecnego zakresu, bo
     `chat_lmstudio.py` nie jest projektowany pod modele "thinking".)
   - Zmiana kolejności bloków w wiadomości (pytanie najpierw, fragmenty
     RAG jako dodatek na końcu, zamiast pytania owiniętego fragmentami)
     -- **wypadło gorzej**: model zignorował samo pytanie i kontynuował
     poprzedni temat z historii. Odrzucone też dlatego, że dotyczyłoby
     KAŻDej tury (też pojedynczych pytań), więc ryzykowałoby regresję
     tam, gdzie dziś działa dobrze.

   Rzeczywista przyczyna okazała się strukturalna, nie leksykalna: przy
   pytaniu wysłanym jako czyste zdanie, bez owijania blokiem "Fragmenty
   aktów prawnych: ... Pytanie: ..." (dokładnie tak, jak działa zwykły
   czat w GUI LM Studio -- bez RAG, bez wstrzykiwanego kontekstu), model
   **poprawnie** korzystał z historii, mimo tego samego, niezmienionego
   `SYSTEM_PROMPT` i mimo wcześniejszej odmowy w poprzedniej turze. Sama
   obecność bloku "oto fragmenty" zaburzała model niezależnie od tego,
   co dokładnie mówiły instrukcje o jego użyciu.

   **Naprawiono:** `prompt.looks_like_meta_question()` -- heurystyka
   (lista wzorców regex po polsku: "o czym mówiliśmy", "przypomnij",
   "podsumuj rozmowę", "pierwszym pytaniu" itd.) wykrywająca pytania o
   przebieg TEJ rozmowy. `answer()` w `chat.py`/`chat_lmstudio.py`/
   `chat_cuda.py` pomija dla nich wstrzykiwanie fragmentów RAG --
   pytanie idzie czysto, jak w GUI -- **tylko gdy historia jest już
   niepusta**; każde inne pytanie (w tym wszystkie pojedyncze przez
   `--prompt` i pierwsza tura rozmowy) przechodzi dokładnie dotychczasową
   ścieżką, bez zmian. To nie jest klasyfikator ML i nie złapie każdego
   sformułowania -- nierozpoznane pytanie meta po prostu wraca do
   opisanego wyżej ryzyka. Zweryfikowano: unit-test heurystyki (8
   przypadków, w tym pytania graniczne jak "A po 15 latach?", które MUSI
   zostać rozpoznane jako pytanie o fakt, nie meta), oraz pełny test
   end-to-end przez prawdziwy `chat_lmstudio.answer()` (Bielik-11B) --
   Q1/Q2 (pytania prawne) bez zmian, Q3 (pytanie meta po odmowie w Q2)
   teraz poprawnie cytuje przepis z pierwszej tury.

Nie przetestowano end-to-end na modelu MLX (`chat.py`) ani CUDA
(`chat_cuda.py`) -- tylko na LM Studio (koszt ładowania pełnego modelu
lokalnie w MLX to kilkadziesiąt sekund do minut na turę).

**Późniejszy test na żywo lokalnie przez `chat.py` (MLX, Bielik-11B +
`bielik11b-kadry-lora-iter25`)** potwierdził krok 10 na tym backendzie
(pytanie meta po odmowie -- poprawnie odtworzone), ale przy okazji
znalazł kolejny, osobny problem opisany niżej w kroku 11.

## Krok 11: RAG nie znajduje właściwego artykułu dla pytań eliptycznych ("a po 15 latach?")

Przy pełnym teście 4-pytaniowym przez `chat.py` (Q1: urlop po 10 latach
-> OK; Q2: minimalne wynagrodzenie w 2025 -> poprawna odmowa; Q3:
pytanie meta -> OK po kroku 10; **Q4: "A po 15 latach?"**) model
odpowiedział merytorycznie nieźle (26 dni się nie zmienia), ale dopisał
nieistniejący "art. 154 § 1 pkt 3" -- w Kodeksie pracy są tylko dwa
punkty (20 dni / 26 dni), nie ma trzeciego.

**Przyczyna:** `rag.search("A po 15 latach?")` samo w sobie w ogóle nie
trafia na art. 154 -- najlepszy wynik to score 0.729, kompletnie inny
temat (brak słowa "urlop" w pytaniu). Model dostał więc nietrafione
fragmenty, poprawnie zgadł z historii rozmowy że to kontynuacja pytania
o urlop (dobry sygnał dla samej ciągłości rozmowy), ale bez świeżego
groundingu zaczął "dopowiadać" nieistniejący szczegół.

**Naprawiono:** `prompt.search_with_history()` -- gdy najlepszy wynik
samego pytania jest słaby (< `ELLIPTICAL_SCORE_THRESHOLD = 0.75`) i
jest już historia, próbuje wyszukań z doklejonym KAŻDYM z ostatnich
`MAX_LOOKBACK_QUESTIONS = 3` pytań użytkownika osobno (nie zbiorczo
całą historią, nie tylko bezpośrednio poprzednim) i zostaje przy
wariancie z najlepszym wynikiem. Pierwsza wersja próbowała tylko
bezpośrednio poprzedniego pytania i **zawiodła w tym samym scenariuszu**
-- poprzednia tura (Q2) dotyczyła innego, zakończonego odmową tematu, a
faktycznie powiązane pytanie (Q1) było dwie tury wcześniej; sklejenie z
Q2 zamiast Q1 przesuwało wynik na fragmenty o minimalnym wynagrodzeniu.
Sprawdzanie każdego z ostatnich pytań osobno (zamiast jednego, zamiast
wszystkich naraz) naprawia to bez ryzyka rozmycia zapytania zbyt dużą
ilością niepowiązanego tekstu.

Świadomie warunkowe (próg score'u), nie bezwarunkowe: doklejanie
poprzedniego pytania do KAŻDEGO zapytania psuje trafność, gdy kolejne
pytanie to nowy, niezwiązany temat -- sprawdzone empirycznie (sklejenie
"urlop" + "minimalne wynagrodzenie" przesuwa poprawny wynik dla pytania
o wynagrodzenie na błędny artykuł o urlopie).

Zweryfikowano: unit-testy `search_with_history` (pytanie eliptyczne
trafia na art. 154 mimo niezwiązanej tury pośredniej; nowy temat nie
zostaje zepsuty przez historię o urlopie; pusta historia = zachowanie
jak zwykły `rag.search`), oraz pełny test end-to-end przez `chat.py`
(MLX, Bielik-11B) -- Q4 teraz poprawnie cytuje art. 154 § 1 pkt 2 i § 3,
bez zmyślonego "pkt 3".

## Krok 12: audyt jakości ekstrakcji artykułów z PDF-ów (`scripts/download_acts.py`)

Cel: użytkownik poprosił o audyt bazy RAG pod kątem "artykułów--potworów"
(bardzo długich wpisów, jak znaleziony wcześniej art. 50 ustawy
systemowej). Analiza rozkładu długości (`data/processed/all_articles.json`,
1650 artykułów przed tym krokiem) pokazała 73 artykuły >6000 znaków --
wszystkie sprawdzone (spot-check: PIT art. 21 -- katalog zwolnień
podatkowych, 121k znaków; rynek pracy art. 2 -- katalog definicji, 24k
znaków) okazały się prawdziwe, nie błąd parsowania; krok 10/11 (`MAX_ARTICLE_CHARS`)
już je wszystkie obcina. Przy tej samej okazji sprawdzono też bardzo
krótkie artykuły (34 sztuki <20 znaków -- to legalnie uchylone przepisy,
"(uchylony)", nieszkodliwe) oraz duplikaty numeru (artykuł, akt) --
20 par w Kodeksie pracy (31, 91, 111-113, 181-185, 221, 222, 231, 232,
251, 261, 291-294), co doprowadziło do trzech powiązanych, poważniejszych
znalezisk opisanych niżej.

**Znalezisko 1: numeracja z indeksem górnym gubiona przez `pypdf`.**
Kodeks pracy od lat jest nowelizowany przez wstawianie artykułów z
indeksem górnym (np. Art. 11¹) między istniejące numery. Sprawdzone
bezpośrednio w surowym PDF-ie: `pypdf.extract_text()` renderuje "Art. 11¹."
i prawdziwy "Art. 111." jako identyczny tekst -- oba nagłówki wyglądają
dosłownie tak samo. Zweryfikowano na poziomie pojedynczych znaków przez
`pdfplumber` (`page.chars`, pola `size`/`top`): indeks górny ma
wyraźnie mniejszą czcionkę (8.04 vs 12.0) i jest wyżej (top 729.28 vs
730.73) -- jednoznacznie mierzalne, nie zgadywanie.

**Naprawiono:** `find_article_numbers_pdfplumber()` -- skanuje znaki
bezpośrednio z PDF-a, wykrywa nagłówki "Art. N." i rozróżnia normalne
cyfry od indeksu górnego po rozmiarze czcionki (`SUPERSCRIPT_SIZE_RATIO
= 0.85`), zapisując indeks górny prawdziwym znakiem Unicode (np. "11¹").
W `process_act()` wynik jest zestawiany z artykułami wyodrębnionymi
przez istniejący, sprawdzony pypdf+regex (`parse_articles`) -- **tylko
gdy liczby się zgadzają**, numery są korygowane; w przeciwnym razie
korekta jest pomijana z ostrzeżeniem (bezpiecznik chroniący przed
uszkodzeniem danych, gdyby coś się nie zgadzało). Rezultat dla Kodeksu
pracy: 204 poprawione numery (znacznie więcej niż 20 pierwotnie
znalezionych duplikatów -- większość to artykuły bez kolizji z innym,
po prostu wcześniej błędnie zapisane bez indeksu górnego).

**Znalezisko 2: całe artykuły całkowicie pomijane (nie tylko źle
numerowane).** Przy pierwszym uruchomieniu liczby się nie zgadzały
(480 wg pdfplumber vs 475 wg pypdf dla Kodeksu pracy) -- dochodzenie
pokazało, że Art. 22³ (monitoring poczty elektronicznej pracownika,
realny, ważny przepis) w ogóle nie istniał jako osobny wpis: jego
nagłówek trafiał się w środku akapitu bez złamania linii, a superskrypt
renderował się ze spacją ("Art. 22 3." zamiast "Art. 223."), więc
`ARTICLE_SPLIT_RE` (wymaga początku linii, bez spacji w numerze) nigdy
go nie łapał -- cała treść artykułu została po cichu wchłonięta przez
poprzedni artykuł (Art. 22²).

**Naprawiono:** `recover_midtext_superscript_headers()` -- wykrywa
wzorzec "Art. N M." (spacja między głównym numerem a indeksem górnym,
1 lub więcej cyfr) w dowolnym miejscu tekstu, skleja numer i wymusza
początek nowej linii przed nim. Zastosowane przed `parse_articles()`.
Rezultat: liczba artykułów w Kodeksie pracy wzrosła z 475 do 480 (a po
naprawieniu drugiego wzorca niżej -- do zgodności 480/480).

**Znalezisko 3 (poważniejsze): tekst jeszcze nieobowiązujący włączany
do bazy, jakby już obowiązywał.** Ustawa o systemie ubezpieczeń
społecznych miała 207 vs 215 (niezgodność). Dochodzenie: Kancelaria
Sejmu oznacza w tekście ujednoliconym prawo uchwalone, ale jeszcze
nieobowiązujące (przyszła data wejścia w życie) nawiasami ostrymi
"< >", a stary tekst, który te przepisy docelowo zastąpią (wciąż ważny
do dnia wejścia w życie nowelizacji), nawiasami kwadratowymi "[ ]".
Potwierdzony przypadek: Art. 85c-85j (zmiany w orzecznictwie lekarskim
ZUS) były w całości oznaczone "< >", a mimo to trafiały do naszej bazy
RAG jako zwykłe, obowiązujące artykuły -- asystent mógłby zacytować
prawo, które jeszcze nie weszło w życie.

**Naprawiono:** `strip_not_yet_in_force_text()` -- usuwa "< ... >" w
całości (niezachłannie, `re.DOTALL`), a z "[ ... ]" zostawia samą
treść w środku (to nadal ważne prawo, nawiasy to tylko adnotacja
edytorska Kancelarii Sejmu). Zastosowane jako pierwszy krok czyszczenia
tekstu, przed naprawą numeracji. Rezultat: ustawa systemowa spadła z
207 do 191 artykułów (16 fragmentów jeszcze nieobowiązującego prawa
usuniętych). Dodano też licznik niesparowanych nawiasów z ostrzeżeniem
-- w tym akcie jest jedna para "<" bez pasującego ">" (prawdopodobnie
utracony znak przy ekstrakcji PDF-a w okolicach Art. 68); podstawienie
jest niezachłanne, więc niesparowany "<" po prostu nie zostaje
dopasowany (nic się nie usuwa) zamiast ryzykować pochłonięcie za
dużego fragmentu poprawnego tekstu.

**Znalezisko 4 (przy okazji, niezwiązane z superskryptami): litery
polskie w sufiksie numeru artykułu gubione przez główny regex.**
`ARTICLE_SPLIT_RE` używał `[a-z]{0,3}` (tylko ASCII) dla sufiksu typu
"22a"/"18c" -- artykuł "Art. 22ł." (ustawa o PIT, polska litera "ł")
nigdy nie pasował do wzorca i był całkowicie pomijany, tak samo jak
znalezisko 2 (treść wchłaniana przez poprzedni artykuł). **Naprawiono:**
`[a-z]{0,3}` -> `[a-ząćęłńóśźż]{0,3}` w `ARTICLE_SPLIT_RE` i
`MIDTEXT_SUPERSCRIPT_HEADER_RE`.

**Stan końcowy (zweryfikowany przez pełne, realne uruchomienie
`download_acts.py` + `build_rag_index.py`, nie tylko offline na
zcache'owanych PDF-ach):**
- Kodeks pracy: 480/480 zgodność, 204 poprawione numery.
- Ustawa o systemie ubezpieczeń społecznych: 191/191 zgodność (207 -> 191
  po usunięciu tekstu nieobowiązującego), 0 poprawek numeracji potrzebnych.
- ustawa_o_minimalnym_wynagrodzeniu, ustawa_zasilkowa, ustawa_o_ppk:
  zgodność bez zmian (te akty nie mają numeracji z indeksem górnym).
- ustawa_o_pit: 267/268 -- **pozostała rozbieżność** (1 artykuł, "Art. 52zb",
  prawdopodobnie ten sam typ problemu co znalezisko 2, ale rzadszy
  wariant nie objęty obecną naprawą) -- korekta numeracji bezpiecznie
  pominięta dla tego aktu (nic nie ucierpiało, po prostu nie skorzystał
  z poprawek).
- ustawa_o_rynku_pracy: 461/464 -- **pozostała rozbieżność**, ale z innej
  przyczyny: `find_article_numbers_pdfplumber` fałszywie wykrywa
  cytowane w treści aktu nagłówki innych, nowelizowanych ustaw (np.
  „Art. 7a." w cudzysłowie, wewnątrz opisu zmiany innej ustawy) --
  fałszywy alarm w NOWYM narzędziu audytowym, nie błąd w danych;
  oryginalny regex (`^Art\.`) poprawnie je pomija dzięki wymogowi
  początku linii. Korekta numeracji bezpiecznie pominięta.
- `data/processed/all_articles.json`: 1650 -> 1640 artykułów (+5 Kodeks
  pracy odzyskane, +1 PIT odzyskany, -16 ustawa systemowa usunięte jako
  nieobowiązujące).
- `data/processed/rag_index.npy`: 4425 -> 4362 fragmentów po przebudowie.

**Zweryfikowano żywymi zapytaniami RAG po przebudowie indeksu:**
"monitoring poczty elektronicznej pracownika" -> poprawnie zwraca art.
22³ (score 0.847) jako najlepszy wynik; "Pracodawca jest obowiązany
szanować godność pracownika" -> poprawnie zwraca art. 11¹ (score
0.913); "badanie przez lekarza orzecznika ZUS" -> wyniki NIE zawierają
już żadnego z usuniętych art. 85c-85j.

**Nowa zależność:** `pdfplumber==0.11.10` dodana do
`requirements-common.txt` -- używana tylko przy budowaniu bazy
(`download_acts.py`), nie w runtime chatu.

**Co świadomie zostawiono:** dwa rzadkie przypadki brzegowe opisane
wyżej (PIT: 1 artykuł, rynek pracy: fałszywy alarm w audycie) --
bezpiecznik (porównanie liczby nagłówków) gwarantuje, że nic nie zostało
uszkodzone, tylko te dwa akty nie skorzystały z ewentualnych poprawek
numeracji (a rynek pracy, jako akt z 2025 r., prawdopodobnie i tak nie
ma żadnej historycznej numeracji z indeksem górnym do poprawienia).

Zmiany z tego kroku scommitowane i wypchnięte na `origin/main` jako
`ed7d6bb` ("Napraw ekstrakcję artykułów z PDF: indeks górny, pominięte
artykuły, tekst nieobowiązujący").

## Krok 13: domknięcie dwóch ostatnich przypadków brzegowych z kroku 12

Cel: krok 12 zostawił dwie znane rozbieżności (PIT 267/268, rynek pracy
461/464) jako świadomie pominięte -- bezpiecznik chronił przed
uszkodzeniem danych, ale użytkownik poprosił o ich domknięcie.

**PIT (brakujący Art. 52zb):** przyczyna inna niż w kroku 12 -- nagłówek
był poprawnie złamany do nowej linii, ale `pypdf` wstawił spację TUŻ
PRZED kropką kończącą numer ("Art. 52zb ." zamiast "Art. 52zb."), więc
`ARTICLE_SPLIT_RE` (wymaga kropki bezpośrednio po numerze) nie łapał
tego nagłówka -- treść po cichu wchłonięta przez poprzedni artykuł,
ten sam mechanizm co w kroku 12, inny wariant usterki. **Naprawiono:**
`fix_stray_space_before_period()` -- usuwa spację między numerem a
kropką. Rezultat: PIT 267 -> 268 artykułów, zgodność 268/268.

**Rynek pracy (fałszywy alarm, nie błąd danych):** `find_article_numbers_pdfplumber`
wykrywał też "Art. 7a." wewnątrz cudzysłowu („Art. 7a. 1. Podmiotowi...")
-- to cytat treści nowelizowanego przepisu z INNEJ ustawy (fragment
opisujący, jak inna ustawa ma zostać zmieniona: "po art. 7 dodaje się
art. 7a w brzmieniu: „Art. 7a. ..."), a nie prawdziwy artykuł tego aktu.
Sprawdzone na poziomie znaków (`pdfplumber`): znak bezpośrednio przed
"A" to „ (ten sam `top`, czyli ta sama linia -- to NIE była kwestia
złamania linii, tylko brakującego rozpoznania cudzysłowu). Oryginalny
`ARTICLE_SPLIT_RE` poprawnie to pomijał już wcześniej (stąd "błąd" był
tylko w nowym narzędziu audytowym z kroku 12, nie w faktycznych danych).
**Naprawiono:** dodano `QUOTE_CHARS` (`„ " " " »`) -- pomijamy dopasowanie,
jeśli poprzedni znak to cudzysłów. Rezultat: rynek pracy 461/464 -> 461/461.

**Stan końcowy po pełnym, realnym uruchomieniu pipeline'u:** wszystkie
siedem aktów ma zgodność 1:1 między liczbą nagłówków wykrytych przez
niezależną analizę geometrii PDF-a (`pdfplumber`) a liczbą artykułów
wyodrębnionych przez `pypdf`+regex -- żaden akt nie korzysta już z
bezpiecznika/pomijania korekty. `data/processed/all_articles.json`:
1640 -> 1641 (odzyskany Art. 52zb). `rag_index.npy`: bez zmian liczby
fragmentów (4362) -- Art. 52zb mieści się w istniejącym podziale.
Zweryfikowano: unit-test offline na wszystkich 7 zcache'owanych PDF-ach
(pełna zgodność), pełne uruchomienie `download_acts.py` +
`build_rag_index.py` na żywych danych z ELI API, oraz obecność Art. 52zb
w przebudowanym `rag_index_meta.json`.

## Krok 14: pierwsze testy automatyczne (`tests/`, pytest)

Cel: cała weryfikacja logiki napisanej w krokach 10-13 (kontekst
rozmowy, wykrywanie pytań meta, fallback wyszukiwania RAG, ekstrakcja
artykułów z PDF) była do tej pory wyłącznie ręczna -- żywe testy przez
skrypty w trakcie sesji, bez niczego zapisanego, co pilnowałoby przed
regresją przy kolejnych zmianach.

**Dodano:**
- `requirements-dev.txt` -- `pytest==8.4.2`, osobno od
  `requirements-common.txt` (niepotrzebne do samego korzystania z
  asystenta).
- `conftest.py` (root) -- dodaje `scripts/` do `sys.path`, bo to
  celowo nie jest pakiet (zbiór niezależnych CLI, importujących się
  nawzajem po nazwie modułu).
- `tests/test_prompt.py` -- `trim_history` (okno przesuwne),
  `looks_like_meta_question` (8 przypadków, w tym pytania graniczne
  jak "A po 15 latach?", które MUSI zostać rozpoznane jako pytanie o
  fakt, nie meta), `build_context` (obcinanie długich artykułów),
  `search_with_history` (w tym reprodukcja realnego przypadku: fallback
  musi pominąć niezwiązaną, bezpośrednio poprzednią turę i sięgnąć do
  wcześniejszej, faktycznie powiązanej). Używa lekkiego `FakeRag`
  (zwraca predefiniowane wyniki) zamiast prawdziwego `RagIndex` --
  żadnych ciężkich zależności (embedding model) przy uruchamianiu testów.
- `tests/test_download_acts.py` -- testy jednostkowe na małych,
  ręcznie skonstruowanych fragmentach tekstu dla
  `strip_not_yet_in_force_text`, `recover_midtext_superscript_headers`,
  `fix_stray_space_before_period`, `ARTICLE_SPLIT_RE`/`parse_articles`
  (w tym regresja polskiej litery w sufiksie, znalezisko 4 z kroku 12).
  Osobno, oznaczone `@pytest.mark.skipif` (pomijane, jeśli nie ma
  lokalnie pobranej bazy -- `data/raw/`, `data/processed/` są
  gitignored): regresje DOKŁADNIE tych błędów znalezionych w audycie --
  `find_article_numbers_pdfplumber` zgadza się z `parse_articles` dla
  Kodeksu pracy, brak zdublowanych numerów artykułów, obecność Art. 22³
  (nie wtopiony w Art. 22²), brak Art. 85c-85j (jeszcze nieobowiązujące)
  w ustawie systemowej, obecność Art. 52zb w PIT.

**Zweryfikowano:** `pytest -v` -- 28/28 testów przechodzi (uruchomione
lokalnie, z pobraną bazą, więc łącznie z testami oznaczonymi `skipif`).

Dopisano sekcję "Testy" do README.md z instrukcją uruchomienia.

## Krok 15: upload pliku PDF jako dodatkowy kontekst (`scripts/file_index.py`)

Cel: użytkownik zaproponował na początku sesji, żeby oprócz kontekstu
rozmowy dało się też wgrać własny plik (np. umowę o pracę) jako
dodatkowy kontekst obok RAG-a nad ustawami -- na start ograniczone do
PDF-ów tekstowych (nie skanów/obrazów, OCR poza zakresem).

**Nowy moduł `scripts/file_index.py`:** `SessionFileIndex` -- lekki,
tymczasowy indeks żyjący tylko w pamięci na czas sesji (NIE zapisywany
na dysk, w przeciwieństwie do `rag_search.RagIndex`), żeby nie mieszać
dokumentów użytkownika z kuratorowaną bazą ustaw. Dzieli model
embeddingowy z `RagIndex` (`SessionFileIndex(model=rag.model)`) --
model ładowany tylko raz, nie dubluje ~kilkuset MB w pamięci.
`add_pdf()` wyciąga tekst przez `pypdf`, dzieli na fragmenty (reużyte
`split_long_text`/`MAX_CHUNK_CHARS`/`OVERLAP_CHARS` z
`build_rag_index.py` -- ta sama logika co dla ustaw, żadnej duplikacji),
liczy embeddingi. Pusty tekst (skan bez OCR) rzuca czytelny `ValueError`
zamiast po cichu zaindeksować nic. Obsługuje jeden plik na raz (kolejny
`add_pdf()` zastępuje poprzedni) -- prosty zakres na start.

**Integracja z promptem:** `prompt.build_user_message()` przyjmuje
teraz opcjonalny `file_results`, dokłada osobną sekcję "Fragmenty z
wgranego przez użytkownika pliku" wyraźnie oddzieloną od "Fragmenty
aktów prawnych". `SYSTEM_PROMPT` zaktualizowany o regułę: fragmenty z
pliku użytkownika NIE są obowiązującym prawem i model ma je tak
oznaczać w odpowiedzi (żeby np. zapis w prywatnej umowie nie został
zacytowany tak, jakby to był przepis ustawy).

**Komendy w każdym z trzech wariantów czatu** (`chat.py`,
`chat_lmstudio.py`, `chat_cuda.py`, ten sam wzorzec co przy kroku 10):
`--file <ścieżka>` (wgranie przy starcie) i `/plik <ścieżka>` (w trakcie
rozmowy). Dla pytań rozpoznanych jako meta (`looks_like_meta_question`,
krok 10) plik -- tak jak RAG nad ustawami -- jest pomijany, spójnie z
istniejącą logiką.

Zweryfikowano: 12 nowych testów jednostkowych (`tests/test_file_index.py`,
`PdfReader` mockowany, deterministyczny `FakeModel` -- bez ciężkich
zależności) oraz pełny test end-to-end: ręcznie skonstruowany, prawdziwy
plik PDF (zweryfikowany, że `pypdf` faktycznie wyciąga z niego tekst)
wgrany przez `SessionFileIndex` z prawdziwym modelem embeddingowym,
wyszukiwanie w nim działa równolegle z głównym RAG-iem nad ustawami
(różne, sensowne wyniki dla obu pul), i pełny przebieg przez żywy
LM Studio (Bielik-11B): pytanie o okres wypowiedzenia w przykładowej
umowie (4 miesiące) poprawnie zacytowane jako treść pliku użytkownika,
z trafną uwagą, że to może odbiegać od Kodeksu pracy (art. 36) --
model nie pomylił dwóch źródeł.

## Krok 16: rozszerzenie zbioru LoRA o brakujące ustawy i naprawa dwóch nieaktualnych odmów

Cel: `data/finetune/train.jsonl` (35 przykładów) w ogóle nie miał
przykładów dla 4 z 7 ustaw obecnych w `ACTS`/RAG (ustawa zasiłkowa,
PIT, PPK, ustawa o rynku pracy) -- cały zbiór kręcił się wokół Kodeksu
pracy, ustawy o systemie ubezpieczeń społecznych i ustawy o minimalnym
wynagrodzeniu. Przy okazji audytu znaleziono też realny błąd: dwa
przykłady (dawne nr 31 i 32) uczyły model odmawiać odpowiedzi na
pytania o zasiłek macierzyński i zaliczkę na PIT, tłumacząc to brakiem
tych ustaw w bazie wiedzy -- a `ustawa_zasilkowa` i `ustawa_o_pit` są w
`ACTS` (i w RAG) od tego samego commita co dane treningowe. LoRA
uczyła więc model fałszywej, niepotrzebnej odmowy.

**Naprawiono:** oba przykłady zastąpiono realnymi odpowiedziami,
opartymi na faktycznej treści pobranych ustaw (art. 29 i 30 ustawy
zasiłkowej; art. 32 ustawy o PIT) -- zweryfikowanymi bezpośrednio w
`data/processed/*.json`, a nie z pamięci.

**Dodano 12 nowych przykładów do `train.jsonl` i 3 do `valid.jsonl`**
(35/5 -> 47/8), pokrywających dotąd nieobecne ustawy -- każdy cytat
zweryfikowany względem realnego tekstu artykułu w `data/processed/`:
- ustawa zasiłkowa: okres wyczekiwania na zasiłek chorobowy (art. 4),
  limit dni zasiłku opiekuńczego (art. 33), świadczenie rehabilitacyjne
  (art. 18), zasiłek macierzyński po ustaniu zatrudnienia wskutek
  upadłości pracodawcy (art. 30);
- PIT: ulga dla młodych do 26. roku życia (art. 21 ust. 1 pkt 148),
  standardowe i podwyższone koszty uzyskania przychodu (art. 22),
  nowa, realna odmowa dla pytania faktycznie spoza zakresu (rejestracja
  spółki z o.o. w KRS);
- PPK: wysokość wpłat podstawowych pracodawcy/pracownika (art. 26-27),
  obniżona wpłata przy niskich zarobkach (art. 27 ust. 2),
  dobrowolność uczestnictwa i tryb rezygnacji (art. 23), autozapis co
  4 lata (art. 23 ust. 5-6, w `valid.jsonl`);
- ustawa o rynku pracy i służbach zatrudnienia: warunek 365 dni w 18
  miesiącach na prawo do zasiłku dla bezrobotnych (art. 218), utrata
  prawa do zasiłku przy wypowiedzeniu z własnej inicjatywy (art. 219) i
  przy porozumieniu stron (art. 220, w `valid.jsonl`);
- dodatkowo w `valid.jsonl`: graniczny przypadek dla art. 36 § 1 pkt 2
  Kodeksu pracy (zatrudnienie dokładnie 6 miesięcy).

Świadomie NIE dodano przykładów wieloturowych -- format treningowy
(`{"messages": [...]}`) by je obsłużył, ale obsługa historii rozmowy
(`search_with_history`, `trim_history`, krok 10-11) to mechanizm w
kodzie RAG, a nie coś, czego LoRA musi się uczyć z przykładów; bazowy
model instrukcyjny (Bielik-*-Instruct) już ma tę zdolność z własnego
treningu.

**Zweryfikowano:** wszystkie 55 linii to poprawny JSON, `pytest -v` --
36/36 (bez regresji w istniejących testach, niezwiązanych z tą zmianą).

**Ponowny trening LoRA na 4.5B (`mlx_lm.lora`, te same hiperparametry co
krok 4b: batch 2, 200 iteracji, ale ewaluacja/zapis co 25 zamiast co 50,
po doświadczeniu z krokiem 6) -- ten sam wzorzec przeuczenia co przy
mniejszym zbiorze:** val loss minimum (1.233) przy iter 25/50, potem
monotonicznie rósł do 1.927 przy iter 200, mimo że train loss spadł do
~0. Jako punkt odniesienia przyjęto checkpoint z iter 50
(`adapters/bielik-kadry-lora-v2-iter50`) -- pełna historia checkpointów
w `adapters/bielik-kadry-lora-v2/`.

**Test end-to-end (RAG + nowy adapter, `chat.py --model
models/Bielik-4.5B-v3.0-Instruct-mlx`) na pytaniach spoza dosłownej
treści treningu ujawnił realną, niepokojącą lukę -- w przeciwieństwie
do kroku 4c/6, tym razem model MYLI SIĘ mimo poprawnie wyszukanego przez
RAG kontekstu:**
- "Ile procent wynagrodzenia wpłaca pracodawca do PPK jako wpłatę
  podstawową?" -- poprawna odpowiedź to 1,5% (art. 26 ust. 1, i RAG
  faktycznie zwrócił ten artykuł z tą liczbą jako najlepszy wynik,
  zweryfikowano bezpośrednio przez `rag.search()`). Model z nowym
  adapterem odpowiedział **"2,5%"** (to wartość z ust. 2 -- wpłata
  DODATKOWA pracodawcy -- błędnie podpisana jako ust. 1). Ten sam
  stary adapter (`bielik-kadry-lora-iter50`, trenowany bez żadnego
  przykładu o PPK) też odpowiedział błędnie "2,5%", ale przynajmniej
  dopisał poprawny cytat "1,5%" zaraz potem (sprzeczne ze sobą, ale
  poprawna liczba była gdzieś w odpowiedzi). **Model bazowy bez LoRA
  (`--no-adapter`) odpowiedział w 100% poprawnie**, cytując dokładnie
  art. 26 ust. 1 z właściwą liczbą.
- "Przez ile dni w ciągu ostatnich 18 miesięcy trzeba było pracować,
  żeby dostać zasiłek dla bezrobotnych?" (poprawna odpowiedź: 365 dni,
  art. 218 ust. 1) -- nowy adapter poprawnie zacytował warunek o
  minimalnym wynagrodzeniu i art. 218 ust. 1 pkt 1, ale **nigdy nie
  podał samej liczby 365**, myląc "18 miesięcy" (okno czasowe) z
  odpowiedzią na pytanie "ile dni".
- Dla kontrastu, znane z wcześniejszych kroków pytanie graniczne z
  Kodeksu pracy ("zatrudniony od 7 lat -> jaki okres wypowiedzenia?")
  nowy adapter rozwiązał bezbłędnie, z poprawnym cytowaniem.

**Wniosek (nowy, ważniejszy niż w kroku 4b):** LoRA w tej architekturze
(mały model + krótki fine-tuning) nie tylko "nie wystarcza, by wszyć
precyzyjne progi liczbowe" (krok 4b) -- w artykułach z KILKOMA blisko
siebie leżącymi, łatwymi do pomylenia liczbami (PPK: 0,5/1,5/2/2,5%;
rynek pracy: 365 dni vs 18 miesięcy) potrafi **przesłonić poprawny
kontekst z RAG błędną, ale pewnie brzmiącą liczbą z pamięci wag** --
gorzej niż model bazowy bez żadnego douczenia na tym samym pytaniu.
Prawdopodobna przyczyna: styl wyuczony na przykładach z Kodeksu pracy
(jedna liczba na pytanie, podana pewnie na początku odpowiedzi) nie
uogólnia się dobrze na akty z tabelami kilku blisko siebie leżących
wartości procentowych, których w zbiorze treningowym jest na razie
niewiele (po 3-4 przykłady na akt). **Adapter `bielik-kadry-lora-v2-iter50`
NIE został ustawiony jako domyślny w `chat.py`** -- świadomie
pozostawiono `bielik11b-kadry-lora-iter25` (11B, niezmieniony, patrz
krok 6) jako domyślny, dopóki ta luka nie zostanie zbadana głębiej.

**Domknięcie na 11B, na wyraźną prośbę użytkownika ("czy to ma
znaczenie, na jakim modelu testowałeś?"):** przetrenowano LoRA na tym
samym rozszerzonym zbiorze na `Bielik-11B-v3.0-Instruct` (te same
hiperparametry co krok 6/16: batch 2, 200 iteracji, ewaluacja/zapis co
25). Ten sam wzorzec przeuczenia co zawsze, tylko szybszy i wyraźniejszy
niż na 4.5B: val loss startuje niżej (1.109) i osiąga minimum (0.584)
już przy iter 25, po czym rośnie monotonicznie do 0.793 przy iter 200.
Jako punkt odniesienia przyjęto `adapters/bielik11b-kadry-lora-v2-iter25`
(analogicznie do `bielik11b-kadry-lora-iter25` z kroku 6).

**Te same dwa pytania, które zawodziły na 4.5B, na 11B wypadły
bezbłędnie:**
- PPK, wpłata podstawowa pracodawcy -> poprawnie "1,5%", art. 26 ust. 1,
  bez wzmianki o 2,5% (ani u nowego, ani u starego 11B adaptera --
  **oba** 11B adaptery, w przeciwieństwie do 4.5B, poradziły sobie z tym
  pytaniem poprawnie).
- Zasiłek dla bezrobotnych, wymagany okres pracy -> poprawnie "365 dni",
  art. 218 ust. 1, z trafnym dodatkowym wyjaśnieniem warunków.
- Kontrolne pytanie z Kodeksu pracy (zatrudniony 7 lat) -> bez zmian,
  poprawne.

**Wniosek: to była słabość MAŁEGO modelu (4.5B), nie danych ani samej
metody LoRA.** 11B -- nawet stary adapter, trenowany BEZ żadnego
przykładu o PPK -- poprawnie korzystał z dostarczonego kontekstu RAG
zamiast go przesłaniać halucynacją z wag. Większy model ma najwyraźniej
wystarczająco dużo "miejsca", żeby nie tracić zdolności do wiernego
cytowania kontekstu przy okazji uczenia się stylu odpowiedzi -- 4.5B tej
rezerwy nie ma. Ponieważ `chat.py` i tak domyślnie używa 11B (decyzja z
kroku 6), **ustawiono `bielik11b-kadry-lora-v2-iter25` jako nowy
domyślny adapter** (`DEFAULT_ADAPTER_PATH` w `scripts/chat.py`) -- to
czysta poprawa względem poprzedniego domyślnego adaptera (dokłada 4
nowe ustawy, nie traci nic z dotychczasowej jakości na Kodeksie pracy).
Wariant 4.5B (`bielik-kadry-lora-v2-iter50`) pozostaje dostępny przez
`--model`/`--adapter-path` dla szybszej pracy, ale z jasnym zastrzeżeniem
(patrz wyżej) o mniejszej niezawodności na nowo dodanych tematach.

## Krok 17: CI (GitHub Actions)

Cel: testy z kroku 14 i kontrola aktualności z kroku 8 do tej pory
wymagały ręcznego uruchomienia -- łatwo zapomnieć.

**`.github/workflows/tests.yml`** -- `pytest` przy każdym push/PR do
`main` (Python 3.12, `pip install -r requirements-common.txt -r
requirements-dev.txt`). Na świeżym klonie repo (bez lokalnie pobranej
bazy) testy oznaczone `skipif` (krok 14) poprawnie się pomijają --
zachowanie nie różni się od uruchomienia lokalnie bez `data/raw`/
`data/processed`.

**`.github/workflows/acts-freshness.yml`** -- cotygodniowy (poniedziałek
6:00 UTC) i ręczny (`workflow_dispatch`) przebieg
`check_acts_freshness.py`. Wymagało zmiany w `.gitignore`: `data/raw/*`
zamiast `data/raw/` z wyjątkiem `!data/raw/*_meta.json` -- same metadane
z ELI API (kilkadziesiąt KB na akt, bez treści PDF) są teraz
commitowane jako "ostatni znany stan" ustaw, względem którego CI
porównuje świeży stan API (bez tego CI nie miałoby punktu odniesienia,
bo `data/raw/*.pdf` i `data/processed/` zostają gitignored jak wcześniej).

Zweryfikowano lokalnie: obie definicje YAML (`yaml.safe_load`),
`check_acts_freshness.py` uruchomiony ręcznie (nie wymaga PDF-ów, tylko
`*_meta.json` + zapytanie do API -- potwierdzone czytaniem
`pick_text_file()`), `pytest` bez regresji (36/36). **Przy tej okazji
skrypt faktycznie znalazł 3 akty z rozbieżnym `changeDate`** względem
ostatniego pobrania (Kodeks pracy, ustawa systemowa, ustawa o PIT) --
miękki sygnał "coś się ruszyło", do ręcznej weryfikacji, nie
potwierdzona jeszcze nowelizacja.

## Krok 18: spike walidacyjny dla wersjonowania czasowego ("as-of")

Cel: przed pisaniem docelowego kodu dla funkcji "prawo obowiązujące na dany
dzień w przeszłości" (patrz "Co dalej" pkt 4 niżej -- pełny plan
zaakceptowany, zapisany w `/Users/szymon/.claude/plans/drifting-toasting-wozniak.md`)
zweryfikować na żywym ELI API dla WSZYSTKICH 7 ustaw (nie tylko ustawy o
minimalnym wynagrodzeniu, jedynej sprawdzonej ręcznie wcześniej), czy
podejście oparte o `references["Inf. o tekście jednolitym"]` (lista
obwieszczeń ogłaszających kolejne teksty ujednolicone, każde z własnymi
`legalStatusDate`/`expirationDate`) faktycznie daje spójny, ciągły łańcuch
okien czasowych -- rzucany skrypt, zero zmian w kodzie produkcyjnym.

**Wynik (66 zapytań HTTP, tylko metadane):**

| Ustawa | W mocy od | Obwieszczeń | Najstarsza data z `legalStatusDate` | `current_valid_from` (heurystyka) |
|---|---|---|---|---|
| kodeks_pracy | 1975-01-01 | 10 | 2014-09-16 (najstarsze ma `legalStatusDate=None`) | 2025-02-07 |
| ustawa_o_systemie_ubezpieczen_spolecznych | 1999-01-01 | 14 | 2013-10-16 (2 najstarsze mają `None`) | 2026-02-06 |
| ustawa_o_minimalnym_wynagrodzeniu | 2003-01-01 | 5 | 2015-11-09 (wszystkie mają datę) | 2024-11-27 |
| ustawa_zasilkowa | 1999-09-01 | 12 | 2013-12-16 (2 najstarsze mają `None`) | 2026-06-17 |
| ustawa_o_pit | 1992-01-01 | 14 | 2012-01-05 (3 najstarsze mają `None`) | 2026-04-01 |
| ustawa_o_ppk | 2019-01-01 | 4 | 2022-10-26 (wszystkie mają datę) | 2026-02-10 |
| ustawa_o_rynku_pracy | 2025-06-01 | 0 | -- (brak obwieszczeń, ustawa za młoda) | brak |

**Znalezisko 1 (ważne, wymaga poprawki w docelowym kodzie): kolejność
obwieszczeń bez `legalStatusDate` w odpowiedzi API NIE jest chronologiczna.**
Cztery ustawy (kodeks_pracy, ustawa systemowa, ustawa zasiłkowa, PIT) mają
2-3 najstarsze obwieszczenia z `legalStatusDate=None` (pole najwyraźniej nie
było uzupełniane w starszych wpisach metadanych ELI, mimo że `expirationDate`
jest podane) -- np. dla ustawy systemowej `DU/2009/1585` (expirationDate
2013-12-04) występuje w surowej liście `references` PRZED `DU/2007/74`
(expirationDate 2009-11-10), czyli w odwrotnej kolejności niż faktyczna. Prosty
`sort(key=lambda a: a["legalStatusDate"] or "")` (użyty w spike'u) tego nie
wyłapuje -- stabilne sortowanie zostawia oryginalną (błędną) kolejność API dla
wpisów z tym samym kluczem `""`. **Do zastosowania w `compute_version_windows`
(faza 1 planu):** dla wpisów bez `legalStatusDate` sortować dodatkowo po
`expirationDate` (który jest uzupełniony nawet dla najstarszych wpisów) jako
kluczu drugorzędnym, zamiast ufać kolejności z API.

**Znalezisko 2 (do udokumentowania jako znane ograniczenie, nie błąd do
naprawienia): sąsiednie okna NIE stykają się idealnie.** `expirationDate`
jednego obwieszczenia i `legalStatusDate` następnego różnią się typowo o
kilka dni do kilku tygodni (czasem `legalStatusDate` następnego wypada
WCZEŚNIEJ niż `expirationDate` poprzedniego, czyli okna się nawet zachodzą) --
sprawdzone na wszystkich 7 ustawach, to systemowa cecha danych ELI (czas
między "stan prawny na dzień" a formalnym wygaśnięciem poprzedniego obwieszczenia
w rejestrze), nie błąd naszego kodu. **Wniosek dla `compute_version_windows`:**
nie próbować wymuszać jednej, idealnie ciągłej osi czasu -- każda wersja ma
swoje własne, niezależne `valid_from`/`valid_to` z własnych pól API; w wąskim
oknie kilku tygodni wokół granicy dwóch wersji dokładność co do dnia nie jest
gwarantowana. Do zapisania jako ograniczenie w README/kodzie.

**Znalezisko 3 (pozytywne, koryguje wcześniejsze szacunki z "Co dalej" pkt 4):
zasięg historyczny sięga dalej wstecz niż sądzono.** Dla ustaw, których
najstarsze obwieszczenie ma `legalStatusDate=None` (kodeks_pracy, ustawa
systemowa, ustawa zasiłkowa, PIT), ten wpis i tak ma sens jako wersja "od
zawsze do jego `expirationDate`" (pole `valid_from=None` w naszym modelu
danych oznacza właśnie brak znanej dolnej granicy, nie brak pokrycia) -- czyli
realny brak pokrycia (data sprzed najstarszego obwieszczenia w łańcuchu)
dotyczy w praktyce głównie ustawy o minimalnym wynagrodzeniu (~12 lat, 2003-2015)
i ustawy o PPK (~1,5 roku, 2019-2020), a nie wszystkich 7 ustaw jak sugerował
wcześniejszy, ostrożniejszy zapis w "Co dalej" pkt 4.

**Koszt pełnego przebiegu `--history`:** 59 obwieszczeń łącznie (10+14+5+12+14+4+0)
-- 59 dodatkowych zapytań o metadane + 59 pobrań PDF + parsowanie każdego tym
samym pipeline'em co dziś (włącznie z geometrią `pdfplumber`). Rząd wielkości
akceptowalny dla rzadko uruchamianego, opt-in kroku z zachowaniem grzecznościowych
opóźnień między zapytaniami (ten sam wzorzec co już w `download_acts.py`).

Rzucany skrypt spike'a (nie część repo, w scratchpadzie sesji) importował
bezpośrednio `ACTS`/`fetch_meta` z `scripts/download_acts.py` -- zero zmian w
kodzie produkcyjnym na tym etapie, zgodnie z fazą 0 planu.

## Krok 19: warstwa danych i indeksu dla wersjonowania czasowego ("as-of")

Cel: Faza 1 (dane) i Faza 2 (indeks) planu z Kroku 18 --
`/Users/szymon/.claude/plans/drifting-toasting-wozniak.md`. Zaimplementowane
przez dwa kolejne uruchomienia w tle (forki), z ręczną weryfikacją i jedną
dodatkową poprawką na wierzchu.

**Faza 1 (`scripts/download_acts.py`, nowy `scripts/download_acts_history.py`):**
wydzielono `_extract_articles()` z `process_act()` (współdzielony pipeline
fetch->czyszczenie->parsowanie dla aktu bazowego LUB obwieszczenia -- ten sam
kształt `{publisher, year, position}`), dodano `compute_version_windows()`
(czysta), `fetch_version_windows()`, `process_act_version()`. Zweryfikowano
bajtową identyczność `all_articles.json` przed/po refaktorze (`cmp`) --
identyczne. Dwa odchylenia od pierwotnego planu, oba potwierdzone empirycznie
na żywym API i udokumentowane w kodzie:
- **Klucz sortowania obwieszczeń bez `legalStatusDate`:** plan proponował
  fallback na `expirationDate`, ale to fałszywie przestawiało stare, długo
  obowiązujące wpisy ZA późniejsze (sprawdzone na ustawie systemowej:
  `DU/2009/1585`, `expirationDate=2013-12-04`, plasowałby się po
  `DU/2013/1442`, mimo że jest wcześniejszy) -- zamiast tego użyto
  `announcementDate` (data ogłoszenia samego obwieszczenia), niezawodnie
  uzupełnianego nawet w najstarszych wpisach i poprawnie sortującego.
- **Obwieszczenia z `expirationDate=None` (wciąż otwarte) wykluczone z listy
  PRZESZŁYCH okien**, nie tylko z wyliczania granicy bieżącej wersji -- inaczej
  dawałyby drugą, identyczną "historyczną" wersję pokrywającą się z
  syntetycznym wpisem "bieżący" (znalezione przy smoke teście na ustawie o PPK).

Przy okazji smoke testu znaleziono i naprawiono też realny błąd: `act_title`/
`eli` zwracanych artykułów historycznych po cichu wyciekał tytuł/numer
SAMEGO OBWIESZCZENIA ("Obwieszczenie Marszałka Sejmu...") zamiast aktu
bazowego -- naprawione przepuszczeniem `base_eli`/`base_title` osobno przez
`fetch_version_windows()`/`process_act_version()`.

7 nowych testów jednostkowych dla `compute_version_windows` (łańcuch
chronologiczny, regresja błędu sortowania, otwarte/wygasłe najnowsze
obwieszczenie, zero obwieszczeń) plus jeden `skipif`-guarded test end-to-end.

**Faza 2 (`scripts/build_rag_index.py --include-history`, `scripts/rag_search.py`):**
`rag_search.py` dostał `rank_chunks()`/`_version_covers()` (czyste, testowalne
bez modelu/dysku -- pierwsze bezpośrednie testy prawdziwej logiki rankingu w
tym projekcie, wcześniej tylko atrapa `FakeRag`) i `RagIndex.search(...,
as_of=None)`. **Ważna zasada poprawności:** `as_of=None` filtruje "na dziś",
NIE pomija filtrowania -- inaczej przestarzała wersja historyczna mogłaby po
cichu wygrać rankingiem ze zwykłą, bieżącą wersją przy zapytaniu bez podanej
daty. Dowiedzione bajtowo no-opem na danych sprzed tej funkcji (wszystkie mają
`valid_from=valid_to=None`, więc filtr zawsze przepuszcza).

**Błąd złapany i naprawiony w trakcie implementacji (nie na etapie
przeglądu):** pierwsza wersja `build_chunks()` dopisywała pola
`valid_from`/`valid_to`/`announcement_eli` bezwarunkowo (`.get(..., None)`)
do KAŻDEGO chunka -- co łamało obietnicę "bajtowo identyczny wynik domyślnie"
z planu (zwykłe uruchomienie bez `--include-history` dostawało trzy dodatkowe
klucze o wartości `null` w każdym wpisie `rag_index_meta.json`). Wykryte
dopiero przez faktyczny `cmp` zapisanego wcześniej baseline'u względem
świeżego przebiegu domyślnego, nie przez samo czytanie kodu -- naprawione:
pola dopisywane TYLKO gdy artykuł źródłowy faktycznie niesie choć jedno z
nich. Ponownie zweryfikowano `cmp` -- bajtowa identyczność domyślnej ścieżki
potwierdzona zarówno dla `rag_index_meta.json`, jak i `rag_index.npy`.

**Realny błąd znaleziony przy ręcznej weryfikacji pilotażowej, naprawiony
osobno (poza zakresem forka, dograne od razu):** ustawy BEZ pliku
`_history.json` (czyli `download_acts_history.py` nigdy dla nich nie
uruchomiony) zostawały z `valid_from=None` bezwarunkowo -- bezpieczne dla
starych ustaw bez konkurencyjnych wersji, ale błędne dla ustawy o rynku pracy
(w mocy dopiero od 2025-06-01, bez obwieszczeń w ogóle, patrz Krok 18):
`as_of=2010-01-01` błędnie zwracał jej artykuły, jakby zawsze obowiązywała.
**Naprawiono** w `load_articles_with_history()`: `valid_from` bieżących
artykułów KAŻDEJ ustawy (nie tylko tych z `_history.json`) jest teraz
dodatkowo podbijane (`max`) datą `entryIntoForce` z już posiadanego
`data/raw/{short}_meta.json` -- bez dodatkowych zapytań HTTP, to pole jest
tam od zawsze. Zweryfikowano na żywo: `rag_search.py "zasiłek dla
bezrobotnych..." --as-of 2010-01-01` po poprawce nie zwraca już żadnego
artykułu ustawy o rynku pracy w top-5. Dodano 7 nowych testów jednostkowych
(`tests/test_build_rag_index.py`, z `monkeypatch` na ścieżki plików -- brak
pliku `_meta.json` też pokryty, nie wywala się).

**Ręczna weryfikacja end-to-end (`ustawa_o_minimalnym_wynagrodzeniu`, żywe
API):** pobrano 4 wersje historyczne (93 artykuły), przebudowano indeks z
`--include-history` (4364 -> 5604 fragmentów po dograniu PPK i minimalnego
wynagrodzenia). `rag_search.py "ile wynosi minimalne wynagrodzenie za pracę"
--as-of <data>` dla 5 dat z różnych okien poprawnie zwracał inny, zgodny z
datą fragment za każdym razem, z adnotacją `[stan: valid_from -- valid_to]`;
`as_of=None` i data bliska dziś poprawnie trafiały w to samo, bieżące okno
(`2024-11-27 -- nadal`). Zastrzeżenie: sama ustawa o minimalnym wynagrodzeniu
nie podaje kwoty w złotówkach wprost (ustala ją coroczne rozporządzenie Rady
Ministrów, poza `ACTS`) -- zweryfikowano więc poprawny wybór wersji artykułu
dla danej daty, nie literalną historyczną kwotę PLN.

**Stan testów:** 59/59 (było 36 przed Krokiem 18/19).

## Krok 20: warstwa prompt/czat dla "as-of" -- i ważne sprostowanie własnej weryfikacji

Cel: Faza 3 planu z Kroku 18 -- `prompt.py` (adnotacje wersji w
`build_context`, nowy punkt w `SYSTEM_PROMPT`, `as_of` przez
`search_with_history`/`build_user_message`) i identyczna, mechaniczna zmiana
w `chat.py`/`chat_lmstudio.py`/`chat_cuda.py` (`--as-of`, `/data RRRR-MM-DD`,
`/nowy` świadomie NIE czyści `as_of` -- ten sam precedens co z wgranym
plikiem). 6 nowych testów (65/65 łącznie), w tym regresja pilnująca, że
nagłówek fragmentu bez dat wersji jest bajtowo identyczny z dotychczasowym.

**Ważne sprostowanie ręcznej weryfikacji end-to-end.** Pierwsza próba
(`chat.py --as-of 2018-01-15 --prompt "Ile wynosi minimalne wynagrodzenie za
pracę?"`) zwróciła odpowiedź "2100 zł" z cytatem "art. 25 ... (stan prawny na
dzień 2018-01-15)" -- **liczba jest faktycznie poprawna historycznie (realna
płaca minimalna w styczniu 2018), ale NIE POCHODZI z dostarczonego
fragmentu.** Sprawdzone bezpośrednio w `data/processed/ustawa_o_minimalnym_wynagrodzeniu_history.json`:
art. 25 dla tego okna (2017-04-07 -- 2018-11-21) to legacy definicja
("ilekroć w przepisach jest mowa o »najniższym wynagrodzeniu« ... oznacza to
kwotę 760 zł") -- kompletnie inny, przestarzały koncept, niezwiązany z
faktyczną płacą minimalną z tamtego okresu. Model dostał więc poprawnie
wyszukany, poprawnie oznaczony datą fragment (mechanizm wyboru WERSJI
zadziałał bez zarzutu -- `rag_search.py --as-of 2018-01-15` zwraca dokładnie
ten sam, poprawnie oznaczony `[stan: 2017-04-07 -- 2018-11-21]` art. 25), ale
zamiast wprost przyznać się do braku danych (co nowy punkt 5 `SYSTEM_PROMPT`
miał wymusić), **zignorował treść fragmentu i podał poprawnie brzmiącą,
ale niepodpartą liczbę z pamięci wag, podpisując ją tak, jakby pochodziła z
cytowanego artykułu.**

**Dla kontrastu, to samo pytanie BEZ `--as-of` (bieżące prawo)** zwróciło
"760 zł" -- czyli model w tym wariancie WIERNIE zacytował dokładnie to, co
jest we fragmencie (poprawne zachowanie w sensie groundingu), ale to i tak
zła odpowiedź merytorycznie, bo 760 zł to ta sama, przestarzała definicja, nie
realna bieżąca płaca minimalna (rząd wielkości: ~4666 zł w 2025 -- ustawa w
ogóle nie zawiera aktualnej kwoty, patrz "Co dalej" pkt 5 niżej). **Ten
konkretny błąd (RAG trafiający w art. 25 zamiast w faktyczną informację o
kwocie, bo ta ostatnia po prostu nie istnieje w tej ustawie) jest
niezależny od funkcji "as-of" -- to ten sam, pre-existing problem co "Co
dalej" pkt 5** (kwota ustalana osobnym rozporządzeniem, poza `ACTS`).

**Wniosek:** mechanizm wyboru wersji artykułu na zadaną datę działa poprawnie
i to jest udowodnione niezależnie na dwa sposoby -- bezpośrednio przez
`rag_search.py --as-of` (Krok 19, wielokrotnie, na żywych danych) oraz teraz
na tym samym pytaniu. To, co NIE działa niezawodnie, to zachowanie modelu,
gdy dostarczony (poprawnie wybrany) fragment nie zawiera odpowiedzi na
zadane pytanie -- nowy punkt 5 `SYSTEM_PROMPT` obsługuje tylko przypadek
"żaden fragment nie pokrywa tej daty", nie przypadek "fragment pokrywa tę
datę, ale nie odpowiada na pytanie". To realne, warte odnotowania
ograniczenie warstwy modelu (ten sam rodzaj problemu co PPK 1,5%/2,5% z
Kroku 16), nie błąd w kodzie tej fazy -- naprawa wymagałaby albo lepszego
źródła danych (patrz "Co dalej" pkt 5 -- prawdziwa kwota w RAG rozwiązałaby
oba warianty pytania), albo dalszej pracy nad `SYSTEM_PROMPT`/LoRA nad
rozróżnianiem "fragment obecny" od "fragment odpowiada na pytanie" -- świadomie
NIE podjęto tej drugiej naprawy w tej fazie (spory, niepewny zakres, wykracza
poza plan wersjonowania czasowego).

## Krok 21: rozporządzenia o wysokości minimalnego wynagrodzenia ("Co dalej" pkt 5)

Cel: znalezisko z Kroku 20 -- ustawa o minimalnym wynagrodzeniu nie zawiera
samej kwoty w złotych, ustala ją co roku osobny akt. Nowy skrypt
`scripts/download_wage_regulations.py` pobiera i wersjonuje te akty,
podłączając się pod ten sam mechanizm "as-of" z Kroków 18-20 (bez ŻADNYCH
zmian w `rag_search.py` -- patrz niżej).

**Odkrycie źródła listy: `references["Akty wykonawcze"]` okazało się
NIEKOMPLETNE.** Pierwsza wersja skryptu czerpała listę lat z tego pola w
metadanych ustawy bazowej -- dało 22 wpisy, ale z DZIURĄ: brakowało roku
2021. Sprawdzone ręcznie przez `/eli/acts/search?title=...`: akt na 2021 r.
(`DU/2020/1596`) istnieje i jest poprawnie zindeksowany w ELI. **Przyczyna
znaleziona przy odczycie treści aktu, nie zgadywana:** to jedyny rok w całej
serii wydany na podstawie prawnej INNEJ ustawy -- "art. 79 ust. 5 ustawy z
dnia 31 marca 2020 r. o zmianie ustawy o szczególnych rozwiązaniach
związanych z zapobieganiem [...] COVID-19 [...]", nie na podstawie zwykłego
art. 2 ust. 5 ustawy o minimalnym wynagrodzeniu jak wszystkie pozostałe lata
-- covidowa nowelizacja tymczasowo przeniosła tę samą delegację do innej
ustawy, więc ELI poprawnie NIE zlinkował go jako "Akt wykonawczy" ustawy o
minimalnym wynagrodzeniu (wierne odzwierciedlenie realnej podstawy prawnej,
nie błąd/luka w samym ELI). **Naprawiono:** źródłem listy jest teraz
`/eli/acts/search?title="wysokości minimalnego wynagrodzenia za pracę"` --
zwraca kompletną, ciągłą serię 23 wpisów, po jednym na każdy rok 2004-2026,
w tym dwa najstarsze (2009, 2010) jako "Obwieszczenie Prezesa Rady
Ministrów" zamiast "Rozporządzenie Rady Ministrów" (inny mechanizm prawny z
tamtego okresu, ten sam wzorzec tytułu i treści). Dodano `_warn_on_gaps()` --
sprawdza ciągłość lat i ostrzega, gdyby ten sam typ luki wystąpił ponownie
przy odświeżeniu w przyszłości (nie zaufano samej liczbie wyników z
`search` bez weryfikacji).

**Model danych -- prostszy niż dla obwieszczeń "tekst jednolity" (Krok
18/19).** Każdy akt to jednorazowe, całkowicie odrębne rozporządzenie
(wchodzi w życie 1 stycznia danego roku, obowiązuje do wejścia w życie
NASTĘPNEGO w kolejności) -- `valid_from`/`valid_to` liczone wprost z roku w
tytule (regex, niezawodny nawet dla dwóch najstarszych wpisów bez pola
`entryIntoForce`), bez fuzzy granic i nakładających się okien, które
komplikowały Krok 18. Tekst używa numeracji "§ N." (nie "Art. N." jak w
ustawach) i jest krótki (2-3 paragrafy, dobrze poniżej `MAX_CHUNK_CHARS`),
więc traktowany jako JEDEN "artykuł" na wersję (`article = "1"` stałe na
wszystkie lata) -- działa od razu z niezmienionym mechanizmem grupowania
`(act_short, article)` w `rag_search.py`, zero zmian w tamtym pliku.

**Nowy typ pliku wejściowego dla `build_rag_index.py --include-history`:**
`data/processed/*_series.json` -- w przeciwieństwie do `*_history.json`
(wymaga patchowania "bieżącej", wcześniej już pobranej gdzie indziej wersji),
plik `*_series.json` to samodzielny, kompletny "akt" -- KAŻDA wersja,
łącznie z bieżącą, ma `valid_from`/`valid_to` ustawione już przy zapisie,
więc dogrywana wprost, bez patchowania. `load_articles_with_history()`
rozszerzona o obsługę tego wzorca (`PROCESSED_DIR.glob("*_series.json")`).

**Weryfikacja end-to-end (żywe dane, `chat.py`, Bielik-11B) -- ważna
poprawa względem Kroku 20:**
- `--as-of 2018-06-01 --prompt "Ile wynosi minimalne wynagrodzenie za
  pracę?"` -> poprawnie: "art. 1 Rozporządzenia [...] z 2017 r. [...]
  (Dz.U. z 2017 r. poz. 1747) [...] 2100 zł" -- **prawdziwie ugruntowane w
  dostarczonym fragmencie** (nie zmyślone jak w Kroku 20 -- ten sam
  scenariusz, teraz z realną, cytowalną treścią w kontekście), zgodne z
  realną stawką z 2018 r.
- Bez `--as-of` (domyślny `--top-k 5`): model **wciąż** odpowiadał "760 zł"
  (stary błąd z Kroku 20) -- **przyczyna: nowy fragment o kwocie bieżącej
  (rozporządzenie na 2026 r.) mieścił się w top-6 wyników RAG (score 0.847),
  ale NIE w domyślnym top-5** -- wyprzedzony przez dwa niezwiązane
  fragmenty ustawy o rynku pracy (art. 145, art. 218 -- o zasiłku dla
  bezrobotnych), które przypadkiem mają wysoki wynik podobieństwa
  semantycznego dla tego pytania. **Z `--top-k 8`** ten sam prompt (bez
  `--as-of`) poprawnie zwrócił "4806 zł" (rozporządzenie na 2026 r.,
  poprawnie zacytowane z Dz.U.). To ograniczenie samego RANKINGU RAG (ten
  sam rodzaj problemu co "rodzaje umów o pracę" z Kroku 3), NIE regresja
  mechanizmu "as-of" (który w obu testach -- historycznym i bieżącym --
  poprawnie wybrał właściwe okno czasowe) ani powrót hallucynacji z Kroku 20
  (gdy fragment trafia do kontekstu, model go wiernie i poprawnie cytuje w
  obu testach). Świadomie NIE zmieniono globalnego domyślnego `--top-k` (5)
  -- szerszy wpływ na długość promptu przy WSZYSTKICH pytaniach, nie tylko
  tych konkurujących z ustawą o rynku pracy; użytkownik może użyć wyższego
  `--top-k` dla tego typu pytań.

**Testy:** 6 nowych (`tests/test_download_wage_regulations.py`) dla
`extract_year`/`_warn_on_gaps`, w tym regresja dokładnie tej luki z roku
2021. **72/72 łącznie.**

**Zakres:** na razie tylko minimalne wynagrodzenie (na wyraźną prośbę
użytkownika, jako pilotaż). Limit 30-krotności (ustawa systemowa) i inne
podobne przypadki -- patrz "Co dalej" pkt 5 niżej, wzorzec z tego kroku
(`search_acts_by_title` + stały `article` + proste roczne okna) powinien się
przenosić bez większych zmian.

## Krok 22: limit 30-krotności składek ZUS -- drugi przypadek z tego samego wzorca

Cel: domknięcie drugiego kandydata z "Co dalej" pkt 5 -- roczny limit
ograniczenia podstawy wymiaru składek emerytalno-rentowych ("30-krotność",
art. 19 ustawy o systemie ubezpieczeń społecznych, DU/1998/887 -- sama
ustawa podaje tylko wzór/mnożnik, nie konkretną kwotę w złotych). Nowy
`scripts/download_zus_limit_regulations.py`, ten sam wzorzec co Krok 21
(`search_acts_by_title` + stały `article = "1"` + proste, nienakładające
się roczne okna) -- **świadomie od razu przez `/eli/acts/search`, nie
`references["Akty wykonawcze"]`**, bo ta druga ścieżka okazała się
niekompletna dla minimalnego wynagrodzenia (Krok 21); nie było powodu, żeby
tym razem miała być bardziej wiarygodna.

**Ten sam typ weryfikacji co w Kroku 21, ale bez powtórki tamtych
problemów:** `/eli/acts/search?title="kwoty ograniczenia rocznej podstawy
wymiaru składek"` zwrócił od razu kompletną, ciągłą serię 28 wpisów,
1999-2026, bez żadnych luk (`_warn_on_gaps` -- kopia funkcji z Kroku 21 --
nic nie zgłosiła). Wzorzec tytułu inny niż przy minimalnym wynagrodzeniu:
"w roku RRRR" zamiast "w RRRR r.", i -- dla starszych wpisów (1999-2007) --
rok bywa na samym końcu tytułu, a dla nowszych (od 2008, z dodatkową
klauzulą "oraz przyjętej do jej ustalenia [...]") -- w środku, nie na
końcu. Regex dopasowany osobno (`r"w roku (\d{4})"`, bez zakotwiczenia na
końcu stringa), sprawdzony na obu wariantach.

**Struktura tekstu jeszcze prostsza niż rozporządzenia płacowe:** jedno
zdanie, bez podziału na "§" w ogóle -- "ogłasza się, że kwota ograniczenia
[...] w roku RRRR wynosi X zł, a przyjęta do jej ustalenia kwota
prognozowanego przeciętnego wynagrodzenia wynosi Y zł." (sprawdzone na
MP/2024/1051). Jeden akt = jeden "artykuł", tak jak w Kroku 21.

**Drobna, nieszkodliwa anomalia znaleziona przy pobieraniu roku 2012
(MP/2011/1160):** `strip_not_yet_in_force_text` zgłosiła "niesparowane
nawiasy < >" -- sprawdzone ręcznie w wynikowym tekście: kompletny, spójny,
nic nie ucięte ani nie uszkodzone (prawdopodobnie artefakt konwersji PDF w
okolicach przypisu/indeksu górnego, nie prawdziwy fragment "jeszcze
nieobowiązującego" tekstu -- ten typ oznaczeń dotyczy tekstów ujednoliconych
ustaw, nie jednorazowych obwieszczeń). Bezpiecznik (niezachłanne
podstawienie) zadziałał zgodnie z projektem -- nic nie zostało po cichu
wycięte, tylko wypisane ostrzeżenie do wglądu.

**Weryfikacja end-to-end (żywe dane, `chat.py`, Bielik-11B) -- ten sam
wzorzec ograniczenia rankingu co w Kroku 21, tym razem jeszcze wyraźniejszy:**
domyślny `--top-k 5` NIE wystarczał w ŻADNYM z dwóch testów (`--as-of
2013-05-01` i bez `--as-of`) -- właściwy fragment obwieszczenia mieścił się
dopiero na 6.-7. miejscu, wyprzedzony przez sześć bardzo blisko
spunktowanych artykułów samej ustawy systemowej (prawdopodobnie art. 19 i
jego liczne ustępy, tematycznie nakładające się). **Z `--top-k 8` oba testy
wypadły bezbłędnie:**
- `--as-of 2013-05-01`: poprawnie zacytowano zarówno ogólną zasadę (art. 19
  ust. 1) jak i konkretną, historyczną kwotę za 2013 r. (111 390 zł,
  obwieszczenie z 14 grudnia 2012 r., poz. 1018) -- model dodatkowo
  samodzielnie zweryfikował arytmetykę (3713 zł x 30 = 111 390 zł).
- bez `--as-of` (stan bieżący): poprawnie zacytowano obwieszczenie na 2026
  r. (poz. 1206), kwota 282 600 zł.

Potwierdza to wniosek z Kroku 21 jako ogólniejszy: mechanizm "as-of" i
jakość samego groundingu są solidne -- ograniczeniem jest wyłącznie
domyślny budżet `--top-k`, gdy nowo dodana treść konkuruje z wieloma,
tematycznie bliskimi artykułami tej samej ustawy bazowej. Świadomie
pozostawiono bez zmiany globalnego domyślnego `--top-k` (5) z tego samego
powodu co w Kroku 21.

**Testy:** 5 nowych (`tests/test_download_zus_limit_regulations.py`),
łącznie **77/77**.

**Zakres:** dwa z co najmniej dwóch znanych kandydatów z "Co dalej" pkt 5
teraz zrobione (minimalne wynagrodzenie -- Krok 21, limit 30-krotności --
ten krok). Przeciętne wynagrodzenie ogłaszane komunikatem Prezesa GUS
(używane do wielu innych przeliczeń -- odprawy, niektóre zasiłki) wciąż nie
jest dodane -- ten sam wzorzec powinien się przenosić.

## Krok 23: przeciętne wynagrodzenie GUS -- trzeci przypadek, ostrożniej niż poprzednie dwa

Cel: domknięcie trzeciego, ostatniego znanego kandydata z "Co dalej" pkt 5.
Nowy `scripts/download_avg_wage_regulations.py` -- ale ten przypadek okazał
się realnie bardziej ryzykowny niż Kroki 21/22, więc zakres jest tu celowo
węższy.

**Ryzyko znalezione PRZED napisaniem kodu (dobrze, że sprawdzone najpierw):**
`/eli/acts/search?title="przeciętnego wynagrodzenia w gospodarce narodowej"`
zwraca 81 wyników, bo GUS publikuje pod bardzo podobnymi tytułami KILKA
różnych, prawnie odrębnych wielkości: (1) "Komunikat [...] w sprawie
przeciętnego wynagrodzenia w gospodarce narodowej w RRRR r." -- wartość
ROCZNA, podstawa prawna art. 20 ustawy o emeryturach i rentach z FUS (2)
"Obwieszczenie [...] w sprawie przeciętnego wynagrodzenia MIESIĘCZNEGO w
gospodarce narodowej w RRRR r. I W DRUGIM PÓŁROCZU RRRR r." -- inna
wielkość, inne zastosowania (3) "Obwieszczenie [...] w sprawie przeciętnego
[...] wynagrodzenia [...] W WOJEWÓDZTWACH w RRRR r." -- rozbicie regionalne.
Luźny filtr podłańcuchowy (jak w Krokach 21/22) złapałby WSZYSTKIE trzy pod
jednym `act_short`, mieszając odrębne wielkości -- gorsze niż brak danych,
bo model mógłby podać niewłaściwą, ale przekonująco brzmiącą liczbę.
**Naprawiono zanim się stało problemem:** `TITLE_EXACT_RE` -- regex
zakotwiczony na POCZĄTKU i KOŃCU tytułu, nie luźny substring -- łapie
WYŁĄCZNIE wariant (1). Sprawdzone: dokładny wzorzec zwraca czystą, ciągłą
serię 23 wpisów, 2003-2025, bez przerw.

**`valid_from`/`valid_to` liczone inaczej niż w Krokach 21/22 -- z
`promulgation` (faktyczna data ogłoszenia), NIE z "1 stycznia roku z
tytułu".** Powód: w przeciwieństwie do rozporządzeń płacowych/obwieszczeń
o limicie (które mają wprost klauzulę "wchodzi w życie z dniem 1 stycznia
RRRR r."), komunikat GUS o przeciętnym wynagrodzeniu retrospektywnie
ogłasza fakt za rok miniony, bez klauzuli wskazującej, od kiedy dokładnie
staje się "tą właściwą" wartością dla różnych, korzystających z niej ustaw
(zależy od przepisów każdej z osobna). Użycie `promulgation` (zawsze pewne,
zweryfikowane wprost z metadanych) zamiast zgadywanej daty jest
bezpieczniejsze -- nie twierdzi nic więcej, niż da się sprawdzić.
**Konsekwencja UX warta odnotowania:** pytanie "ile wynosiło przeciętne
wynagrodzenie w 2022 r." wymaga `--as-of` z datą w 2023 r. (kiedy fakt za
2022 r. został OGŁOSZONY), nie w 2022 r. -- trochę nieintuicyjne, ale
zgodne z rzeczywistą semantyką tych danych.

**Znalezisko przy pobieraniu (kosmetyczne, nie blokujące):** 6 z 23
najstarszych wpisów (lata 2003-2008, publikowane 2004-2009) ma zniekształcone
polskie znaki diakrytyczne w wyekstrahowanym tekście ("przeci´tne" zamiast
"przeciętne", "wynios∏o" zamiast "wyniosło") -- artefakt starego
kodowania/fontu w PDF-ach Monitora Polskiego sprzed ~2010 r., którego
`pypdf` nie mapuje poprawnie. **Sama liczba (kwota w zł) pozostaje
poprawna** -- sprawdzone bezpośrednio: 2003 -> 1829,24 zł, 2008 -> 2943,88
zł, oba zgodne ze znanymi historycznymi wartościami. Świadomie NIE
naprawiono (niska wartość względem nakładu -- naprawa kodowania starych
fontów PDF to osobny, głębszy problem, dotyczy tylko 6 najstarszych z 23
wpisów, a sama treść merytoryczna -- liczba -- nie jest uszkodzona).

**Weryfikacja end-to-end (żywe dane, `chat.py`, Bielik-11B):**
- Pytanie o rok 2022 BEZ `--as-of` (czyli efektywnie "stan bieżący") --
  model POPRAWNIE odmówił zgadywania ("Nie znalazłem [...] informacji o
  [...] 2022 roku"), zamiast pomylić to z inną, bieżącą wartością z
  kontekstu -- oczekiwane i poprawne zgodnie z semantyką `promulgation`
  opisaną wyżej (as_of=dziś nie pokrywa okna z 2022 r.).
- Z `--as-of 2023-06-01` (data w oknie ważności komunikatu za 2022 r.) --
  poprawnie zacytowano "6346,15 zł", zgodne z realną historyczną wartością.

**Testy:** 6 nowych (`tests/test_download_avg_wage_regulations.py`, w tym
regresja obu świadomie odrzucanych wariantów tytułu -- półrocznego i
wojewódzkiego). Łącznie **83/83**.

**Zakres "Co dalej" pkt 5 zamknięty:** wszystkie trzy znane kandydaci
(minimalne wynagrodzenie, limit 30-krotności, przeciętne wynagrodzenie GUS)
zrobione (Kroki 21-23). Nowe podobne przypadki, jeśli się pojawią, powinny
najpierw przejść ten sam wstępny test: czy `/eli/acts/search` po
oczywistym tytule zwraca JEDNĄ, czystą serię, czy kilka pomieszanych --
zanim się napisze kod zakładający luźny filtr podłańcuchowy.

## Krok 24: podniesienie domyślnego `--top-k` (5 -> 8)

Cel: domknięcie "Co dalej" pkt 6 -- ten sam wzorzec ograniczenia rankingu
RAG znaleziony niezależnie w Krokach 21 i 22 (nowa treść z rozporządzeń
przegrywała z domyślnym `--top-k 5` z kilkoma tematycznie bliskimi
artykułami tej samej ustawy bazowej).

**Decyzja: proste globalne podniesienie (5 -> 8), NIE heurystyka
adaptacyjna dla pytań o kwoty.** Rozważono odpowiednik `looks_like_meta_question`
(wykrywanie pytań "ile wynosi/jaki jest limit" i podnoszenie `top_k` tylko
dla nich) -- odrzucone jako niepotrzebna złożoność: proste globalne
podniesienie już rozwiązało oba znane, potwierdzone przypadki (Krok 21 i
22) i nie wprowadziło regresji na kontrolnych pytaniach sprawdzonych
ręcznie (patrz niżej) -- kolejna heurystyka regexowa dodawałaby kod do
utrzymania bez potwierdzonej potrzeby.

**Ręczna weryfikacja braku regresji (żywe dane, `chat.py`, Bielik-11B) na
dwóch znanych pytaniach kontrolnych z wcześniejszych kroków:**
- "Ile dni urlopu przysługuje po 10 latach pracy?" (znane dobre pytanie z
  Kroku 4c) -- z `--top-k 8` nadal poprawnie: 26 dni, art. 154 § 1 pkt 2.
- "Jakie są rodzaje umów o pracę?" (znane SŁABE pytanie z Kroku 3, RAG od
  zawsze nie trafiał w art. 25 nawet w top-5 na dużo mniejszym indeksie) --
  z `--top-k 8` na obecnym, znacznie większym indeksie (5770 fragmentów)
  wciąż nie trafia -- ale model bezpiecznie przyznaje się do braku
  informacji zamiast zgadywać, zamiast pogorszenia to ten sam, już znany i
  udokumentowany, bezpieczny tryb awarii.

Zmienione: domyślne `--top-k` w `chat.py`/`chat_lmstudio.py`/`chat_cuda.py`
(argparse) oraz `RagIndex.search()`/`rag_search.py` CLI (dla spójności --
te dwa miejsca i tak zawsze były przesłaniane jawnym `args.top_k` z
chat*.py, ale podniesiono też je, żeby zachowanie biblioteki bez zmian w
wywołującym kodzie było spójne z nowym domyślnym doświadczeniem czatu).
Koszt: dłuższy prompt (np. ~8-15k tokenów zamiast ~6-7k na pytaniach z
tego kroku) -- wciąż daleko poniżej praktycznych limitów, zaakceptowany.

## Co dalej

1. ~~Zweryfikować ręcznie 3 rozbieżności `changeDate` znalezione przy
   kroku 17~~ **Domknięte przy okazji Kroku 19.** Bajtowa weryfikacja
   refaktoru `_extract_articles` (dwa pełne, żywe uruchomienia
   `download_acts.py`) potwierdziła, że `all_articles.json` jest bajtowo
   identyczny mimo że `changeDate` w metadanych 5 z 7 ustaw (Kodeks pracy,
   ustawa systemowa, PIT, ustawa zasiłkowa, ustawa o rynku pracy) się
   przesunęło -- to nieistotne zmiany metadanych ELI (np. techniczne
   odświeżenie rekordu), nie realne nowelizacje wymagające ponownego
   przebudowania indeksu. Świeże `data/raw/*_meta.json` scommitowane w
   ramach Kroku 19 (`git log`, "Dodaj wersjonowanie czasowe..."); po
   wypchnięciu na `origin` kolejny przebieg `acts-freshness.yml` powinien
   znów raportować "aktualne" dla wszystkich 7 ustaw -- potwierdzono lokalnie
   `check_acts_freshness.py` (2026-08-29): wszystkie 7 OK.
2. Mniejsza waga niż wcześniej sądzono (patrz krok 16, domknięcie na
   11B) -- ale wciąż warto: wzmocnić dane treningowe dla 4.5B
   przykładami KONTRASTUJĄCYMI wprost różne, blisko siebie leżące
   wartości w tym samym akcie (np. "czym różni się wpłata podstawowa od
   dodatkowej w PPK"), jeśli zależałoby nam na tym, żeby wariant 4.5B
   był tak samo niezawodny jak 11B na nowo dodanych tematach.
3. Wariant CUDA (krok 9) wymaga realnego testu na maszynie z kartą
   NVIDIA -- użytkownik nie ma takiego sprzętu i nie planuje testować
   tego samodzielnie, więc pozostaje "nieprzetestowane" do czasu, aż
   zrobi to ktoś inny z dostępem do NVIDIA; jeśli to nastąpi,
   zaktualizować ten plik i README (usunąć oznaczenie, poprawić
   ewentualne niezgodności API).
4. ~~Wersjonowanie w czasie -- prawo obowiązujące na dany dzień w
   przeszłości, nie tylko aktualne.~~ **Domknięte w Krokach 18-24**, wg
   planu w `/Users/szymon/.claude/plans/drifting-toasting-wozniak.md`
   (spike -> warstwa danych/indeksu -> warstwa prompt/czat -> integracja z
   rozporządzeniami o kwotach -> top-k). `download_acts_history.py`
   pobiera i wersjonuje historyczne obwieszczenia (`{short}_history.json`),
   `build_rag_index.py --include-history` scala je z bieżącym indeksem,
   `rag_search.py`/`prompt.py`/`chat*.py` przyjmują `--as-of RRRR-MM-DD`
   / `/data RRRR-MM-DD`. Zweryfikowane end-to-end na żywych danych (Krok
   20) z jawnym sprostowaniem własnej pierwszej, zbyt optymistycznej
   weryfikacji (halucynacja modelu omyłkowo uznana za poprawne
   grounding). Świadomie poza zakresem, nienaprawione: daty sprzed
   pierwszego dostępnego obwieszczenia danej ustawy (dla większości ustaw
   to lata 2012-2016, dla ustawy o minimalnym wynagrodzeniu ~2015 mimo że
   ustawa obowiązuje od 2003 -- patrz Krok 18, tabela) nie mają pokrycia;
   model ma to wprost powiedzieć, zamiast zgadywać z tekstu pierwotnego.
   Rekonstrukcja brzmienia sprzed pierwszego obwieszczenia z listy "Akty
   zmieniające" pozostaje niezbadana i nierozpoczęta -- osobne,
   znacznie trudniejsze zadanie, do rozważenia tylko jeśli pojawi się
   konkretna potrzeba.
5. ~~Rozporządzenia/obwieszczenia z konkretnymi kwotami, których nie ma w
   samych ustawach.~~ **Domknięte w Krokach 21-23** -- wszyscy trzej znani
   kandydaci zrobieni tym samym wzorcem (`search_acts_by_title` + stały
   `article` + proste roczne okna): minimalne wynagrodzenie (Krok 21), limit
   30-krotności (Krok 22), przeciętne wynagrodzenie GUS (Krok 23, węższy
   zakres -- patrz tam po realne ryzyko pomylenia kilku pokrewnych serii
   GUS). Nowe podobne przypadki, jeśli się pojawią: najpierw sprawdź, czy
   `/eli/acts/search` po tytule zwraca JEDNĄ czystą serię, nie kilka
   pomieszanych (patrz Krok 23), zanim napiszesz kod zakładający luźny
   filtr podłańcuchowy.
6. ~~Domyślny `--top-k 5` bywa za niski dla nowo dodanych rozporządzeń.~~
   **Domknięte w Kroku 24** -- podniesiono globalnie do 8 (`chat.py`/
   `chat_lmstudio.py`/`chat_cuda.py`/`rag_search.py`), zweryfikowano brak
   regresji na znanych pytaniach kontrolnych z Kroków 3/4c.
