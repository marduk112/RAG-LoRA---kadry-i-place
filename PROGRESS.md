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

## Co dalej

1. Do rozważenia: większy, bardziej zróżnicowany zbiór LoRA (więcej
   przykładów granicznych) jeśli zależałoby nam na poprawie
   niezawodności modelu również bez kontekstu RAG.
2. Wariant CUDA (krok 9) wymaga realnego testu na maszynie z kartą
   NVIDIA -- jeśli ktoś to zrobi, zaktualizować ten plik i README
   (usunąć oznaczenie "nieprzetestowane", poprawić ewentualne
   niezgodności API).
2. Do rozważenia: uruchamianie `check_acts_freshness.py` okresowo
   (np. cron/`schedule`), żeby wychwycić nowelizacje bez czekania na
   ręczne sprawdzenie.
