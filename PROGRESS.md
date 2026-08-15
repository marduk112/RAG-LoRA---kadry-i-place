# Postęp prac — asystent kadrowo-płacowy (Bielik + RAG + LoRA)

Ten plik to dziennik tego, co zostało zrobione krok po kroku, żebyś mógł
się z tym zapoznać. Docelowa dokumentacja "jak tego używać" trafi do
README.md na końcu projektu.

## Krok 0-1: Środowisko

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
