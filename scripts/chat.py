"""
Asystent kadrowo-płacowy: RAG (rag_search.RagIndex) + douczony Bielik (LoRA).

RAG dostarcza aktualny, źródłowy tekst artykułów ustaw jako kontekst w
promptcie -- LoRA uczy model odpowiedniego stylu odpowiedzi (zwięzłość,
cytowanie artykułów, przyznawanie się do niewiedzy, gdy kontekst nie
wystarcza). Fakty mają pochodzić z kontekstu RAG, a nie z pamięci wag
modelu -- to naprawia niedokładności zaobserwowane w testach samego
douczonego modelu bez kontekstu (patrz PROGRESS.md, krok 4b).

Domyślny model: Bielik-11B-v3.0-Instruct + adapter
bielik11b-kadry-lora-hedge-iter50 (patrz PROGRESS.md, Krok 27 -- douczony
na zbiorze z Kroku 26 rozszerzonym o przykłady hedgingu). Zastrzeżenie:
ten adapter NIE jest w 100% niezawodny w przyznawaniu się do niewiedzy
poza korpusem RAG (patrz Krok 27, test VAT) -- czysty model bazowy bez
adaptera (--no-adapter) wypadał na tym wymiarze czyściej we wszystkich
dotąd przetestowanych przypadkach, kosztem gorszego stylu/cytowania na
faktach, które RAG dostarcza poprawnie.

Tryb interaktywny pamięta kontekst rozmowy (poprzednie pytania i
odpowiedzi trafiają do promptu przy kolejnych turach) -- wpisz /nowy,
żeby zacząć nowy wątek i wyczyścić historię. Historia jest ucinana do
ostatnich --max-turns par pytanie/odpowiedź, żeby nie przepełnić okna
kontekstu modelu (każda tura dokłada też fragmenty RAG).

Można też wgrać własny plik PDF (tekstowy, nie skan) jako dodatkowy
kontekst -- wpisz /plik <ścieżka> w trybie interaktywnym albo podaj
--file przy starcie. Fragmenty z pliku są wyraźnie oznaczane jako
"treść pliku użytkownika", nie mylone z obowiązującym prawem.

Można też zapytać o stan prawny na konkretny dzień w przeszłości (wymaga
indeksu RAG zbudowanego z --include-history, patrz PROGRESS.md Krok 18/19)
-- wpisz /data RRRR-MM-DD w trybie interaktywnym albo podaj --as-of przy
starcie; samo /data (bez daty) wraca do stanu bieżącego.

Użycie:
    python scripts/chat.py                          # tryb interaktywny
    python scripts/chat.py --prompt "pytanie..."     # pojedyncze pytanie
    python scripts/chat.py --no-adapter              # bazowy model bez LoRA
    python scripts/chat.py --file umowa.pdf          # z wgranym plikiem od startu
    python scripts/chat.py --as-of 2019-06-01        # stan prawny na dany dzień
    python scripts/chat.py \\
        --model models/Bielik-4.5B-v3.0-Instruct-mlx \\
        --adapter-path adapters/bielik-kadry-lora-iter50  # szybszy wariant 4.5B
"""

import argparse
from datetime import date
from pathlib import Path

from mlx_lm import generate, load

from file_index import SessionFileIndex
from prompt import SYSTEM_PROMPT, build_user_message, looks_like_meta_question, search_with_history, trim_history
from rag_search import RagIndex

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "Bielik-11B-v3.0-Instruct-mlx"
DEFAULT_ADAPTER_PATH = ROOT / "adapters" / "bielik11b-kadry-lora-hedge-iter50"


def answer(
    model,
    tokenizer,
    rag: RagIndex,
    file_index: SessionFileIndex,
    history: list[dict],
    question: str,
    top_k: int,
    verbose: bool,
    as_of: str | None = None,
) -> str:
    if history and looks_like_meta_question(question):
        # Pytanie o przebieg rozmowy -- pomijamy fragmenty RAG (wyszukane
        # od nowa na podstawie samej treści pytania, więc dla takich pytań
        # bywają nietrafione i zaburzają korzystanie z historii; patrz
        # prompt.looks_like_meta_question i PROGRESS.md, krok 10).
        current_turn = {"role": "user", "content": question}
    else:
        results = search_with_history(rag, history, question, top_k, as_of=as_of)
        file_results = file_index.search(question, top_k=3)
        current_turn = {
            "role": "user",
            "content": build_user_message(question, results, file_results, as_of=as_of),
        }
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [current_turn]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    text = generate(model, tokenizer, prompt=prompt, max_tokens=500, verbose=verbose)
    # W historii trzymamy samo pytanie (bez fragmentów RAG) -- fragmenty
    # doklejane są od nowa przy każdej turze, więc trzymanie ich też w
    # historii bardzo szybko przepełnia okno kontekstu (patrz PROGRESS.md,
    # krok 10).
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": text})
    return text


def main():
    parser = argparse.ArgumentParser(description="Asystent kadrowo-płacowy (RAG + Bielik LoRA)")
    parser.add_argument("--prompt", type=str, default=None, help="Pojedyncze pytanie zamiast trybu interaktywnego")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="Ścieżka do modelu bazowego w formacie MLX")
    parser.add_argument("--adapter-path", type=str, default=str(DEFAULT_ADAPTER_PATH))
    parser.add_argument("--no-adapter", action="store_true", help="Użyj bazowego modelu bez LoRA")
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Liczba fragmentów RAG dołączanych do kontekstu (domyślnie 8 -- podniesione z 5, patrz "
        "PROGRESS.md, Krok 21/22/24: przy 5 fragment z nowo dodanych rozporządzeń o konkretnych kwotach "
        "czasem przegrywał rankingiem z kilkoma tematycznie bliskimi artykułami tej samej ustawy bazowej)",
    )
    parser.add_argument("--max-turns", type=int, default=6, help="Ile ostatnich par pytanie/odpowiedź zachować w kontekście rozmowy")
    parser.add_argument("--file", type=str, default=None, help="Ścieżka do pliku PDF (tekstowego) wgrywanego jako dodatkowy kontekst")
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        metavar="RRRR-MM-DD",
        help="Pytaj o stan prawny na ten dzień zamiast bieżącego (wymaga indeksu "
        "zbudowanego z build_rag_index.py --include-history)",
    )
    args = parser.parse_args()

    adapter_path = None if args.no_adapter else args.adapter_path
    print(f"[INFO] Ładowanie modelu ({Path(args.model).name}, adapter={adapter_path or '(brak)'})...")
    model, tokenizer = load(args.model, adapter_path=adapter_path)

    print("[INFO] Ładowanie indeksu RAG...")
    rag = RagIndex()
    file_index = SessionFileIndex(model=rag.model)

    if args.file:
        n = file_index.add_pdf(args.file)
        print(f"[INFO] Wgrano plik {args.file} ({n} fragmentów)")

    if args.as_of:
        print(f"[INFO] Stan prawny na dzień: {args.as_of}")

    if args.prompt:
        answer(model, tokenizer, rag, file_index, [], args.prompt, args.top_k, verbose=True, as_of=args.as_of)
        return

    print("\nAsystent kadrowo-płacowy gotowy. Wpisz pytanie (Ctrl+C aby zakończyć, /nowy aby zacząć nowy wątek, /plik <ścieżka> aby wgrać PDF, /data RRRR-MM-DD aby zapytać o stan prawny na dany dzień).\n")
    history: list[dict] = []
    as_of = args.as_of
    while True:
        try:
            question = input("Ty: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question == "/nowy":
            # Celowo NIE czyści `as_of` -- tak samo jak nie czyści wgranego
            # pliku (file_index) -- to osobny "tryb" sesji, nie część
            # historii rozmowy; czyści go wyłącznie jawne /data.
            history = []
            print("[INFO] Rozpoczęto nowy wątek rozmowy.\n")
            continue
        if question.startswith("/plik "):
            path = question[len("/plik "):].strip()
            try:
                n = file_index.add_pdf(path)
                print(f"[INFO] Wgrano plik {path} ({n} fragmentów)\n")
            except (FileNotFoundError, ValueError) as e:
                print(f"[BŁĄD] {e}\n")
            continue
        if question == "/data" or question.startswith("/data "):
            arg = question[len("/data"):].strip()
            if not arg:
                as_of = None
                print("[INFO] Wrócono do stanu bieżącego.\n")
            else:
                try:
                    date.fromisoformat(arg)
                except ValueError as e:
                    print(f"[BŁĄD] Nieprawidłowa data ({e}), oczekiwano RRRR-MM-DD.\n")
                    continue
                as_of = arg
                print(f"[INFO] Stan prawny na dzień: {as_of}\n")
            continue
        print("Asystent: ", end="", flush=True)
        text = answer(model, tokenizer, rag, file_index, history, question, args.top_k, verbose=False, as_of=as_of)
        print(text, "\n")
        history = trim_history(history, args.max_turns)


if __name__ == "__main__":
    main()
