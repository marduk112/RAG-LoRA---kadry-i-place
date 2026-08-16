"""
Asystent kadrowo-płacowy: RAG (rag_search.RagIndex) + douczony Bielik (LoRA).

RAG dostarcza aktualny, źródłowy tekst artykułów ustaw jako kontekst w
promptcie -- LoRA uczy model odpowiedniego stylu odpowiedzi (zwięzłość,
cytowanie artykułów, przyznawanie się do niewiedzy, gdy kontekst nie
wystarcza). Fakty mają pochodzić z kontekstu RAG, a nie z pamięci wag
modelu -- to naprawia niedokładności zaobserwowane w testach samego
douczonego modelu bez kontekstu (patrz PROGRESS.md, krok 4b).

Domyślny model: Bielik-11B-v3.0-Instruct + adapter bielik11b-kadry-lora-iter25.

Tryb interaktywny pamięta kontekst rozmowy (poprzednie pytania i
odpowiedzi trafiają do promptu przy kolejnych turach) -- wpisz /nowy,
żeby zacząć nowy wątek i wyczyścić historię. Historia jest ucinana do
ostatnich --max-turns par pytanie/odpowiedź, żeby nie przepełnić okna
kontekstu modelu (każda tura dokłada też fragmenty RAG).

Użycie:
    python scripts/chat.py                          # tryb interaktywny
    python scripts/chat.py --prompt "pytanie..."     # pojedyncze pytanie
    python scripts/chat.py --no-adapter              # bazowy model bez LoRA
    python scripts/chat.py \\
        --model models/Bielik-4.5B-v3.0-Instruct-mlx \\
        --adapter-path adapters/bielik-kadry-lora-iter50  # szybszy wariant 4.5B
"""

import argparse
from pathlib import Path

from mlx_lm import generate, load

from prompt import SYSTEM_PROMPT, build_user_message, looks_like_meta_question, search_with_history, trim_history
from rag_search import RagIndex

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "Bielik-11B-v3.0-Instruct-mlx"
DEFAULT_ADAPTER_PATH = ROOT / "adapters" / "bielik11b-kadry-lora-iter25"


def answer(model, tokenizer, rag: RagIndex, history: list[dict], question: str, top_k: int, verbose: bool) -> str:
    if history and looks_like_meta_question(question):
        # Pytanie o przebieg rozmowy -- pomijamy fragmenty RAG (wyszukane
        # od nowa na podstawie samej treści pytania, więc dla takich pytań
        # bywają nietrafione i zaburzają korzystanie z historii; patrz
        # prompt.looks_like_meta_question i PROGRESS.md, krok 10).
        current_turn = {"role": "user", "content": question}
    else:
        results = search_with_history(rag, history, question, top_k)
        current_turn = {"role": "user", "content": build_user_message(question, results)}
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
    parser.add_argument("--top-k", type=int, default=5, help="Liczba fragmentów RAG dołączanych do kontekstu")
    parser.add_argument("--max-turns", type=int, default=6, help="Ile ostatnich par pytanie/odpowiedź zachować w kontekście rozmowy")
    args = parser.parse_args()

    adapter_path = None if args.no_adapter else args.adapter_path
    print(f"[INFO] Ładowanie modelu ({Path(args.model).name}, adapter={adapter_path or '(brak)'})...")
    model, tokenizer = load(args.model, adapter_path=adapter_path)

    print("[INFO] Ładowanie indeksu RAG...")
    rag = RagIndex()

    if args.prompt:
        answer(model, tokenizer, rag, [], args.prompt, args.top_k, verbose=True)
        return

    print("\nAsystent kadrowo-płacowy gotowy. Wpisz pytanie (Ctrl+C aby zakończyć, /nowy aby zacząć nowy wątek).\n")
    history: list[dict] = []
    while True:
        try:
            question = input("Ty: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question == "/nowy":
            history = []
            print("[INFO] Rozpoczęto nowy wątek rozmowy.\n")
            continue
        print("Asystent: ", end="", flush=True)
        text = answer(model, tokenizer, rag, history, question, args.top_k, verbose=False)
        print(text, "\n")
        history = trim_history(history, args.max_turns)


if __name__ == "__main__":
    main()
