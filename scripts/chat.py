"""
Asystent kadrowo-płacowy: RAG (rag_search.RagIndex) + douczony Bielik (LoRA).

RAG dostarcza aktualny, źródłowy tekst artykułów ustaw jako kontekst w
promptcie -- LoRA uczy model odpowiedniego stylu odpowiedzi (zwięzłość,
cytowanie artykułów, przyznawanie się do niewiedzy, gdy kontekst nie
wystarcza). Fakty mają pochodzić z kontekstu RAG, a nie z pamięci wag
modelu -- to naprawia niedokładności zaobserwowane w testach samego
douczonego modelu bez kontekstu (patrz PROGRESS.md, krok 4b).

Domyślny model: Bielik-11B-v3.0-Instruct + adapter bielik11b-kadry-lora-iter25.

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

from rag_search import RagIndex

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "Bielik-11B-v3.0-Instruct-mlx"
DEFAULT_ADAPTER_PATH = ROOT / "adapters" / "bielik11b-kadry-lora-iter25"

SYSTEM_PROMPT = """Jesteś asystentem kadrowo-płacowym, odpowiadasz po polsku na pytania \
dotyczące polskiego prawa pracy i ubezpieczeń społecznych.

Poniżej, przed każdym pytaniem użytkownika, otrzymasz fragmenty aktów \
prawnych wyszukane jako potencjalnie pomocny kontekst. Zasady:
1. Odpowiadaj WYŁĄCZNIE na podstawie dostarczonych fragmentów -- nie \
korzystaj z własnej wiedzy o konkretnych liczbach, kwotach czy terminach.
2. Jeśli dostarczone fragmenty nie zawierają odpowiedzi na pytanie, wprost \
napisz, że nie znalazłeś tego w dostępnych aktach prawnych, i nie zgaduj.
3. Cytuj numer artykułu i nazwę aktu, na którym opierasz odpowiedź.
4. Nie jesteś substytutem porady prawnej ani księgowej -- w sprawach \
spornych zasugeruj konsultację ze specjalistą."""


def build_context(results: list[dict]) -> str:
    parts = []
    for r in results:
        parts.append(
            f"### {r['act_title']} -- art. {r['article']}\n{r['text']}"
        )
    return "\n\n".join(parts)


def answer(model, tokenizer, rag: RagIndex, question: str, top_k: int, verbose: bool) -> str:
    results = rag.search(question, top_k=top_k)
    context = build_context(results)
    user_content = (
        f"Fragmenty aktów prawnych:\n\n{context}\n\n---\n\nPytanie: {question}"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    return generate(model, tokenizer, prompt=prompt, max_tokens=500, verbose=verbose)


def main():
    parser = argparse.ArgumentParser(description="Asystent kadrowo-płacowy (RAG + Bielik LoRA)")
    parser.add_argument("--prompt", type=str, default=None, help="Pojedyncze pytanie zamiast trybu interaktywnego")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="Ścieżka do modelu bazowego w formacie MLX")
    parser.add_argument("--adapter-path", type=str, default=str(DEFAULT_ADAPTER_PATH))
    parser.add_argument("--no-adapter", action="store_true", help="Użyj bazowego modelu bez LoRA")
    parser.add_argument("--top-k", type=int, default=5, help="Liczba fragmentów RAG dołączanych do kontekstu")
    args = parser.parse_args()

    adapter_path = None if args.no_adapter else args.adapter_path
    print(f"[INFO] Ładowanie modelu ({Path(args.model).name}, adapter={adapter_path or '(brak)'})...")
    model, tokenizer = load(args.model, adapter_path=adapter_path)

    print("[INFO] Ładowanie indeksu RAG...")
    rag = RagIndex()

    if args.prompt:
        answer(model, tokenizer, rag, args.prompt, args.top_k, verbose=True)
        return

    print("\nAsystent kadrowo-płacowy gotowy. Wpisz pytanie (Ctrl+C aby zakończyć).\n")
    while True:
        try:
            question = input("Ty: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        print("Asystent: ", end="", flush=True)
        text = answer(model, tokenizer, rag, question, args.top_k, verbose=False)
        print(text, "\n")


if __name__ == "__main__":
    main()
