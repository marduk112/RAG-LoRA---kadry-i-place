"""
Asystent kadrowo-płacowy: RAG (rag_search.RagIndex) + model załadowany w
LM Studio (lokalny serwer zgodny z API OpenAI), zamiast lokalnego MLX.

Przydatne, gdy masz już w LM Studio załadowany model (np. Bielik w
formacie GGUF) i nie chcesz go dublować w MLX. Ponownie wykorzystuje
dokładnie ten sam prompt systemowy i sposób budowania kontekstu z RAG
co scripts/chat.py (wspólny moduł scripts/prompt.py) -- różni się
tylko backendem generacji. Ten skrypt NIE zależy od mlx_lm, więc
działa też na Linuksie/Windows (wszystko, co robi, to zapytania HTTP
do lokalnego serwera LM Studio + embedding modelu RAG przez
sentence-transformers, który jest wieloplatformowy).

Uwaga: LoRA wytrenowane w scripts/chat.py (adapters/*) NIE działa tutaj
-- to wagi w formacie MLX, a LM Studio ładuje osobny plik GGUF. Ten
skrypt daje sam RAG + Twój model z LM Studio, bez dotrenowanego stylu.

Wymaga uruchomionego serwera lokalnego w LM Studio: zakładka
"Developer" -> "Start Server" (domyślnie http://localhost:1234).

Tryb interaktywny pamięta kontekst rozmowy (poprzednie pytania i
odpowiedzi trafiają do promptu przy kolejnych turach) -- wpisz /nowy,
żeby zacząć nowy wątek i wyczyścić historię. Historia jest ucinana do
ostatnich --max-turns par pytanie/odpowiedź, żeby nie przepełnić okna
kontekstu modelu (każda tura dokłada też fragmenty RAG).

Użycie:
    python scripts/chat_lmstudio.py                      # tryb interaktywny
    python scripts/chat_lmstudio.py --prompt "pytanie..."
    python scripts/chat_lmstudio.py --model bielik-11b-v3.0-instruct  # jeśli LM Studio ma kilka modeli naraz
    python scripts/chat_lmstudio.py --url http://localhost:1234/v1
"""

import argparse
import sys

import requests

from prompt import SYSTEM_PROMPT, build_user_message, looks_like_meta_question, trim_history
from rag_search import RagIndex

DEFAULT_URL = "http://localhost:1234/v1"


def detect_model(base_url: str) -> str:
    try:
        resp = requests.get(f"{base_url}/models", timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Nie mogę połączyć się z LM Studio pod {base_url}. "
            "Upewnij się, że lokalny serwer jest uruchomiony "
            "(zakładka 'Developer' -> 'Start Server')."
        ) from e
    ids = [m["id"] for m in resp.json().get("data", [])]
    if not ids:
        raise RuntimeError("LM Studio nie zgłasza żadnego załadowanego modelu.")
    if len(ids) > 1:
        print(f"[INFO] LM Studio ma kilka załadowanych modeli: {ids}. Użyję: {ids[0]!r} (podaj --model, żeby wybrać inny).")
    return ids[0]


def answer(base_url: str, model_id: str, rag: RagIndex, history: list[dict], question: str, top_k: int) -> str:
    if history and looks_like_meta_question(question):
        # Pytanie o przebieg rozmowy -- pomijamy fragmenty RAG (wyszukane
        # od nowa na podstawie samej treści pytania, więc dla takich pytań
        # bywają nietrafione i zaburzają korzystanie z historii; patrz
        # prompt.looks_like_meta_question i PROGRESS.md, krok 10).
        current_turn = {"role": "user", "content": question}
    else:
        results = rag.search(question, top_k=top_k)
        current_turn = {"role": "user", "content": build_user_message(question, results)}
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [current_turn]
    resp = requests.post(
        f"{base_url}/chat/completions",
        json={"model": model_id, "messages": messages, "max_tokens": 500, "temperature": 0.2},
        timeout=300,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    # W historii trzymamy samo pytanie (bez fragmentów RAG) -- fragmenty
    # doklejane są od nowa przy każdej turze, więc trzymanie ich też w
    # historii bardzo szybko przepełnia okno kontekstu (patrz PROGRESS.md,
    # krok 10 -- realny błąd znaleziony w testach: 2 tury = przekroczenie
    # limitu kontekstu LM Studio).
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": text})
    return text


def main():
    parser = argparse.ArgumentParser(description="Asystent kadrowo-płacowy (RAG + model w LM Studio)")
    parser.add_argument("--prompt", type=str, default=None, help="Pojedyncze pytanie zamiast trybu interaktywnego")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="Bazowy URL lokalnego serwera LM Studio")
    parser.add_argument("--model", type=str, default=None, help="ID modelu w LM Studio (domyślnie: auto-wykrycie)")
    parser.add_argument("--top-k", type=int, default=5, help="Liczba fragmentów RAG dołączanych do kontekstu")
    parser.add_argument("--max-turns", type=int, default=6, help="Ile ostatnich par pytanie/odpowiedź zachować w kontekście rozmowy")
    args = parser.parse_args()

    model_id = args.model or detect_model(args.url)
    print(f"[INFO] LM Studio: {args.url}, model={model_id}")

    print("[INFO] Ładowanie indeksu RAG...")
    rag = RagIndex()

    if args.prompt:
        print(answer(args.url, model_id, rag, [], args.prompt, args.top_k))
        return

    print("\nAsystent kadrowo-płacowy (LM Studio) gotowy. Wpisz pytanie (Ctrl+C aby zakończyć, /nowy aby zacząć nowy wątek).\n")
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
        text = answer(args.url, model_id, rag, history, question, args.top_k)
        print("Asystent:", text, "\n")
        history = trim_history(history, args.max_turns)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"[BŁĄD] {e}", file=sys.stderr)
        sys.exit(1)
