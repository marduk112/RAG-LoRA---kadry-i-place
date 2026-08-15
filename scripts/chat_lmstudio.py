"""
Asystent kadrowo-płacowy: RAG (rag_search.RagIndex) + model załadowany w
LM Studio (lokalny serwer zgodny z API OpenAI), zamiast lokalnego MLX.

Przydatne, gdy masz już w LM Studio załadowany model (np. Bielik w
formacie GGUF) i nie chcesz go dublować w MLX. Ponownie wykorzystuje
dokładnie ten sam prompt systemowy i sposób budowania kontekstu z RAG
co scripts/chat.py -- różni się tylko backendem generacji.

Uwaga: LoRA wytrenowane w scripts/chat.py (adapters/*) NIE działa tutaj
-- to wagi w formacie MLX, a LM Studio ładuje osobny plik GGUF. Ten
skrypt daje sam RAG + Twój model z LM Studio, bez dotrenowanego stylu.

Wymaga uruchomionego serwera lokalnego w LM Studio: zakładka
"Developer" -> "Start Server" (domyślnie http://localhost:1234).

Użycie:
    python scripts/chat_lmstudio.py                      # tryb interaktywny
    python scripts/chat_lmstudio.py --prompt "pytanie..."
    python scripts/chat_lmstudio.py --model bielik-11b-v3.0-instruct  # jeśli LM Studio ma kilka modeli naraz
    python scripts/chat_lmstudio.py --url http://localhost:1234/v1
"""

import argparse
import sys

import requests

from chat import SYSTEM_PROMPT, build_context
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


def answer(base_url: str, model_id: str, rag: RagIndex, question: str, top_k: int) -> str:
    results = rag.search(question, top_k=top_k)
    context = build_context(results)
    user_content = f"Fragmenty aktów prawnych:\n\n{context}\n\n---\n\nPytanie: {question}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    resp = requests.post(
        f"{base_url}/chat/completions",
        json={"model": model_id, "messages": messages, "max_tokens": 500, "temperature": 0.2},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description="Asystent kadrowo-płacowy (RAG + model w LM Studio)")
    parser.add_argument("--prompt", type=str, default=None, help="Pojedyncze pytanie zamiast trybu interaktywnego")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="Bazowy URL lokalnego serwera LM Studio")
    parser.add_argument("--model", type=str, default=None, help="ID modelu w LM Studio (domyślnie: auto-wykrycie)")
    parser.add_argument("--top-k", type=int, default=5, help="Liczba fragmentów RAG dołączanych do kontekstu")
    args = parser.parse_args()

    model_id = args.model or detect_model(args.url)
    print(f"[INFO] LM Studio: {args.url}, model={model_id}")

    print("[INFO] Ładowanie indeksu RAG...")
    rag = RagIndex()

    if args.prompt:
        print(answer(args.url, model_id, rag, args.prompt, args.top_k))
        return

    print("\nAsystent kadrowo-płacowy (LM Studio) gotowy. Wpisz pytanie (Ctrl+C aby zakończyć).\n")
    while True:
        try:
            question = input("Ty: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        text = answer(args.url, model_id, rag, question, args.top_k)
        print("Asystent:", text, "\n")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"[BŁĄD] {e}", file=sys.stderr)
        sys.exit(1)
