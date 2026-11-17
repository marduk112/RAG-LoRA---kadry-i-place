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

Można też wgrać własny plik PDF (tekstowy, nie skan) jako dodatkowy
kontekst -- wpisz /plik <ścieżka> w trybie interaktywnym albo podaj
--file przy starcie. Fragmenty z pliku są wyraźnie oznaczane jako
"treść pliku użytkownika", nie mylone z obowiązującym prawem.

Można też zapytać o stan prawny na konkretny dzień w przeszłości (wymaga
indeksu RAG zbudowanego z --include-history, patrz PROGRESS.md Krok 18/19)
-- wpisz /data RRRR-MM-DD w trybie interaktywnym albo podaj --as-of przy
starcie; samo /data (bez daty) wraca do stanu bieżącego.

Użycie:
    python scripts/chat_lmstudio.py                      # tryb interaktywny
    python scripts/chat_lmstudio.py --prompt "pytanie..."
    python scripts/chat_lmstudio.py --model bielik-11b-v3.0-instruct  # jeśli LM Studio ma kilka modeli naraz
    python scripts/chat_lmstudio.py --url http://localhost:1234/v1
    python scripts/chat_lmstudio.py --file umowa.pdf     # z wgranym plikiem od startu
    python scripts/chat_lmstudio.py --as-of 2019-06-01    # stan prawny na dany dzień
    python scripts/chat_lmstudio.py --model qwen/qwen3.8-27b --prompt "pytanie..."   # Qwen: domyślnie
        # --max-tokens 3000 i --reasoning-effort low (model "thinking"; patrz PROGRESS.md, Krok 34)
"""

import argparse
import json
import sys
from datetime import date

import requests

from file_index import SessionFileIndex
from prompt import (
    SYSTEM_PROMPT,
    build_followup_rewrite_messages,
    build_user_message,
    looks_like_meta_question,
    search_with_history,
    trim_history,
)
from rag_search import RagIndex

DEFAULT_URL = "http://localhost:1234/v1"
EMPTY_RETRY_MAX_TOKENS = 2000  # jedno ponowienie krótkiego wywołania ("thinking"), gdy pierwsze zwróciło pusty tekst


def resolve_generation_defaults(model_id: str, max_tokens: int | None, reasoning_effort: str | None) -> tuple[int, str | None]:
    """Modele Qwen w trybie "thinking" z limitem 500 tokenów i bez `reasoning_effort` zwracają PUSTE odpowiedzi
    (rozumowanie zjada cały budżet). Dla modeli Qwen domyślnie 3000 tokenów i reasoning_effort=low; jawnie
    podane argumenty zawsze wygrywają. Pozostałe modele (Bielik): 500 i brak reasoning_effort, jak dotąd.
    Zasada przeniesiona z repozytorium wielodziedzinowego (tam Krok 110; tu Krok 34)."""
    is_qwen = "qwen" in model_id.lower()
    if max_tokens is None:
        max_tokens = 3000 if is_qwen else 500
    if reasoning_effort is None and is_qwen:
        reasoning_effort = "low"
    return max_tokens, reasoning_effort


def _apply_thinking_params(payload: dict, reasoning_effort: str | None, template_kwargs: dict | None) -> dict:
    """Dokłada do zapytania parametry trybu "thinking": `reasoning_effort` (LM Studio/vLLM) oraz przez
    `chat_template_kwargs` (vLLM przyjmuje parametry szablonu tylko tą drogą) i ewentualne dodatkowe
    `template_kwargs` (np. {"enable_thinking": false}); niczego nie zmienia, gdy oba są puste."""
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
        payload["chat_template_kwargs"] = {"reasoning_effort": reasoning_effort}
    if template_kwargs:
        payload["chat_template_kwargs"] = {**payload.get("chat_template_kwargs", {}), **template_kwargs}
    return payload


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


def rewrite_followup(
    base_url: str,
    model_id: str,
    history: list[dict],
    question: str,
    reasoning_effort: str | None = None,
    template_kwargs: dict | None = None,
) -> str:
    """Krótkie wywołanie modelu: pytanie uzupełniające -> samodzielne (patrz prompt.build_followup_rewrite_messages).
    Dla modeli z "myśleniem" (Qwen): 800 tokenów i jedno ponowienie z większym budżetem, gdy wynik jest pusty;
    gdyby i to dało pusty tekst, zostaje oryginalne pytanie (czyści to prompt.clean_rewritten_question)."""
    messages = build_followup_rewrite_messages(history, question)
    payload = {"model": model_id, "messages": messages, "max_tokens": 800 if reasoning_effort else 120, "temperature": 0.0}
    _apply_thinking_params(payload, reasoning_effort, template_kwargs)
    resp = requests.post(f"{base_url}/chat/completions", json=payload, timeout=300)
    resp.raise_for_status()
    text = (resp.json()["choices"][0]["message"].get("content") or "").strip()
    if not text and reasoning_effort is not None:
        payload["max_tokens"] = EMPTY_RETRY_MAX_TOKENS
        resp = requests.post(f"{base_url}/chat/completions", json=payload, timeout=600)
        resp.raise_for_status()
        text = (resp.json()["choices"][0]["message"].get("content") or "").strip()
    return text


def answer(
    base_url: str,
    model_id: str,
    rag: RagIndex,
    file_index: SessionFileIndex,
    history: list[dict],
    question: str,
    top_k: int,
    as_of: str | None = None,
    rewrite_followups: bool = True,
    max_tokens: int = 500,
    reasoning_effort: str | None = None,
    template_kwargs: dict | None = None,
) -> str:
    if history and looks_like_meta_question(question):
        # Pytanie o przebieg rozmowy -- pomijamy fragmenty RAG (wyszukane
        # od nowa na podstawie samej treści pytania, więc dla takich pytań
        # bywają nietrafione i zaburzają korzystanie z historii; patrz
        # prompt.looks_like_meta_question i PROGRESS.md, krok 10).
        current_turn = {"role": "user", "content": question}
    else:
        rewrite = (
            (lambda h, q: rewrite_followup(base_url, model_id, h, q, reasoning_effort, template_kwargs))
            if rewrite_followups
            else None
        )
        results = search_with_history(rag, history, question, top_k, as_of=as_of, rewrite=rewrite)
        file_results = file_index.search(question, top_k=3)
        current_turn = {
            "role": "user",
            "content": build_user_message(question, results, file_results, as_of=as_of),
        }
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [current_turn]
    payload = {"model": model_id, "messages": messages, "max_tokens": max_tokens, "temperature": 0.2}
    _apply_thinking_params(payload, reasoning_effort, template_kwargs)
    resp = requests.post(f"{base_url}/chat/completions", json=payload, timeout=600)
    resp.raise_for_status()
    choice = resp.json()["choices"][0]
    text = (choice["message"].get("content") or "").strip()
    if not text:
        # Model "thinking" zużył cały budżet na rozumowanie (finish_reason=length) -- nie dopisujemy pustej
        # odpowiedzi do historii; mówimy wprost, co zrobić.
        raise RuntimeError(
            "Model zwrócił pustą odpowiedź (prawdopodobnie cały budżet --max-tokens poszedł na rozumowanie, "
            f"finish_reason={choice.get('finish_reason')!r}). Podnieś --max-tokens (np. 3000) i/lub ustaw "
            "--reasoning-effort low."
        )
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
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Liczba fragmentów RAG dołączanych do kontekstu (domyślnie 8 -- podniesione z 5, patrz "
        "PROGRESS.md, Krok 21/22/24: przy 5 fragment z nowo dodanych rozporządzeń o konkretnych kwotach "
        "czasem przegrywał rankingiem z kilkoma tematycznie bliskimi artykułami tej samej ustawy bazowej)",
    )
    parser.add_argument(
        "--no-rewrite-followups",
        action="store_true",
        help="Nie przepisuj pytań uzupełniających (\"a po 15 latach?\") na samodzielne dodatkowym wywołaniem modelu "
        "(PROGRESS.md, Krok 29) -- wraca stary fallback oparty na progu kosinusa",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Limit tokenów odpowiedzi (domyślnie 500; dla modeli Qwen 3000, bo rozumowanie zjada budżet i "
        "odpowiedź wychodzi pusta)",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        default=None,
        help="Dla modeli z trybem \"thinking\" (Qwen): ile rozumować (domyślnie low dla Qwena, brak dla pozostałych)",
    )
    parser.add_argument(
        "--template-kwargs",
        default=None,
        help='JSON z dodatkowymi parametrami szablonu czatu, np. \'{"enable_thinking": false}\' (łączony z --reasoning-effort)',
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

    model_id = args.model or detect_model(args.url)
    max_tokens, reasoning_effort = resolve_generation_defaults(model_id, args.max_tokens, args.reasoning_effort)
    template_kwargs = json.loads(args.template_kwargs) if args.template_kwargs else None
    print(
        f"[INFO] LM Studio: {args.url}, model={model_id}, max_tokens={max_tokens}"
        + (f", reasoning_effort={reasoning_effort}" if reasoning_effort else "")
    )

    print("[INFO] Ładowanie indeksu RAG...")
    rag = RagIndex()
    file_index = SessionFileIndex(model=rag.model)

    if args.file:
        n = file_index.add_pdf(args.file)
        print(f"[INFO] Wgrano plik {args.file} ({n} fragmentów)")

    if args.as_of:
        print(f"[INFO] Stan prawny na dzień: {args.as_of}")

    if args.prompt:
        print(answer(
            args.url, model_id, rag, file_index, [], args.prompt, args.top_k, as_of=args.as_of,
            max_tokens=max_tokens, reasoning_effort=reasoning_effort, template_kwargs=template_kwargs,
        ))
        return

    print("\nAsystent kadrowo-płacowy (LM Studio) gotowy. Wpisz pytanie (Ctrl+C aby zakończyć, /nowy aby zacząć nowy wątek, /plik <ścieżka> aby wgrać PDF, /data RRRR-MM-DD aby zapytać o stan prawny na dany dzień).\n")
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
            # pliku (file_index) -- osobny "tryb" sesji, czyści go
            # wyłącznie jawne /data.
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
        try:
            text = answer(
                args.url, model_id, rag, file_index, history, question, args.top_k, as_of=as_of,
                rewrite_followups=not args.no_rewrite_followups,
                max_tokens=max_tokens, reasoning_effort=reasoning_effort, template_kwargs=template_kwargs,
            )
        except RuntimeError as e:  # np. pusta odpowiedź modelu "thinking" -- nie przerywamy rozmowy
            print(f"[BŁĄD] {e}\n")
            continue
        print("Asystent:", text, "\n")
        history = trim_history(history, args.max_turns)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"[BŁĄD] {e}", file=sys.stderr)
        sys.exit(1)
