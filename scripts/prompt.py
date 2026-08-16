"""
Wspólny prompt systemowy i budowanie kontekstu RAG, używane przez
wszystkie warianty czatu (chat.py / MLX, chat_lmstudio.py / LM Studio,
chat_cuda.py / transformers+peft). Celowo bez ciężkich zależności
(mlx, torch), żeby dało się zaimportować na każdej platformie.
"""

import re

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


MAX_ARTICLE_CHARS = 6000


def build_context(results: list[dict]) -> str:
    parts = []
    for r in results:
        text = r["text"]
        if len(text) > MAX_ARTICLE_CHARS:
            # Niektóre artykuły (np. art. 50 ustawy systemowej, ok. 51k
            # znaków po latach nowelizacji) potrafią same zapełnić okno
            # kontekstu modelu -- patrz PROGRESS.md, krok 10. Ucinamy i
            # odsyłamy do source_url zamiast wstrzykiwać całość.
            text = text[:MAX_ARTICLE_CHARS] + f"\n[...treść artykułu skrócona, pełny tekst: {r['source_url']}...]"
        parts.append(f"### {r['act_title']} -- art. {r['article']}\n{text}")
    return "\n\n".join(parts)


def build_user_message(question: str, results: list[dict]) -> str:
    context = build_context(results)
    return f"Fragmenty aktów prawnych:\n\n{context}\n\n---\n\nPytanie: {question}"


_META_QUESTION_PATTERNS = [
    r"o czym (m[oó]wili[śs]my|rozmawiali[śs]my|by[łl]a mowa)",
    r"co (m[oó]wili[śs]my|napisa[łl]e[śs]|powiedzia[łl]e[śs]|odpowiedzia[łl]e[śs])",
    r"\bprzypomnij\b",
    r"\bpodsumuj\b.*\brozmow",
    r"na (samym )?pocz[ąa]tku.*(rozmowy|pytaniu)",
    r"\bpierwszym pytaniu\b",
    r"wcze[śs]niejszej (odpowiedzi|cz[ęe][śs]ci rozmowy)",
    r"o tym co (m[oó]wili[śs]my|napisa[łl]e[śs]|powiedzia[łl]e[śs])",
    r"zacytuj.*(m[oó]wili[śs]my|wcze[śs]niej|na pocz[ąa]tku)",
]
_META_QUESTION_RE = re.compile("|".join(_META_QUESTION_PATTERNS), re.IGNORECASE)


def looks_like_meta_question(question: str) -> bool:
    """Heurystyka (nie klasyfikator) wykrywająca pytania o przebieg TEJ
    rozmowy (np. "przypomnij", "o czym mówiliśmy") zamiast o nowy fakt
    prawny. Fragmenty RAG są wyszukiwane od nowa na podstawie samej
    treści pytania -- dla pytań meta bywają kompletnie nietrafione i
    eksperymentalnie potrafią zablokować zdolność modelu do korzystania
    z historii rozmowy, zwłaszcza po wcześniejszej odmowie w tej samej
    rozmowie (patrz PROGRESS.md, krok 10). Niewyczerpująca -- pytanie o
    nietypowym sformułowaniu po prostu wraca do domyślnego, sprawdzonego
    zachowania (fragmenty RAG jak zwykle)."""
    return bool(_META_QUESTION_RE.search(question))


def trim_history(history: list[dict], max_turns: int) -> list[dict]:
    """Zachowuje tylko ostatnie `max_turns` par (user, assistant) w historii
    rozmowy, żeby nie przepełnić okna kontekstu modelu -- każda tura dokłada
    do promptu również fragmenty RAG wstrzyknięte w poprzednich pytaniach
    użytkownika, więc historia rośnie szybciej niż liczba wiadomości."""
    max_messages = max_turns * 2
    if len(history) > max_messages:
        del history[: len(history) - max_messages]
    return history
