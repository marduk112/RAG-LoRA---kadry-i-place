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
prawnych wyszukane jako potencjalnie pomocny kontekst, a czasem też \
fragmenty pliku wgranego przez użytkownika (np. jego umowy o pracę, \
regulaminu). Zasady:
1. Odpowiadaj WYŁĄCZNIE na podstawie dostarczonych fragmentów -- nie \
korzystaj z własnej wiedzy o konkretnych liczbach, kwotach czy terminach.
2. Jeśli dostarczone fragmenty nie zawierają odpowiedzi na pytanie, wprost \
napisz, że nie znalazłeś tego w dostępnych materiałach, i nie zgaduj.
3. Cytuj numer artykułu i nazwę aktu, na którym opierasz odpowiedź. \
Fragmenty z wgranego pliku użytkownika wyraźnie oznaczaj jako treść \
tego pliku, a NIE jako obowiązujące prawo -- to różne źródła.
4. Nie jesteś substytutem porady prawnej ani księgowej -- w sprawach \
spornych zasugeruj konsultację ze specjalistą.
5. Jeśli zapytanie dotyczy konkretnej daty w przeszłości (zobaczysz to w \
notatce "Data, na którą ma obowiązywać odpowiedź" i/lub w adnotacjach \
"stan prawny" przy fragmentach), odpowiadaj TĄ wersją przepisu i wyraźnie \
zaznacz w odpowiedzi, że dotyczy ona tej konkretnej daty. Jeśli żaden \
dostarczony fragment nie obejmuje tej daty, powiedz to wprost -- nie \
zgaduj i nie podawaj w zamian stanu bieżącego."""


MAX_ARTICLE_CHARS = 6000


LOW_CONFIDENCE_SCORE_THRESHOLD = 0.75


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
        header = f"### {r['act_title']} -- art. {r['article']}"
        # Adnotacja "stan prawny" -- tylko gdy fragment faktycznie niesie
        # wymiar czasowy (indeks zbudowany z --include-history, patrz
        # PROGRESS.md Krok 19). Gdy oba pola None (dotychczasowy, domyślny
        # przypadek), nagłówek zostaje DOKŁADNIE taki jak dziś -- celowo,
        # żeby nie zmieniać formatu promptu dla nikogo, kto tej funkcji
        # nie używa (`as_of` samego `build_context` nie jest tu nawet
        # potrzebne do tej decyzji -- to obecność pól w wyniku decyduje,
        # nie to, czy wołający w ogóle podał `as_of`).
        if r.get("valid_from") or r.get("valid_to"):
            valid_from = r.get("valid_from") or "..."
            valid_to = r.get("valid_to") or "nadal"
            header += f" (stan prawny: {valid_from} – {valid_to})"
        parts.append(f"{header}\n{text}")
    return "\n\n".join(parts)


def build_file_context(file_results: list[dict]) -> str:
    parts = []
    for r in file_results:
        parts.append(f"### Plik użytkownika: {r['source_file']}\n{r['text']}")
    return "\n\n".join(parts)


def build_user_message(
    question: str, results: list[dict], file_results: list[dict] | None = None, as_of: str | None = None
) -> str:
    parts = []
    if as_of:
        # Notatka niezależna od adnotacji "stan prawny" na pojedynczych
        # fragmentach w build_context() -- ważne zwłaszcza gdy ŻADEN
        # dostarczony fragment jakiejś ustawy nie pokrywa tej daty (taki
        # fragment nie ma czego adnotować), a model i tak musi wiedzieć,
        # że konkretna data była proszona, żeby mógł to wprost powiedzieć
        # zamiast po cichu odpowiedzieć stanem bieżącym (patrz punkt 5
        # SYSTEM_PROMPT).
        parts.append(f"[Data, na którą ma obowiązywać odpowiedź: {as_of}]")
    if not results or results[0]["score"] < LOW_CONFIDENCE_SCORE_THRESHOLD:
        # Jawny sygnał zamiast liczenia na to, że model sam zauważy słabe
        # dopasowanie -- testy end-to-end pokazały, że nawet z regułą 1/2
        # SYSTEM_PROMPT ("nie korzystaj z własnej wiedzy o liczbach") model
        # potrafi po cichu dopowiedzieć konkretną wartość z pamięci, gdy
        # żaden fragment tematycznie nie pasuje (patrz PROGRESS.md, temat
        # spoza pobranych ustaw). NIE pomaga to w przypadku, gdy fragment
        # jest trafiony tematycznie, ale mimo to nie zawiera szukanej
        # liczby (np. artykuł delegujący do nieobecnego w indeksie
        # rozporządzenia) -- na to nie ma tu obrony, to wymagałoby
        # sprawdzania treści, nie samego score'u dopasowania.
        best_score = results[0]["score"] if results else None
        score_note = f" (najlepszy wynik dopasowania: {best_score:.2f})" if best_score is not None else " (brak wyników)"
        parts.append(
            f"[Uwaga: żaden odnaleziony fragment nie jest wysoko dopasowany do tego pytania{score_note} "
            "-- fragmenty poniżej prawdopodobnie NIE zawierają odpowiedzi. Jeśli faktycznie jej nie "
            "zawierają, powiedz to wprost zamiast zgadywać (patrz zasada 2 wyżej).]"
        )
    parts.append(f"Fragmenty aktów prawnych:\n\n{build_context(results)}")
    if file_results:
        parts.append(f"Fragmenty z wgranego przez użytkownika pliku:\n\n{build_file_context(file_results)}")
    parts.append(f"---\n\nPytanie: {question}")
    return "\n\n".join(parts)


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


ELLIPTICAL_SCORE_THRESHOLD = 0.75


MAX_LOOKBACK_QUESTIONS = 3


def search_with_history(
    rag, history: list[dict], question: str, top_k: int, as_of: str | None = None
) -> list[dict]:
    """RAG search z fallbackiem dla pytań eliptycznych typu "a po 15
    latach?", które same w sobie nie mają wystarczających słów
    kluczowych do trafnego wyszukania (np. brak słowa "urlop") -- w
    testach takie pytanie w ogóle nie trafiało na właściwy artykuł, a
    model bez groundingu z RAG "dopowiadał" nieistniejące szczegóły z
    pamięci historii (patrz PROGRESS.md, krok 10/11).

    Jeśli najlepszy wynik dla samego pytania jest słaby
    (< ELLIPTICAL_SCORE_THRESHOLD) i jest już jakaś historia rozmowy,
    próbujemy też wyszukań z doklejonym KAŻDYM z ostatnich
    MAX_LOOKBACK_QUESTIONS pytań użytkownika osobno (nie tylko
    bezpośrednio poprzednim -- ono może być z zupełnie innego,
    zakończonego odmową wątku, np. "ile wynosi minimalne wynagrodzenie"
    tuż przed "a po 15 latach?", podczas gdy faktycznie powiązane
    pytanie o urlop było dwie tury wcześniej) i zostajemy przy tym
    wariancie, który dał najlepszy najlepszy wynik. Nie robimy tego
    bezwarunkowo ani zbiorczo (całą historią naraz) -- doklejanie
    niezwiązanego pytania do zapytania psuje trafność (sprawdzone
    empirycznie), dlatego każde poprzednie pytanie jest próbowane
    osobno."""
    results = rag.search(question, top_k=top_k, as_of=as_of)
    if not history or not results or results[0]["score"] >= ELLIPTICAL_SCORE_THRESHOLD:
        return results
    best_results, best_score = results, results[0]["score"]
    prior_questions = [m["content"] for m in history if m["role"] == "user"][-MAX_LOOKBACK_QUESTIONS:]
    for prior in prior_questions:
        candidate = rag.search(f"{prior} {question}", top_k=top_k, as_of=as_of)
        if candidate and candidate[0]["score"] > best_score:
            best_results, best_score = candidate, candidate[0]["score"]
    return best_results


def trim_history(history: list[dict], max_turns: int) -> list[dict]:
    """Zachowuje tylko ostatnie `max_turns` par (user, assistant) w historii
    rozmowy, żeby nie przepełnić okna kontekstu modelu -- każda tura dokłada
    do promptu również fragmenty RAG wstrzyknięte w poprzednich pytaniach
    użytkownika, więc historia rośnie szybciej niż liczba wiadomości."""
    max_messages = max_turns * 2
    if len(history) > max_messages:
        del history[: len(history) - max_messages]
    return history
