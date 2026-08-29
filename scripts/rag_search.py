"""
Wyszukiwanie w indeksie RAG zbudowanym przez build_rag_index.py.

Użycie jako biblioteka:
    from rag_search import RagIndex
    idx = RagIndex()
    results = idx.search("Ile dni urlopu przysługuje po 10 latach pracy?", top_k=8)

Użycie z linii poleceń (do szybkiego testowania samego wyszukiwania,
bez Bielika):
    python scripts/rag_search.py "Ile dni urlopu przysługuje po 10 latach pracy?"
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
INDEX_EMB_PATH = PROCESSED_DIR / "rag_index.npy"
INDEX_META_PATH = PROCESSED_DIR / "rag_index_meta.json"

EMBEDDING_MODEL = "sdadas/mmlw-retrieval-roberta-large"
QUERY_PREFIX = "zapytanie: "  # wymagane przez ten model dla pytań (nie dla fragmentów)


def _version_covers(chunk: dict, as_of: str) -> bool:
    """Czy dany chunk (wersja artykułu) obowiązywał na dzień `as_of`
    (ISO "RRRR-MM-DD")? `valid_from`/`valid_to` to opcjonalne pola z
    download_acts_history.py/build_rag_index.py --include-history --
    `None` oznacza "brak znanej granicy" w tym kierunku (nie "brak
    pokrycia"), więc chunki sprzed tej funkcji (bez tych pól w ogóle,
    `.get(...)` zwraca None) zawsze przechodzą -- stąd filtrowanie jest
    no-opem na danych sprzed tej funkcji. `valid_to` jest EXCLUSIVE,
    zgodnie z semantyką pola `expirationDate` w ELI API (potwierdzone w
    PROGRESS.md, Krok 18)."""
    valid_from = chunk.get("valid_from")
    if valid_from is not None and as_of < valid_from:
        return False
    valid_to = chunk.get("valid_to")
    if valid_to is not None and as_of >= valid_to:
        return False
    return True


def rank_chunks(
    embeddings: np.ndarray, meta: list[dict], query_vec: np.ndarray, top_k: int, as_of: str | None
) -> list[dict]:
    """Czysta logika rankingu -- bez modelu embeddingowego, bez I/O, w pełni
    testowalna (patrz tests/test_rag_search.py). Wydzielona z RagIndex.search().

    WAŻNE: `as_of=None` NIE oznacza "pomiń filtrowanie" -- oznacza "filtruj
    na DZIŚ" (`effective_as_of = as_of or dzisiejsza data ISO`). Bez tego,
    po zbudowaniu indeksu z --include-history, przestarzała wersja
    historyczna mogłaby po cichu wygrać rankingiem z bieżącą wersją tego
    samego artykułu przy zwykłym pytaniu bez podanej daty -- cicho zwrócone
    złe prawo. To jest prowizorycznym no-opem na danych sprzed tej funkcji:
    każdy chunk z dotychczasowego pipeline'u ma valid_from=valid_to=None,
    więc `_version_covers` zwraca True niezależnie od tego, czym jest
    "dziś" -- wynik bajtowo identyczny z zachowaniem sprzed tej zmiany.

    Filtrowanie (`_version_covers`) następuje PRZED grupowaniem po
    `(act_short, article)` -- klucz grupowania celowo NIE zawiera wymiaru
    wersji, bo dla danego `as_of` kwalifikuje się co najwyżej jedna wersja
    danego artykułu (okna ważności różnych wersji tego samego artykułu się
    nie pokrywają -- to filtrowanie właśnie to gwarantuje)."""
    effective_as_of = as_of or date.today().isoformat()

    # Długie artykuły są pocięte na kilka nakładających się fragmentów
    # (patrz build_rag_index.py). Gdybyśmy rankingowali surowe fragmenty,
    # artykuł pocięty na 4 części miałby 4x więcej "szans" na przypadkowo
    # wysoki wynik niż krótki, jednofragmentowy artykuł -- co realnie
    # zniekształca wyniki. Dlatego bierzemy NAJLEPSZY wynik na artykuł
    # (tylko spośród chunków obowiązujących na `effective_as_of`).
    best_per_article: dict[tuple, tuple[float, int]] = {}
    for i, chunk in enumerate(meta):
        if not _version_covers(chunk, effective_as_of):
            continue
        score = float(embeddings[i] @ query_vec)
        key = (chunk["act_short"], chunk["article"])
        if key not in best_per_article or score > best_per_article[key][0]:
            best_per_article[key] = (score, i)

    ranked = sorted(best_per_article.values(), key=lambda x: -x[0])[:top_k]
    results = []
    for score, i in ranked:
        chunk = meta[i]
        results.append(
            {
                "score": score,
                "act_short": chunk["act_short"],
                "act_title": chunk["act_title"],
                "article": chunk["article"],
                "eli": chunk["eli"],
                "text": chunk["full_article_text"],
                "chunk_text": chunk["chunk_text"],
                "source_url": chunk["source_url"],
                "valid_from": chunk.get("valid_from"),
                "valid_to": chunk.get("valid_to"),
                "announcement_eli": chunk.get("announcement_eli"),
            }
        )
    return results


class RagIndex:
    def __init__(self):
        if not INDEX_EMB_PATH.exists():
            raise FileNotFoundError(
                "Brak indeksu RAG. Najpierw uruchom: python scripts/build_rag_index.py"
            )
        self.embeddings = np.load(INDEX_EMB_PATH)  # znormalizowane wektory, shape (N, D)
        self.meta = json.loads(INDEX_META_PATH.read_text(encoding="utf-8"))
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def search(self, query: str, top_k: int = 8, as_of: str | None = None) -> list[dict]:
        query_vec = self.model.encode(
            [QUERY_PREFIX + query], normalize_embeddings=True
        )[0]
        # embeddingi są znormalizowane -> iloczyn skalarny = podobieństwo kosinusowe
        return rank_chunks(self.embeddings, self.meta, query_vec, top_k, as_of)


def main():
    parser = argparse.ArgumentParser(description="Wyszukiwanie w indeksie RAG (bez odpalania modelu językowego).")
    parser.add_argument("query", nargs="+", help="Treść pytania")
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        metavar="RRRR-MM-DD",
        help="Szukaj wersji przepisów obowiązującej na ten dzień (wymaga indeksu "
        "zbudowanego z build_rag_index.py --include-history, patrz PROGRESS.md Krok 18). "
        "Domyślnie: stan bieżący.",
    )
    args = parser.parse_args()
    query = " ".join(args.query)

    idx = RagIndex()
    results = idx.search(query, top_k=8, as_of=args.as_of)

    as_of_label = f" (stan na {args.as_of})" if args.as_of else ""
    print(f'\nPytanie: "{query}"{as_of_label}\n')
    for r in results:
        version_label = ""
        if r.get("valid_from") or r.get("valid_to"):
            version_label = f" [stan: {r.get('valid_from') or '...'} -- {r.get('valid_to') or 'nadal'}]"
        print(f"[{r['score']:.3f}] {r['act_title']} -- art. {r['article']} ({r['eli']}){version_label}")
        preview = r["text"][:220].replace("\n", " ")
        print(f"          {preview}...\n")


if __name__ == "__main__":
    main()
