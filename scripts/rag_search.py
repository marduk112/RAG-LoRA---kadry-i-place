"""
Wyszukiwanie w indeksie RAG zbudowanym przez build_rag_index.py.

Użycie jako biblioteka:
    from rag_search import RagIndex
    idx = RagIndex()
    results = idx.search("Ile dni urlopu przysługuje po 10 latach pracy?", top_k=5)

Użycie z linii poleceń (do szybkiego testowania samego wyszukiwania,
bez Bielika):
    python scripts/rag_search.py "Ile dni urlopu przysługuje po 10 latach pracy?"
"""

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
INDEX_EMB_PATH = PROCESSED_DIR / "rag_index.npy"
INDEX_META_PATH = PROCESSED_DIR / "rag_index_meta.json"

EMBEDDING_MODEL = "sdadas/mmlw-retrieval-roberta-large"
QUERY_PREFIX = "zapytanie: "  # wymagane przez ten model dla pytań (nie dla fragmentów)


class RagIndex:
    def __init__(self):
        if not INDEX_EMB_PATH.exists():
            raise FileNotFoundError(
                "Brak indeksu RAG. Najpierw uruchom: python scripts/build_rag_index.py"
            )
        self.embeddings = np.load(INDEX_EMB_PATH)  # znormalizowane wektory, shape (N, D)
        self.meta = json.loads(INDEX_META_PATH.read_text(encoding="utf-8"))
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def search(self, query: str, top_k: int = 6) -> list[dict]:
        query_vec = self.model.encode(
            [QUERY_PREFIX + query], normalize_embeddings=True
        )[0]
        # embeddingi są znormalizowane -> iloczyn skalarny = podobieństwo kosinusowe
        scores = self.embeddings @ query_vec

        # Długie artykuły są pocięte na kilka nakładających się fragmentów
        # (patrz build_rag_index.py). Gdybyśmy rankingowali surowe fragmenty,
        # artykuł pocięty na 4 części miałby 4x więcej "szans" na przypadkowo
        # wysoki wynik niż krótki, jednofragmentowy artykuł -- co realnie
        # zniekształca wyniki. Dlatego bierzemy NAJLEPSZY wynik na artykuł.
        best_per_article: dict[tuple, tuple[float, int]] = {}
        for i, score in enumerate(scores):
            chunk = self.meta[i]
            key = (chunk["act_short"], chunk["article"])
            if key not in best_per_article or score > best_per_article[key][0]:
                best_per_article[key] = (float(score), i)

        ranked = sorted(best_per_article.values(), key=lambda x: -x[0])[:top_k]
        results = []
        for score, i in ranked:
            chunk = self.meta[i]
            results.append(
                {
                    "score": score,
                    "act_title": chunk["act_title"],
                    "article": chunk["article"],
                    "eli": chunk["eli"],
                    "text": chunk["full_article_text"],
                    "chunk_text": chunk["chunk_text"],
                    "source_url": chunk["source_url"],
                }
            )
        return results


def main():
    if len(sys.argv) < 2:
        print('Użycie: python scripts/rag_search.py "treść pytania"')
        sys.exit(1)
    query = " ".join(sys.argv[1:])

    idx = RagIndex()
    results = idx.search(query, top_k=5)

    print(f'\nPytanie: "{query}"\n')
    for r in results:
        print(f"[{r['score']:.3f}] {r['act_title']} -- art. {r['article']} ({r['eli']})")
        preview = r["text"][:220].replace("\n", " ")
        print(f"          {preview}...\n")


if __name__ == "__main__":
    main()
