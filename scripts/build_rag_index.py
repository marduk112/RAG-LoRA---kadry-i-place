"""
Buduje indeks RAG (Retrieval-Augmented Generation) na fragmentach ustaw.

Co to jest RAG, w skrócie:
  Model językowy (Bielik) sam z siebie NIE zna treści konkretnych,
  polskich ustaw na pamięć w sposób wiarygodny -- a nawet gdyby był
  fine-tunowany na ich tekście, ustawy się zmieniają, więc taka wiedza
  szybko by się zdezaktualizowała. RAG obchodzi ten problem inaczej:
  zamiast "wkuwać" przepisy w wagi modelu, w momencie zadawania pytania
  WYSZUKUJEMY najbardziej pasujące fragmenty ustaw i wklejamy je do
  promptu jako kontekst. Model tylko czyta ten kontekst i formułuje
  odpowiedź na jego podstawie -- tak jak człowiek, który dostaje
  podkreślony fragment ustawy do przeczytania, zamiast cytować z pamięci.

Jak działa wyszukiwanie (semantyczne, nie po słowach kluczowych):
  1. Każdy fragment ustawy zamieniamy na wektor liczb (embedding) za
     pomocą modelu embeddingowego -- wektor reprezentuje "znaczenie"
     tekstu w wielowymiarowej przestrzeni. Podobne znaczeniowo teksty
     mają wektory blisko siebie.
  2. Pytanie użytkownika zamieniamy na wektor tym samym modelem.
  3. Liczymy podobieństwo kosinusowe pytania do wszystkich fragmentów
     i bierzemy np. 5 najbardziej podobnych.
  To działa nawet, gdy pytanie nie zawiera dokładnie tych samych słów
  co ustawa (np. "ile urlopu mi się należy" trafi w art. 154 KP, mimo
  że artykuł nie zawiera słowa "należy").

Ten skrypt tylko BUDUJE indeks (liczy i zapisuje embeddingi).
Samo wyszukiwanie jest w scripts/rag_search.py.

Uruchomienie:
    python scripts/build_rag_index.py
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
ARTICLES_PATH = PROCESSED_DIR / "all_articles.json"
INDEX_EMB_PATH = PROCESSED_DIR / "rag_index.npy"
INDEX_META_PATH = PROCESSED_DIR / "rag_index_meta.json"

EMBEDDING_MODEL = "sdadas/mmlw-retrieval-roberta-large"

# Niektóre artykuły (np. w ustawie o systemie ubezpieczeń społecznych)
# mają po kilka tysięcy znaków -- za dużo, żeby sensownie zmieścić się
# jako jeden precyzyjny wektor i za dużo, żeby wstrzykiwać w całości do
# promptu. Długie artykuły tniemy dodatkowo na nakładające się okna.
MAX_CHUNK_CHARS = 900
OVERLAP_CHARS = 150


def split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    words = text.split(" ")
    chunks = []
    current: list[str] = []
    current_len = 0
    for word in words:
        current.append(word)
        current_len += len(word) + 1
        if current_len >= max_chars:
            chunks.append(" ".join(current))
            # cofamy się o ~overlap znaków, żeby zachować kontekst na styku okien
            overlap_words: list[str] = []
            overlap_len = 0
            for w in reversed(current):
                overlap_len += len(w) + 1
                overlap_words.insert(0, w)
                if overlap_len >= overlap:
                    break
            current = overlap_words
            current_len = sum(len(w) + 1 for w in current)
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_chunks(articles: list[dict]) -> list[dict]:
    chunks = []
    for art in articles:
        pieces = split_long_text(art["text"], MAX_CHUNK_CHARS, OVERLAP_CHARS)
        for i, piece in enumerate(pieces):
            chunk_id = art["id"] if len(pieces) == 1 else f"{art['id']}_part{i + 1}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "act_short": art["act_short"],
                    "act_title": art["act_title"],
                    "eli": art["eli"],
                    "article": art["article"],
                    "source_url": art["source_url"],
                    "chunk_text": piece,
                    "full_article_text": art["text"],
                }
            )
    return chunks


def main():
    print(f"Wczytuję fragmenty z {ARTICLES_PATH} ...")
    articles = json.loads(ARTICLES_PATH.read_text(encoding="utf-8"))
    print(f"  {len(articles)} artykułów")

    chunks = build_chunks(articles)
    print(f"Po podziale długich artykułów: {len(chunks)} fragmentów do embedowania")

    print(f"Ładuję model embeddingowy: {EMBEDDING_MODEL} (pierwsze uruchomienie pobierze go z HF)...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["chunk_text"] for c in chunks]
    print("Liczę embeddingi (bez prefiksu -- to są 'dokumenty', nie zapytania)...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # ułatwia potem liczenie podobieństwa kosinusowego
    )

    np.save(INDEX_EMB_PATH, embeddings.astype(np.float32))
    INDEX_META_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nZapisano:")
    print(f"  {INDEX_EMB_PATH}  (macierz embeddingów {embeddings.shape})")
    print(f"  {INDEX_META_PATH}  (metadane fragmentów)")


if __name__ == "__main__":
    main()
