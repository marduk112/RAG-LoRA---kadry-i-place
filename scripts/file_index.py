"""
Tymczasowy, sesyjny indeks RAG dla pojedynczego wgranego pliku PDF --
na razie tylko PDF-y tekstowe (tzn. tekst da się wyciągnąć wprost z
pliku), nie skany/obrazy (OCR poza obecnym zakresem).

W przeciwieństwie do stałej bazy ustaw (rag_search.RagIndex), ten
indeks żyje tylko w pamięci na czas jednej sesji czatu i NIE jest
zapisywany na dysk -- celowo, żeby nie mieszać dokumentów wgrywanych
przez użytkownika (np. własnej umowy o pracę) z kuratorowaną,
zweryfikowaną bazą aktów prawnych (`data/processed/`).

Użycie:
    file_index = SessionFileIndex(model=rag.model)  # dzieli model embeddingowy z RagIndex
    liczba_fragmentow = file_index.add_pdf("umowa.pdf")
    results = file_index.search("okres wypowiedzenia", top_k=3)
"""

from pathlib import Path

import numpy as np
from pypdf import PdfReader

from build_rag_index import MAX_CHUNK_CHARS, OVERLAP_CHARS, split_long_text
from rag_search import QUERY_PREFIX


class SessionFileIndex:
    def __init__(self, model):
        self.model = model  # dzielony z RagIndex -- ten sam model embeddingowy, nie ładujemy drugi raz
        self.chunks: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self.file_name: str | None = None

    def add_pdf(self, path: str) -> int:
        """Wczytuje PDF, dzieli na fragmenty i liczy embeddingi. Zastępuje
        poprzednio wgrany plik (obsługujemy jeden plik na raz -- prosty
        zakres na początek, patrz PROGRESS.md)."""
        reader = PdfReader(path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
        if not text:
            raise ValueError(
                f"Nie udało się wyciągnąć tekstu z {path} -- prawdopodobnie skan/obraz, "
                "nie tekstowy PDF (obsługujemy na razie tylko PDF-y z tekstem, nie OCR)."
            )

        pieces = split_long_text(text, MAX_CHUNK_CHARS, OVERLAP_CHARS)
        self.file_name = Path(path).name
        self.chunks = [{"source_file": self.file_name, "chunk_text": piece} for piece in pieces]
        self.embeddings = np.asarray(
            self.model.encode([c["chunk_text"] for c in self.chunks], normalize_embeddings=True)
        )
        return len(pieces)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        query_vec = self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_vec
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [
            {
                "score": float(scores[i]),
                "source_file": self.chunks[i]["source_file"],
                "text": self.chunks[i]["chunk_text"],
            }
            for i in ranked
        ]
