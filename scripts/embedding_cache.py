"""
Przyrostowe liczenie embeddingów fragmentów (dokumentów) z trwałym cache'em
i strażnikiem pamięci -- używane przez `build_rag_index.py`.

Dlaczego to bezstratne: embedding fragmentu zależy WYŁĄCZNIE od jego tekstu
i modelu (bez globalnego stanu typu IDF; każdy wektor normalizowany osobno),
więc wektor niezmienionego fragmentu wzięty z cache'u jest tym samym
wektorem, który dałoby ponowne liczenie (z dokładnością do szumu
numerycznego GPU ~1e-6, zależnego od składu paczki -- ten sam szum ma pełna
przebudowa). Klucz = (model + tryb, sha256 tekstu): zmiana modelu, trybu
(prefiks/normalizacja) albo sposobu cięcia na fragmenty (inne teksty)
automatycznie omija stare wpisy. Źródło: projekt "RAG-LoRA - prawo polskie",
PROGRESS.md Krok 63.

Pamięć: fragmenty liczone porcjami (`SLICE_SIZE`), po każdej porcji zapis do
SQLite (commit -- przerwany przebieg wznawia się od miejsca przerwania) i
zwolnienie bufora GPU/MPS (bez tego proces rósł skokowo -- projekt "RAG-LoRA - prawo polskie",
PROGRESS.md Krok 57). Przed każdą porcją sprawdzana jest WOLNA pamięć: przy małej zapasie
mniejsza paczka, przy krytycznie małym -- czekanie, a potem czytelne
przerwanie (np. gdy w międzyczasie załadowano Qwena w LM Studio; ta
maszyna już się zawieszała pod presją pamięci).
"""

import hashlib
import sqlite3
import time
from pathlib import Path

import numpy as np

SLICE_SIZE = 512  # fragmentów na porcję między zapisami cache'u i zwolnieniem bufora GPU
DEFAULT_BATCH_SIZE = 32  # jak dotychczas w build_rag_index.py
LOW_MEMORY_BATCH_SIZE = 8
LOW_MEMORY_GB = 8.0  # poniżej -- mniejsze paczki
MIN_FREE_MEMORY_GB = 3.0  # poniżej -- czekamy, potem przerywamy
MEMORY_WAIT_SECONDS = 30
MEMORY_WAIT_ATTEMPTS = 6


def available_ram_gb() -> float | None:
    """Wolna pamięć w GB (`psutil`, wieloplatformowo) albo None, gdy nie da się
    jej odczytać (wtedy strażnik nie ogranicza -- zachowanie jak dotychczas)."""
    try:
        import psutil
    except ImportError:
        return None
    return psutil.virtual_memory().available / 2**30


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingCache:
    """Trwały cache wektorów w SQLite (biblioteka standardowa, zapis
    przyrostowy, bez wczytywania całości do pamięci)."""

    def __init__(self, path: Path, model_key: str):
        self.model_key = model_key
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "model_key TEXT NOT NULL, text_sha256 TEXT NOT NULL, dim INTEGER NOT NULL, vec BLOB NOT NULL, "
            "PRIMARY KEY (model_key, text_sha256))"
        )

    def get_many(self, hashes: list[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        for i in range(0, len(hashes), 500):  # limit parametrów SQLite
            part = hashes[i:i + 500]
            rows = self.conn.execute(
                f"SELECT text_sha256, vec FROM embeddings WHERE model_key = ? AND text_sha256 IN "
                f"({','.join('?' * len(part))})",
                [self.model_key, *part],
            )
            for h, blob in rows:
                found[h] = np.frombuffer(blob, dtype=np.float32)
        return found

    def put_many(self, items: list[tuple[str, np.ndarray]]) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO embeddings (model_key, text_sha256, dim, vec) VALUES (?, ?, ?, ?)",
            [(self.model_key, h, int(v.shape[0]), v.astype(np.float32).tobytes()) for h, v in items],
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


def _batch_size_for_free_memory() -> int:
    """Rozmiar paczki dobrany do wolnej pamięci; przy krytycznie małej --
    czeka (może zwolni się sama), a potem przerywa z czytelnym komunikatem.
    Już policzone wektory są wtedy w cache'u, więc ponowne uruchomienie
    zaczyna od miejsca przerwania."""
    for attempt in range(MEMORY_WAIT_ATTEMPTS + 1):
        free = available_ram_gb()
        if free is None or free >= LOW_MEMORY_GB:
            return DEFAULT_BATCH_SIZE
        if free >= MIN_FREE_MEMORY_GB:
            return LOW_MEMORY_BATCH_SIZE
        if attempt < MEMORY_WAIT_ATTEMPTS:
            print(
                f"[UWAGA] tylko {free:.1f} GB wolnej pamięci (< {MIN_FREE_MEMORY_GB} GB) -- "
                f"czekam {MEMORY_WAIT_SECONDS} s ({attempt + 1}/{MEMORY_WAIT_ATTEMPTS})...",
                flush=True,
            )
            time.sleep(MEMORY_WAIT_SECONDS)
    raise SystemExit(
        f"Przerwano: za mało wolnej pamięci ({free:.1f} GB < {MIN_FREE_MEMORY_GB} GB). Zwolnij pamięć "
        "(np. wyładuj model w LM Studio) i uruchom ponownie -- policzone dotąd wektory są w cache'u."
    )


def encode_with_cache(model, texts: list[str], cache: EmbeddingCache | None, release_cache_fn=None) -> np.ndarray:
    """Znormalizowane embeddingi `texts` (bez prefiksu -- to "dokumenty"),
    z cache'u tam, gdzie tekst już był liczony. `cache=None` -- licz wszystko
    (i niczego nie zapisuj). `release_cache_fn` -- zwolnienie bufora GPU po
    każdej porcji (rag_search.release_accelerator_cache)."""
    hashes = [text_hash(t) for t in texts]
    cached = cache.get_many(sorted(set(hashes))) if cache is not None else {}
    # Każdy brakujący tekst liczony raz, nawet jeśli powtarza się w wejściu --
    # np. artykuł niezmieniony w kilku kolejnych wersjach historycznych.
    first_index: dict[str, int] = {}
    for i, h in enumerate(hashes):
        if h not in cached and h not in first_index:
            first_index[h] = i
    todo = list(first_index.values())
    print(
        f"Embeddingi: {len(texts)} fragmentów, {len(set(hashes))} unikalnych tekstów -- "
        f"{len(cached)} z cache'u, {len(todo)} do policzenia.",
        flush=True,
    )

    computed: dict[str, np.ndarray] = {}
    for start in range(0, len(todo), SLICE_SIZE):
        part = todo[start:start + SLICE_SIZE]
        batch_size = _batch_size_for_free_memory()
        vectors = model.encode(
            [texts[i] for i in part],
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # ułatwia potem liczenie podobieństwa kosinusowego
        ).astype(np.float32)
        items = [(hashes[i], v) for i, v in zip(part, vectors)]
        computed.update(items)
        if cache is not None:
            cache.put_many(items)
        if release_cache_fn is not None:
            release_cache_fn()
        done = min(start + SLICE_SIZE, len(todo))
        print(f"  policzono {done}/{len(todo)} (paczka {batch_size})", flush=True)

    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    dim = next(iter({**cached, **computed}.values())).shape[0]
    out = np.empty((len(texts), dim), dtype=np.float32)
    for i, h in enumerate(hashes):
        out[i] = computed[h] if h in computed else cached[h]
    return out
