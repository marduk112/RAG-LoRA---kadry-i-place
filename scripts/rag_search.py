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
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
INDEX_EMB_PATH = PROCESSED_DIR / "rag_index.npy"
INDEX_META_PATH = PROCESSED_DIR / "rag_index_meta.json"

EMBEDDING_MODEL = "sdadas/mmlw-retrieval-roberta-large"
QUERY_PREFIX = "zapytanie: "  # wymagane przez ten model dla pytań (nie dla fragmentów)


# Drugi etap wyszukiwania: cross-encoder reranker (ten sam autor co model
# embeddingowy -- sdadas -- więc spójny wybór). Podobieństwo kosinusowe z
# bi-encodera (mmlw-retrieval-roberta-large) dobrze łapie TEMATYCZNE
# podobieństwo, ale przy większym/bardziej zróżnicowanym korpusie potrafi
# przegrać ogólny, trafny przepis z niszowym, leksykalnie bliższym, ale
# merytorycznie gorszym trafieniem (potwierdzone empirycznie: pytanie o
# korektę deklaracji VAT nie wyciągało w top-8 art. 81 Ordynacji podatkowej
# -- ogólnej reguły korekty deklaracji -- bo przegrywał z niszowymi
# artykułami ustawy o VAT o procedurze unijnej OSS, mimo że dosłownie
# zawierają frazę "deklaracja VAT"). Reranker ocenia parę (zapytanie,
# PEŁNY tekst artykułu) precyzyjniej niż gołe podobieństwo wektorów,
# kosztem policzenia tego dla większej puli kandydatów niż finalny top_k.
RERANKER_MODEL = "sdadas/polish-reranker-base-ranknet"
RERANK_CANDIDATE_POOL = 30  # ilu kandydatów z etapu kosinusowego trafia do rerankingu
RERANK_TEXT_CHARS = 1500  # obcięcie tekstu artykułu przed rerankingiem (cross-encodery mają ograniczoną długość wejścia)


def apply_rerank(queries: str | list[str], results: list[dict], cross_encoder: CrossEncoder) -> list[dict]:
    """Dogrywa wynik cross-encodera (pole 'rerank_score') do wyników z
    rank_chunks() i sortuje malejąco po nim -- zastępuje kolejność z
    czystego podobieństwa kosinusowego ('score'), które zostaje w wyniku
    tylko jako dodatkowa informacja. Sortowanie samo w sobie jest czyste
    (testowalne bez prawdziwego modelu -- podaj fałszywy cross_encoder z
    ustalonym .predict()); ładowanie modelu i liczenie embeddingów to
    jedyna nieczysta część, wołana tylko przez RagIndex.search().

    `queries` może być pojedynczym stringiem (dotychczasowe zachowanie)
    albo LISTĄ wariantów tego samego pytania (np. oryginalne pytanie
    potoczne + jego sformalizowane rozszerzenie, patrz `extra_queries` w
    RagIndex.search()) -- każdy kandydat dostaje wtedy NAJWYŻSZY wynik
    spośród wszystkich wariantów, nie tylko pierwszego. Sam bi-encoder
    cierpi na lukę słownikową między potocznym pytaniem a formalnym
    językiem ustawy (patrz build_rag_index/rag_search), ale okazuje się,
    że reranker MA TĘ SAMĄ wadę -- ocenianie wyłącznie względem
    oryginalnego pytania potrafi ocenić właściwy, ale sformalizowany
    artykuł nisko, nawet gdy trafił do puli kandydatów dzięki
    rozszerzonemu zapytaniu (zweryfikowane empirycznie: "zakupy
    prywatne" -> art. 86 ustawy o VAT oceniany przez reranker na 0.230
    względem oryginalnego pytania, ale na 0.936 względem sformalizowanego
    rozszerzenia) -- branie maksimum naprawia to bez ryzykowania, że
    lepiej dopasowany do oryginalnego pytania kandydat przegra."""
    if not results:
        return results
    if isinstance(queries, str):
        queries = [queries]
    best_scores: list[float | None] = [None] * len(results)
    for q in queries:
        pairs = [(q, r["text"][:RERANK_TEXT_CHARS]) for r in results]
        scores = cross_encoder.predict(pairs)
        for i, score in enumerate(scores):
            score = float(score)
            if best_scores[i] is None or score > best_scores[i]:
                best_scores[i] = score
    for r, score in zip(results, best_scores):
        r["rerank_score"] = score
    return sorted(results, key=lambda r: -r["rerank_score"])


def fuse_rankings(results: list[dict]) -> list[dict]:
    """Łączy porządek z podobieństwa kosinusowego ('score') i z rerankera
    ('rerank_score') metodą Reciprocal Rank Fusion (RRF, k=0 -- suma
    odwrotności POZYCJI w każdym z dwóch rankingów, nie samych wyników,
    więc różnica skali między dwoma modelami nie ma znaczenia).

    Powód: reranker (`apply_rerank`) systematycznie POPRAWIA ranking w
    części przypadków (np. KP art. 18 -- kosinus #22, reranker #2; KRO
    art. 56 -- kosinus #7, reranker #2), ale w innych systematycznie go
    PSUJE, spychając krótki, ogólny/fundamentalny przepis poniżej
    tuzina dłuższych, szczegółowych przepisów pochodnych z tego samego,
    gęsto obsadzonego tematu -- mimo że bi-encoder ocenia go bardzo
    wysoko (KC art. 415 -- kosinus #2, reranker #9; KK art. 148 --
    kosinus #1, reranker #6; zob. PROGRESS.md, Kroki 8-10). Żadna z
    dwóch metod z osobna nie jest więc uniwersalnie lepsza -- RRF bierze
    to, co obie robią dobrze, bez arbitralnej wagi ani heurystyki opartej
    na długości tekstu (ryzykowałoby to przeuczeniem się pod te 2-3
    znane przypadki, czego świadomie unikaliśmy, dopóki nie zebraliśmy
    wystarczająco wielu -- patrz PROGRESS.md). Zweryfikowane na 10
    przypadkach referencyjnych z 6 dziedzin: naprawia oba poważne
    przypadki (art. 415: #9 -> #3, art. 148: #6 -> #2) bez psucia
    żadnego z wcześniej pewnych wyników (4 przypadki #1 zostają #1)."""
    if not results:
        return results
    by_cos = sorted(range(len(results)), key=lambda i: -results[i]["score"])
    by_rerank = sorted(range(len(results)), key=lambda i: -results[i].get("rerank_score", results[i]["score"]))
    cos_rank = {idx: pos for pos, idx in enumerate(by_cos)}
    rerank_rank = {idx: pos for pos, idx in enumerate(by_rerank)}
    for i, r in enumerate(results):
        r["fused_score"] = 1.0 / (cos_rank[i] + 1) + 1.0 / (rerank_rank[i] + 1)
    return sorted(results, key=lambda r: -r["fused_score"])


def fuse_rankings_multi_query(
    all_queries: list[str], result_lists: list[list[dict]], cross_encoder: CrossEncoder
) -> list[dict]:
    """Wariant `fuse_rankings()` dla WIELU wariantów zapytania (oryginalne +
    rozszerzenia, patrz `extra_queries` w RagIndex.search()) -- naprawia
    lukę znalezioną przy weryfikacji Kroku 11 (PROGRESS.md, Krok 12).

    Zwykła fuzja licząca ranking kosinusowy na SCALONEJ puli kandydatów
    (`merge_candidates`, które trzyma NAJWYŻSZY 'score' spośród wariantów
    per kandydat) ma efekt uboczny: kandydat trafiony słabo przez
    oryginalne pytanie, ale bardzo dobrze przez sformułowanie rozszerzenia
    (np. inny, sąsiedni przepis pasujący leksykalnie do sformalizowanej
    parafrazy), dostaje jego WYSOKI kosinus i na tej podstawie wygrywa z
    kandydatem faktycznie najlepiej pasującym do PRAWDZIWEJ intencji
    użytkownika (oryginalnego pytania) -- zweryfikowane empirycznie: KC
    art. 415 (kosinus #2 względem oryginalnego pytania, naprawione przez
    `fuse_rankings` na #3) wypada z powrotem poza top-8 z włączonym
    rozszerzeniem zapytania, bo sąsiednie art. 361/429 dostają wysoki
    kosinus względem SFORMALIZOWANEJ PARAFRAZY, nie oryginalnego pytania.

    Rozwiązanie: licz fuzję (kosinus+reranking) OSOBNO dla każdego
    wariantu zapytania -- kosinus i pozycja rerankera liczone tylko
    względem kandydatów faktycznie przywołanych przez TEN wariant, nie
    całej scalonej puli -- a dopiero potem weź MAKSIMUM wyniku fuzji
    spośród wariantów per kandydat (ten sam wzorzec "bierz maksimum
    spośród wariantów" co w `apply_rerank`, tylko zastosowany do wyniku
    fuzji, nie samego rerankingu). Naprawia oba przypadki jednocześnie
    (zweryfikowane na wszystkich znanych przypadkach referencyjnych z
    query expansion): KC art. 415 wraca w okolice #3-4, a przypadek z
    Kroku 7 (VAT art. 86, wymagający rozszerzenia, żeby w ogóle trafić
    w top-8) też zostaje naprawiony (#5), bo art. 86 nadal dostaje swoją
    najlepszą szansę -- w ramach fuzji liczonej dla WARIANTU rozszerzenia,
    gdzie jest kosinusowo bardzo dobry -- zamiast być oceniany kosinusowo
    tylko względem oryginalnego, potocznego pytania."""
    merged = merge_candidates(*result_lists)
    if not merged:
        return merged

    def key_of(r: dict) -> tuple:
        return (r["act_short"], r["article"])

    by_key = {key_of(r): r for r in merged}

    rerank_by_query: list[dict[tuple, float]] = []
    for q in all_queries:
        pairs = [(q, r["text"][:RERANK_TEXT_CHARS]) for r in merged]
        scores = cross_encoder.predict(pairs)
        rerank_by_query.append({key_of(r): float(s) for r, s in zip(merged, scores)})

    for key, r in by_key.items():
        r["rerank_score"] = max(rq[key] for rq in rerank_by_query)  # do wyświetlania/porównań, jak dotychczas

    best_fused: dict[tuple, float] = {}
    for qi, qlist in enumerate(result_lists):
        by_cos = sorted(qlist, key=lambda r: -r["score"])
        cos_rank = {key_of(r): i for i, r in enumerate(by_cos)}
        by_rr = sorted(qlist, key=lambda r: -rerank_by_query[qi][key_of(r)])
        rr_rank = {key_of(r): i for i, r in enumerate(by_rr)}
        for r in qlist:
            key = key_of(r)
            fused = 1.0 / (cos_rank[key] + 1) + 1.0 / (rr_rank[key] + 1)
            if key not in best_fused or fused > best_fused[key]:
                best_fused[key] = fused

    for key, r in by_key.items():
        r["fused_score"] = best_fused.get(key, 0.0)
    return sorted(merged, key=lambda r: -r["fused_score"])



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


def merge_candidates(*result_lists: list[dict]) -> list[dict]:
    """Łączy wyniki z kilku wywołań rank_chunks (np. zapytanie oryginalne +
    jego rozszerzenia -- patrz `extra_queries` w RagIndex.search()) w jedną
    listę bez duplikatów tego samego artykułu -- dla zdublowanego
    (act_short, article) zostaje wersja z WYŻSZYM 'score'. Kolejność
    wejściowa nieistotna dla wyniku (przechodzi przez dict), przydatne do
    testowania bez konstruowania pełnych wywołań rank_chunks()."""
    best: dict[tuple, dict] = {}
    for results in result_lists:
        for r in results:
            key = (r["act_short"], r["article"])
            if key not in best or r["score"] > best[key]["score"]:
                best[key] = r
    return list(best.values())


def release_accelerator_cache() -> None:
    """Oddaje systemowi pamięć, którą PyTorch trzyma w swoim cache na GPU
    (MPS/CUDA) po zakończonych obliczeniach embeddingów/rerankingu.

    Bez tego skrypty odpytujące wiele przypadków w jednym procesie (baterie
    regresyjne) rosną skokowo przy każdym nowym, dłuższym zestawie par
    pytanie-fragment dla rerankera -- zmierzone: pamięć sterownika MPS
    3,4 -> 5,4 GB na samym retrievalu 20 pytań, a w pełnym przebiegu
    check_qwen_answer_regressions.py cały proces doszedł do ~12 GB. Z
    wywołaniem po każdym przypadku: stałe ~1,3 GB (= same wagi modeli).
    Patrz PROGRESS.md, Krok 57."""
    import gc

    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()




class RagIndex:
    def __init__(
        self,
        rerank: bool = True,
        fuse: bool = True,
        model: SentenceTransformer | None = None,
        cross_encoder: CrossEncoder | None = None,
    ):
        """Dwuetapowe wyszukiwanie: kosinus (bi-encoder) -> reranker (cross-encoder) -> fuzja RRF.
        `rerank=False` daje wyłącznie etap kosinusowy (zachowanie sprzed backportu, do porównań);
        `fuse=False` zostawia sam reranking bez fuzji RRF. `model`/`cross_encoder` pozwalają
        wstrzyknąć już załadowane modele (żeby nie ładować drugiej kopii)."""
        if not INDEX_EMB_PATH.exists():
            raise FileNotFoundError(
                "Brak indeksu RAG. Najpierw uruchom: python scripts/build_rag_index.py"
            )
        self.embeddings = np.load(INDEX_EMB_PATH)  # znormalizowane wektory, shape (N, D)
        self.meta = json.loads(INDEX_META_PATH.read_text(encoding="utf-8"))
        self.model = model if model is not None else SentenceTransformer(EMBEDDING_MODEL)
        if cross_encoder is not None:
            self.cross_encoder = cross_encoder
        else:
            self.cross_encoder = CrossEncoder(RERANKER_MODEL) if rerank else None
        self.fuse = fuse

    def search(
        self,
        query: str,
        top_k: int = 8,
        as_of: str | None = None,
        extra_queries: list[str] | None = None,
        candidate_pool: int | None = None,
    ) -> list[dict]:
        """`extra_queries` (opcjonalnie): dodatkowe sformułowania tego samego pytania (każde szuka
        osobno, wyniki łączy `fuse_rankings_multi_query`). `candidate_pool`: stała pula kandydatów
        na wariant zamiast `max(top_k, RERANK_CANDIDATE_POOL)` (dla baterii regresyjnych, które
        chcą długą listę wyników przy fuzji liczonej na puli produkcyjnej)."""
        all_queries = [query] + list(extra_queries or [])
        if candidate_pool is not None:
            candidate_k = candidate_pool
        else:
            candidate_k = max(top_k, RERANK_CANDIDATE_POOL) if self.cross_encoder else top_k
        result_lists = []
        for q in all_queries:
            query_vec = self.model.encode([QUERY_PREFIX + q], normalize_embeddings=True)[0]
            # embeddingi są znormalizowane -> iloczyn skalarny = podobieństwo kosinusowe
            result_lists.append(rank_chunks(self.embeddings, self.meta, query_vec, candidate_k, as_of))
        if self.cross_encoder:
            if self.fuse and len(all_queries) > 1:
                results = fuse_rankings_multi_query(all_queries, result_lists, self.cross_encoder)
            else:
                results = merge_candidates(*result_lists) if len(result_lists) > 1 else result_lists[0]
                results = apply_rerank(all_queries, results, self.cross_encoder)
                if self.fuse:
                    results = fuse_rankings(results)
        else:
            results = merge_candidates(*result_lists) if len(result_lists) > 1 else result_lists[0]
            results = sorted(results, key=lambda r: -r["score"])
        return results[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Wyszukiwanie w indeksie RAG (bez odpalania modelu językowego).")
    parser.add_argument("query", nargs="+", help="Treść pytania")
    parser.add_argument("--no-rerank", action="store_true", help="Tylko etap kosinusowy, bez rerankera i fuzji (do porównań).")
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

    idx = RagIndex(rerank=not args.no_rerank)
    results = idx.search(query, top_k=8, as_of=args.as_of)

    as_of_label = f" (stan na {args.as_of})" if args.as_of else ""
    print(f'\nPytanie: "{query}"{as_of_label}\n')
    for r in results:
        version_label = ""
        if r.get("valid_from") or r.get("valid_to"):
            version_label = f" [stan: {r.get('valid_from') or '...'} -- {r.get('valid_to') or 'nadal'}]"
        ref = r["article"] if r["article"].startswith("załącznik") else f"art. {r['article']}"
        print(f"[{r['score']:.3f}] {r['act_title']} -- {ref} ({r['eli']}){version_label}")
        preview = r["text"][:220].replace("\n", " ")
        print(f"          {preview}...\n")


if __name__ == "__main__":
    main()
