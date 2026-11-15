"""
Bateria regresyjna samego wyszukiwania (bez modelu językowego): dla każdego pytania z
data/retrieval_battery.json sprawdza, na której pozycji listy wyników jest oczekiwany artykuł.
Kod wyjścia != 0, jeśli któryś artykuł jest poza `max_acceptable_rank`.

    python scripts/check_retrieval_regressions.py              # reranker + fuzja RRF (domyślnie)
    python scripts/check_retrieval_regressions.py --no-rerank   # tylko kosinus, do porównania

Etykiety (artykuł oczekiwany dla pytania) sprawdzono po treści artykułów w indeksie; zestaw jest
mały (26 pytań), więc to wskazówka jakości, nie dowód.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from rag_search import RagIndex  # noqa: E402

BATTERY = ROOT / "data" / "retrieval_battery.json"
LIST_LEN = 30


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-rerank", action="store_true", help="Tylko etap kosinusowy (bez rerankera i fuzji).")
    args = parser.parse_args()

    cases = json.loads(BATTERY.read_text(encoding="utf-8"))
    idx = RagIndex(rerank=not args.no_rerank)
    ranks, failed = {}, []
    for c in cases:
        results = idx.search(c["question"], top_k=LIST_LEN, candidate_pool=30)
        pos = [i + 1 for i, r in enumerate(results) if (r["act_short"], r["article"]) == (c["act_short"], c["article"])]
        rank = pos[0] if pos else None
        ranks[c["id"]] = rank
        ok = rank is not None and rank <= c["max_acceptable_rank"]
        if not ok:
            failed.append(c["id"])
        print(f"[{'OK ' if ok else 'BAD'}] {c['id']}: art. {c['article']} ({c['act_short']}) -> pozycja {rank}")

    vals = list(ranks.values())
    print(
        f"\nRazem {len(vals)}: top-1 {sum(v == 1 for v in vals)}, top-3 {sum(v is not None and v <= 3 for v in vals)}, "
        f"top-8 {sum(v is not None and v <= 8 for v in vals)}, poza top-{LIST_LEN} {sum(v is None for v in vals)}"
    )
    if failed:
        print("Poza dopuszczalną pozycją:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
