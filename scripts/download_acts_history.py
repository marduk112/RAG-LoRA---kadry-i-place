"""
Pobiera historyczne (przeszłe) wersje tekstu ujednoliconego ustaw z ELI API
Sejmu -- rozszerzenie `download_acts.py` o wymiar czasowy ("prawo
obowiązujące na dany dzień w przeszłości"), patrz PROGRESS.md, "Co dalej"
pkt 4 i Krok 18, oraz plan implementacji w
`/Users/szymon/.claude/plans/drifting-toasting-wozniak.md`.

Dla każdej ustawy z `ACTS` (`download_acts.py`) pobiera listę obwieszczeń
ogłaszających kolejne teksty ujednolicone (`references['Inf. o tekście
jednolitym']`), przelicza okna ważności (`compute_version_windows`) i
parsuje każdą PRZESZŁĄ wersję tym samym pipeline'em co bieżący tekst
(`_extract_articles`, przez `process_act_version`). Bieżący tekst ("U", już
pobrany przez zwykły `download_acts.py`) NIE jest tu ponownie pobierany --
tylko jego granica ważności (`current_valid_from`) jest zapisywana, do
wykorzystania przez `build_rag_index.py --include-history` przy łataniu
`valid_from` bieżących artykułów tej samej ustawy.

Opt-in, osobno od zwykłego `download_acts.py`: znacznie więcej zapytań do
publicznego API (jedno obwieszczenie + jeden PDF na wersję, ~59 wersji
łącznie na 7 ustaw wg spike'a z Kroku 18), więc nie jest częścią domyślnego
przebiegu ani `main()` z `download_acts.py`.

Historyczne PDF-y NIE są zapisywane do `data/raw/` (kolidowałyby z plikami
bieżącego tekstu już tam obecnymi dla tej samej ustawy) -- tylko sparsowany
JSON trafia do wyniku.

Wyjście: `data/processed/{skrot}_history.json` na ustawę:
    {"current_valid_from": "RRRR-MM-DD" | null, "articles": [...]}

Uruchomienie:
    python scripts/download_acts_history.py                     # wszystkie ustawy z ACTS
    python scripts/download_acts_history.py --acts ustawa_o_ppk  # tylko wybrane (po przecinku)
"""

import argparse
import json
import time

from download_acts import ACTS, PROCESSED_DIR, fetch_version_windows, process_act_version


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--acts",
        type=str,
        default=None,
        help="Lista skrótów ustaw po przecinku (domyślnie: wszystkie z ACTS)",
    )
    args = parser.parse_args()

    if args.acts:
        wanted = {s.strip() for s in args.acts.split(",") if s.strip()}
        acts = [a for a in ACTS if a["short"] in wanted]
        missing = wanted - {a["short"] for a in acts}
        if missing:
            raise SystemExit(f"Nieznane skróty ustaw: {', '.join(sorted(missing))}")
    else:
        acts = ACTS

    for act in acts:
        short = act["short"]
        print(f"[{short}] pobieram listę obwieszczeń (teksty jednolite)...")
        base_eli, base_title, windows = fetch_version_windows(act)

        past_windows = [w for w in windows if not w.get("is_current")]
        current_window = next(w for w in windows if w.get("is_current"))

        print(f"[{short}] {len(past_windows)} przeszłych wersji tekstu do pobrania")

        all_articles = []
        for w in past_windows:
            source = {"publisher": w["publisher"], "year": w["year"], "position": w["position"]}
            print(
                f"[{short}] pobieram wersję {w['announcement_eli']} "
                f"(obowiązywała {w['valid_from']} -- {w['valid_to']})..."
            )
            articles = process_act_version(
                source, short, base_eli, base_title, w["valid_from"], w["valid_to"], w["announcement_eli"]
            )
            print(f"[{short}]   {len(articles)} artykułów")
            all_articles.extend(articles)
            time.sleep(1)  # nie bombardujemy publicznego API bez potrzeby -- ten sam wzorzec co download_acts.py

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / f"{short}_history.json"
        out_path.write_text(
            json.dumps(
                {"current_valid_from": current_window["valid_from"], "articles": all_articles},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[{short}] zapisano {len(all_articles)} artykułów historycznych -> {out_path}")
        time.sleep(1)


if __name__ == "__main__":
    main()
