"""
Pobiera historyczne (przeszłe) wersje tekstu ujednoliconego ustaw z ELI API
Sejmu -- rozszerzenie `download_acts.py` o wymiar czasowy ("prawo
obowiązujące na dany dzień w przeszłości"), patrz PROGRESS.md, "Co dalej"
pkt 4 i Krok 18, oraz plan implementacji w
`<lokalny plik planu>`.

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
    python scripts/download_acts_history.py                     # wszystkie akty z konfiguracji
    python scripts/download_acts_history.py --domain podatkowe   # tylko jedna dziedzina
    python scripts/download_acts_history.py --acts ustawa_o_ppk  # tylko wybrane (po przecinku)
"""

import argparse
import json
import time

from download_acts import (
    HISTORY_TEXT_KINDS,
    PROCESSED_DIR,
    SCAN_QUALITY_MAX_ODD_PER_1000,
    ParsePool,
    _fetch_source,
    build_version_articles,
    default_parse_workers,
    fetch_version_windows,
    load_acts_config,
    odd_char_rate_per_1000,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Ogranicz do jednej dziedziny (pole 'domains' w data/acts_config.json).",
    )
    parser.add_argument(
        "--acts",
        type=str,
        default=None,
        help="Lista skrótów ustaw po przecinku (domyślnie: wszystkie pasujące do --domain).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Liczba procesów parsujących PDF-y (domyślnie: dobrane do rdzeni i RAM-u maszyny; 1 = sekwencyjnie).",
    )
    args = parser.parse_args()

    shorts = {s.strip() for s in args.acts.split(",") if s.strip()} if args.acts else None
    acts = load_acts_config(domain=args.domain, shorts=shorts)
    if args.acts:
        missing = shorts - {a["short"] for a in acts}
        if missing:
            raise SystemExit(f"Nieznane (lub spoza --domain) skróty ustaw: {', '.join(sorted(missing))}")
    if not acts:
        raise SystemExit("Brak aktów pasujących do podanych filtrów (--domain/--acts).")

    workers = args.workers or default_parse_workers()
    print(f"[INFO] parsowanie PDF-ów: {workers} proces(ów) równolegle (pobieranie sekwencyjne)")
    # Pobieranie sekwencyjne (łagodne dla publicznego API), parsowanie każdej
    # pobranej wersji od razu w tle w ParsePool -- wyniki zbierane na końcu,
    # w oryginalnej kolejności wersji, więc plik wynikowy jest identyczny jak
    # przy przebiegu sekwencyjnym (--workers 1).
    with ParsePool(workers) as pool:
        per_act = []
        for act in acts:
            short = act["short"]
            print(f"[{short}] pobieram listę obwieszczeń (teksty jednolite)...", flush=True)
            base_eli, base_title, windows = fetch_version_windows(act)

            past_windows = [w for w in windows if not w.get("is_current")]
            current_window = next(w for w in windows if w.get("is_current"))
            print(f"[{short}] {len(past_windows)} przeszłych wersji tekstu do pobrania", flush=True)

            jobs = []
            for w in past_windows:
                source = {"publisher": w["publisher"], "year": w["year"], "position": w["position"]}
                print(
                    f"[{short}] pobieram wersję {w['announcement_eli']} "
                    f"(obowiązywała {w['valid_from']} -- {w['valid_to']})...",
                    flush=True,
                )
                label = f"{short}@{w['valid_from'] or 'current'}"
                _meta, _title, _eli, pdf_bytes, kind, file_name = _fetch_source(source, label, HISTORY_TEXT_KINDS)
                jobs.append((w, source, kind, file_name, pool.submit(pdf_bytes, label)))
                time.sleep(1)  # nie bombardujemy publicznego API bez potrzeby -- ten sam wzorzec co download_acts.py
            per_act.append((act, base_eli, base_title, current_window, jobs))
            time.sleep(1)

        for act, base_eli, base_title, current_window, jobs in per_act:
            short = act["short"]
            all_articles = []
            for w, source, kind, file_name, future in jobs:
                articles = build_version_articles(
                    source,
                    short,
                    base_eli,
                    base_title,
                    w["valid_from"],
                    w["valid_to"],
                    w["announcement_eli"],
                    act.get("domains", []),
                    future.result(),
                    kind,
                    file_name,
                )
                odd = odd_char_rate_per_1000([a["text"] for a in articles])
                if odd > SCAN_QUALITY_MAX_ODD_PER_1000:
                    print(
                        f"[{short}@{w['valid_from']}]   [UWAGA] okno pominięte: zepsuty skan "
                        f"({odd:.1f} dziwnych znaków/1000 > {SCAN_QUALITY_MAX_ODD_PER_1000}) -- Krok 86"
                    )
                    continue
                print(f"[{short}@{w['valid_from']}]   {len(articles)} artykułów")
                all_articles.extend(articles)

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

if __name__ == "__main__":
    main()
