"""
Dobór słów kluczowych Google Trends dla aktywów portfeli.

    python -m scripts.step1_gt_keywords                      # wszystkie aktywa z presetów
    python -m scripts.step1_gt_keywords --tickers MSFT SPY   # wybrane
    python -m scripts.step1_gt_keywords --offline            # tylko dane z data/gt
    python -m scripts.step1_gt_keywords --related            # + powiązane zapytania do ręcznej weryfikacji
    python -m scripts.step1_gt_keywords --related-only       # tylko powiązane zapytania

Pobiera warianty słów z src/data/gt_dictionary.py,
każdy co najmniej dwukrotnie, zapisuje ostateczny wybór (ASSET_KEYWORDS) progami jakości i wskazuje
zamiennik, gdy wybór zawodzi. Zapisuje:
  results/step1/gt_keyword_variants.csv      - wskaźniki wszystkich wariantów
  results/step1/gt_final_validation.csv      - OK / ZAMIENIĆ / NO_GT dla wyboru ostatecznego
  results/step1/gt_threshold_sensitivity.csv - wrażliwość na progi
Flaga --write-selection zapisuje data/gt/selected_keywords.json (nadpisanie wyboru
ostatecznego zamiennikami z walidacji).

Również: nazwy plików cache zawierają zakres dat - zmiana zakresu = nowe pobranie.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import pandas as pd

from config.portfolio_config import PortfolioConfig
from src.data.acquisition import DataAcquisition
from src.data.gt_dictionary import ASSET_KEYWORDS
from src.data.portfolio_presets import PORTFOLIO_PRESETS
from src.selection.gt_keywords import (
    KeywordThresholds,
    apply_thresholds,
    fetch_related_cached,
    investigate_asset,
    validate_final_choice,
    threshold_sensitivity,
)

logger = logging.getLogger("step1b")
OUT = os.path.join("results", "step1")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--tickers", nargs="*", help="domyślnie: wszystkie aktywa z presetów"
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="liczba niezależnych pobrań wariantu (>=2 dla niezawodności)",
    )
    ap.add_argument(
        "--offline",
        action="store_true",
        help="nie łącz się z Google; użyj tylko plików w data/gt",
    )
    ap.add_argument(
        "--write-selection",
        action="store_true",
        help="zapisz data/gt/selected_keywords.json z zamiennikami (nadpisuje wybór ostateczny)",
    )
    ap.add_argument(
        "--related",
        action="store_true",
        help="zapisz powiązane zapytania (ręczna kontrola wieloznaczności)",
    )
    ap.add_argument(
        "--related-only",
        action="store_true",
        help="tylko powiązane zapytania (z pauzami i cache w results/step1/related); "
        "pomija resztę analizy - do ponawiania po błędzie 429",
    )
    ap.add_argument(
        "--start-date",
        default=PortfolioConfig.__dataclass_fields__["start_date"].default,
    )
    ap.add_argument(
        "--end-date", default=PortfolioConfig.__dataclass_fields__["end_date"].default
    )
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument(
        "--max-zero-share", type=float, default=KeywordThresholds.max_zero_share
    )
    ap.add_argument(
        "--min-median-to-peak", type=float, default=KeywordThresholds.min_median_to_peak
    )
    ap.add_argument(
        "--min-reliability", type=float, default=KeywordThresholds.min_reliability
    )
    ap.add_argument(
        "--min-corr-context",
        type=float,
        default=KeywordThresholds.min_corr_with_context,
    )
    ap.add_argument(
        "--min-corr-context-robust",
        type=float,
        default=KeywordThresholds.min_corr_context_robust,
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    os.makedirs(OUT, exist_ok=True)

    cfg = PortfolioConfig(start_date=args.start_date, end_date=args.end_date)
    # presety + USO (zbadane i odrzucone)
    tickers = args.tickers or sorted(
        {t for w in PORTFOLIO_PRESETS.values() for t in w} | {"USO"}
    )
    if args.related_only:
        chosen = {t: ASSET_KEYWORDS[t] for t in tickers if t in ASSET_KEYWORDS}
        rel = fetch_related_cached(
            cfg, list(chosen.values()), os.path.join(OUT, "related")
        )
        if not rel.empty:
            rel.to_csv(os.path.join(OUT, "gt_related_queries.csv"), index=False)
        print(
            f"Powiązane zapytania: {rel['keyword'].nunique() if not rel.empty else 0}/{len(chosen)} słów"
        )
        return
    acq = DataAcquisition(cfg)

    frames = []
    for t in tickers:
        logger.info("Ticker: %s", t)
        frames.append(
            investigate_asset(
                acq, t, args.repeats, args.train_frac, allow_network=not args.offline
            )
        )
    raw = pd.concat(frames, ignore_index=True)
    th = KeywordThresholds(
        args.max_zero_share,
        args.min_median_to_peak,
        args.min_reliability,
        args.min_corr_context,
        args.min_corr_context_robust,
    )
    evaluated = apply_thresholds(raw, th, require_reliability=args.repeats >= 2)
    sens = threshold_sensitivity(raw, th)

    evaluated.round(3).to_csv(os.path.join(OUT, "gt_keyword_variants.csv"), index=False)
    sens.to_csv(os.path.join(OUT, "gt_threshold_sensitivity.csv"), index=False)

    final_val = validate_final_choice(
        evaluated, {t: ASSET_KEYWORDS[t] for t in tickers if t in ASSET_KEYWORDS}
    )
    final_val.round(3).to_csv(os.path.join(OUT, "gt_final_validation.csv"), index=False)
    if args.write_selection:
        repl = {
            r.ticker: r.replacement
            for r in final_val.itertuples()
            if r.status == "ZAMIENIĆ"
        }
        os.makedirs(os.path.join("data", "gt"), exist_ok=True)
        with open(
            os.path.join("data", "gt", "selected_keywords.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(repl, f, indent=2, ensure_ascii=False)
        logger.info("Zapisano zamienniki: %s", repl)
    chosen = {t: ASSET_KEYWORDS[t] for t in tickers if t in ASSET_KEYWORDS}

    if args.related and not args.offline:
        rel = fetch_related_cached(
            cfg, list(chosen.values()), os.path.join(OUT, "related")
        )
        if not rel.empty:
            rel.to_csv(os.path.join(OUT, "gt_related_queries.csv"), index=False)

    print(
        final_val[
            ["ticker", "final_keyword", "status", "failing_criteria", "replacement"]
        ].to_string(index=False)
    )
    bad = [r.ticker for r in final_val.itertuples() if r.status != "OK"]
    print(f"\nDo decyzji: {bad or 'brak'}")


if __name__ == "__main__":
    main()
