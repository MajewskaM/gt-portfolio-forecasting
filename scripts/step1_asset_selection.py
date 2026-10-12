"""
Dobór aktywów do portfeli: ilościowa weryfikacja kandydatów i presetów.

Uruchomienie:
    python -m scripts.step1_asset_selection
    python -m scripts.step1_asset_selection --start-date 2019-01-01 --end-date 2024-01-01

Wyniki results/step1/ :
  selection_funnel.csv       - wszystkie aktywa, kryteria liczbowe
  dollar_volume.png          - rozkład dziennego obrotu vs próg płynności
  high_correlation_pairs.csv - pary aktywów o wysokiej korelacji
  portfolio_comparison.csv   - miary dywersyfikacji presetów i portfela odniesienia
  greedy_selection.csv       - kontrola: selekcja zachłanna
"""

from __future__ import annotations

import argparse
import logging
import os

import pandas as pd

from config.portfolio_config import PortfolioConfig
from src.data.acquisition import DataAcquisition
from src.data.portfolio_presets import PORTFOLIO_PRESETS
from src.selection.universe import UNIVERSE
from src.selection.portfolio_diagnostics import (
    asset_statistics,
    close_frame,
    compare_candidates,
    greedy_selection,
    high_correlation_pairs,
    liquidity_and_coverage_filter,
    plot_dollar_volume,
    train_returns,
)

logger = logging.getLogger("step1a")
OUT = os.path.join("results", "step1")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
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
        "--min-dollar-volume-m",
        type=float,
        default=50.0,
        help="próg płynności [mln USD/dzień]",
    )
    ap.add_argument("--n-select", type=int, default=9)
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="dane sztuczne - tylko test działania kodu",
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    global OUT
    if args.synthetic:
        OUT = os.path.join("results", "step1_synthetic")
    os.makedirs(OUT, exist_ok=True)

    cfg = PortfolioConfig(start_date=args.start_date, end_date=args.end_date)
    tickers = sorted(set(UNIVERSE) | {t for w in PORTFOLIO_PRESETS.values() for t in w})
    if args.synthetic:
        import dataclasses

        from src.data.synthetic import fake_ohlcv_and_gt

        eq = {t: 1 / len(tickers) for t in tickers}
        ohlcv, _ = fake_ohlcv_and_gt(
            dataclasses.replace(cfg, assets=eq, single_ticker=tickers[0])
        )
        logger.warning("TRYB SYNTHETIC: dane sztuczne.")
    else:
        ohlcv = DataAcquisition(cfg).fetch_ohlcv_for(tickers)

    # 1. statystyki aktywów + filtry twarde
    stats = liquidity_and_coverage_filter(
        asset_statistics(ohlcv, args.train_frac), args.min_dollar_volume_m
    )
    plot_dollar_volume(
        ohlcv,
        os.path.join(OUT, "dollar_volume.png"),
        args.train_frac,
        args.min_dollar_volume_m,
    )

    # 2. korelacje na zbiorze treningowym
    close = close_frame(ohlcv)
    ret = train_returns(close, args.train_frac)
    pairs = high_correlation_pairs(ret, 0.85)
    pairs.round(3).to_csv(os.path.join(OUT, "high_correlation_pairs.csv"), index=False)

    # 3. porównanie presetów + portfel odniesienia (równe wagi, wszystkie kwalifikujące się aktywa)
    eligible = list(stats.index[stats["eligible"]])
    candidates = dict(PORTFOLIO_PRESETS)
    if eligible:
        candidates["ref_equal_weight_all_eligible"] = {
            t: 1 / len(eligible) for t in eligible
        }
    comp = compare_candidates(ret, candidates)
    comp.round(3).to_csv(os.path.join(OUT, "portfolio_comparison.csv"))

    # 4. kontrola: selekcja zachłanna z puli kwalifikujących się aktywów
    greedy = greedy_selection(
        ret, eligible, args.n_select, seed=["MSFT"], max_per_sector=2
    )
    greedy.round(3).to_csv(os.path.join(OUT, "greedy_selection.csv"), index=False)

    # 4b. lejek: kryteria liczbowe dla każdego aktywa z uniwersum
    used_in = {t: [k for k, w in PORTFOLIO_PRESETS.items() if t in w] for t in UNIVERSE}
    funnel = stats[
        [
            "kind",
            "asset_class",
            "sector",
            "ann_vol_%",
            "median_dollar_volume_M",
            "mean_dollar_volume_M",
            "max_to_median_volume",
            "obs_share",
            "passes_coverage",
            "passes_liquidity",
            "eligible",
        ]
    ].copy()
    funnel["avg_corr_to_others"] = ret.corr().apply(
        lambda c: (c.sum() - 1) / (len(c) - 1)
    )
    funnel["in_greedy_selection"] = funnel.index.isin(["MSFT"] + list(greedy["added"]))
    funnel["presets"] = [",".join(used_in[t]) for t in funnel.index]
    funnel.round(3).to_csv(os.path.join(OUT, "selection_funnel.csv"))

    print(comp.drop(columns=["missing"], errors="ignore").round(2).to_string())
    print(f"\nWyniki zapisano w {OUT}/")


if __name__ == "__main__":
    main()
