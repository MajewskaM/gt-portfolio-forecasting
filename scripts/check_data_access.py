"""
Kontrola dostępu do prawdziwych danych - do uruchomienia przed pierwszą analizą.
- python -m scripts.check_data_access # szybki test: SPY + jedno słowo GT
- python -m scripts.check_data_access --full # + ceny wszystkich aktywów presetu domyślnego

Sprawdza, czy Yahoo Finance i Google Trends odpowiadają oraz czy format danych jest
zgodny z założeniami kodu: kolumny OHLCV, indeks bez strefy czasowej, rozdzielczość GT
tygodniowa z etykietami w niedzielę.

Kod wyjścia: 0 = wszystko OK, 1 = co najmniej jedna kontrola nie przeszła.
Kontrola GT zapisuje plik w data/gt/ (to samo zapytanie jest potem użyte przez potok).
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile

import pandas as pd

from config.portfolio_config import FeatureSet, PortfolioConfig
from src.data.acquisition import DataAcquisition, infer_resolution

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


def _report(name: str, ok: bool, detail: str) -> bool:
    print(f"[{'OK  ' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def check_ohlcv(cfg: PortfolioConfig) -> bool:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            c = PortfolioConfig(
                start_date=cfg.start_date,
                end_date=cfg.end_date,
                cache_dir=tmp,
                gt_dir=tmp,
            )
            df = DataAcquisition(c).fetch_ohlcv_for(["SPY"])["SPY"]
    except Exception as exc:
        return _report("Yahoo Finance (SPY)", False, str(exc)[:200])
    ok = (
        len(df) > 200
        and df.index.tz is None
        and df["Close"].gt(0).all()
        and list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    )
    return _report(
        "Yahoo Finance (SPY)",
        ok,
        f"{len(df)} sesji {df.index.min().date()} - {df.index.max().date()}, "
        f"ostatni Close={df['Close'].iloc[-1]:.2f}",
    )


def check_gt(cfg: PortfolioConfig, keyword: str) -> bool:
    try:
        acq = DataAcquisition(cfg)
        frame = acq.fetch_gt_frame([keyword])
    except Exception as exc:
        return _report(f"Google Trends ('{keyword}')", False, str(exc)[:240])
    s = frame.iloc[:, 0]
    res = infer_resolution(s.index)
    sundays = float((s.index.dayofweek == 6).mean())
    ok = res == "weekly" and sundays > 0.95 and s.notna().all() and s.std() > 0
    detail = (
        f"{len(s)} punktów, rozdzielczość={res}, etykiety w niedzielę={sundays:.0%}, "
        f"zakres wartości {s.min():.0f}-{s.max():.0f}"
    )
    if res != "weekly" or sundays <= 0.95:
        detail += "  <- GT zakłada tygodnie z etykietą w niedzielę; zakres dat/geo do sprawdzenia"
    return _report(f"Google Trends ('{keyword}')", ok, detail)


def check_full(cfg: PortfolioConfig) -> bool:
    try:
        ohlcv = DataAcquisition(cfg).fetch_ohlcv()
    except Exception as exc:
        return _report("OHLCV wszystkich aktywów", False, str(exc)[:240])
    ok = len(ohlcv) == cfg.n_assets and all(len(df) > 500 for df in ohlcv.values())
    return _report(
        "OHLCV wszystkich aktywów",
        ok,
        f"{len(ohlcv)}/{cfg.n_assets} aktywów, sesji: "
        f"{min(len(d) for d in ohlcv.values())}-{max(len(d) for d in ohlcv.values())}",
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="pobierz ceny wszystkich aktywów presetu domyślnego",
    )
    ap.add_argument("--keyword", default="MSFT stock")
    ap.add_argument(
        "--start-date",
        default=PortfolioConfig.__dataclass_fields__["start_date"].default,
    )
    ap.add_argument(
        "--end-date", default=PortfolioConfig.__dataclass_fields__["end_date"].default
    )
    args = ap.parse_args()
    cfg = PortfolioConfig(start_date=args.start_date, end_date=args.end_date)
    results = [check_ohlcv(cfg), check_gt(cfg, args.keyword)]
    if args.full:
        results.append(check_full(cfg))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
