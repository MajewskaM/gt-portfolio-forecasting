"""
Syntetyczne dane OHLCV/GT - do testów
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from config.portfolio_config import FeatureSet, PortfolioConfig


def fake_ohlcv_and_gt(cfg: PortfolioConfig):
    """Generuje syntetyczne dane OHLCV i Google Trends.
    Cena każdego dnia to cena z poprzedniego dnia plus losowy przyrost.
    Dodajemy niewielki stały przyrost w jedną stronę (powodujac np. lekki trend wzrostowy).
    Tak najprościej symuluje się kurs aktywa, wszystko działa deterministycznie zeby dalo sie odtworzyc wyniki
    """
    dates = pd.bdate_range(start=cfg.start_date, end=cfg.end_date)
    n = len(dates)
    ohlcv = {}
    for ticker in cfg.tickers:
        rng = np.random.default_rng(zlib.crc32(ticker.encode()))
        ret = rng.normal(0.0003, 0.012, n)
        close = 100.0 * np.cumprod(1 + ret)
        spread = np.abs(rng.normal(0, 0.005, n)) * close
        high = close + spread * rng.uniform(0.2, 0.9, n)
        low = np.maximum(close - spread * rng.uniform(0.2, 0.9, n), close * 0.9)
        open_ = low + rng.uniform(0, 1, n) * (high - low)
        volume = rng.integers(1_000_000, 20_000_000, n)
        ohlcv[ticker] = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=dates,
        )

    def svi(label: str, offset: int) -> pd.Series:
        rng = np.random.default_rng(cfg.random_seed + 1 + offset)
        weeks = pd.date_range(start=cfg.start_date, end=cfg.end_date, freq="W-SUN")
        return pd.Series(
            np.clip(rng.normal(50, 15, len(weeks)), 5, 100), index=weeks, name=label
        )

    if cfg.feature_set == FeatureSet.BASELINE:
        gt = {}
    elif cfg.feature_set == FeatureSet.WITH_GT_MACRO:
        gt = {"Trend_SVI_MACRO": svi("Trend_SVI_MACRO", 0)}
    elif cfg.feature_set == FeatureSet.WITH_GT_ASSET:
        gt = {
            f"Trend_SVI_{t}": svi(f"Trend_SVI_{t}", i)
            for i, t in enumerate(cfg.tickers)
        }
    else:
        raise NotImplementedError(cfg.feature_set)
    return ohlcv, gt
