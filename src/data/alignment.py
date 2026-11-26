"""
Łączenie danych Google Trends (tygodniowych) z cenami (dziennymi) i budowa zbioru danych.

Zasady (sekcje 5.6 i 7.2 raportu):
  1. Opóźnienie o 1 tydzień: wartość GT za tydzień jest znana dopiero po jego zakończeniu,
     więc model dostaje ją od poniedziałku następnego tygodnia.
  2. Wartość tygodniową powielamy na wszystkie dni sesyjne (forward fill).
  3. Nie interpolujemy, bo to wprowadziłoby informację z przyszłości.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config.portfolio_config import PortfolioConfig


@dataclass
class AlignedDataset:
    dates: pd.DatetimeIndex
    portfolio_value: pd.Series
    asset_prices: pd.DataFrame
    shares: Dict[str, float]
    single_asset_price: pd.Series
    gt_features: pd.DataFrame
    train_end: int # liczba sesji w zbiorze uczącym
    val_end: int # liczba sesji w uczącym + walidacyjnym


def align_gt_to_trading_days(gt_series: Dict[str, pd.Series], trading_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Dla każdego dnia sesyjnego bierze ostatnią znaną wartość GT (z opóźnieniem 1 tygodnia)."""
    result = pd.DataFrame(index=trading_index)
    for name, series in gt_series.items():
        delayed = series.sort_index().shift(1) # opóźnienie o 1 tydzień
        result[name] = delayed.reindex(trading_index, method="ffill") # ostatnia znana wartość
    return result


def build_aligned_dataset(cfg: PortfolioConfig, ohlcv: Dict[str, pd.DataFrame], gt_series: Dict[str, pd.Series]) -> AlignedDataset:
    """Zbiór danych o tych samych datach dla wszystkich modeli."""
    prices = pd.concat({t: ohlcv[t]["Close"] for t in cfg.tickers}, axis=1).dropna().sort_index()
    gt = align_gt_to_trading_days(gt_series, prices.index)

    # na początku próby GT jeszcze nie ma - te dni odrzucamy
    if not gt.empty:
        has_gt = gt.notna().all(axis=1)
        prices, gt = prices[has_gt], gt[has_gt]

    # buy-and-hold: stała liczba jednostek kupiona pierwszego dnia
    shares = {t: cfg.initial_capital * w / prices[t].iloc[0] for t, w in cfg.assets.items()}
    portfolio_value = sum(prices[t] * shares[t] for t in cfg.tickers)

    n = len(prices)
    return AlignedDataset(
        dates=prices.index,
        portfolio_value=portfolio_value,
        asset_prices=prices,
        shares=shares,
        single_asset_price=prices[cfg.single_ticker],
        gt_features=gt,
        train_end=int(n * cfg.train_frac),
        val_end=int(n * (cfg.train_frac + cfg.val_frac)),
    )


def scale_gt_train_only(gt: Optional[pd.DataFrame], train_end: int) -> Optional[np.ndarray]:
    """Standaryzacja GT: średnia i odchylenie liczone tylko ze zbioru uczącego."""
    if gt is None or gt.empty:
        return None
    values = gt.to_numpy(dtype=float)
    return StandardScaler().fit(values[:train_end]).transform(values)