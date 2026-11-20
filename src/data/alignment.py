"""
Wyrównanie tygodniowych szeregów Google Trends do sesji giełdowych i budowa zbioru danych
(reguła z sekcji 5.6 i 7.2 raportu):
  1. lag = 1 tydzień: SVI o etykiecie niedzielnej d_i jest znane dopiero po tygodniu,
     więc trafia do modelu od poniedziałku tygodnia następnego,
  2. forward fill na wszystkie sesje tego tygodnia,
  3. brak interpolacji (interpolacja = wyciek informacji z przyszłości).
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
    train_end: int
    val_end: int


def align_gt_to_trading_days(gt_series: Dict[str, pd.Series], trading_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Lag = 1 tydzień + forward fill (bez interpolacji)."""
    cols = {}
    for name, s in gt_series.items():
        lagged = s.sort_index().astype(float).shift(1)
        cols[name] = lagged.reindex(lagged.index.union(trading_index)).ffill().reindex(trading_index)
    return pd.DataFrame(cols, index=trading_index)


def build_aligned_dataset(cfg: PortfolioConfig, ohlcv: Dict[str, pd.DataFrame], gt_series: Dict[str, pd.Series]) -> AlignedDataset:
    """Spójny zbiór z identycznym zakresem dat dla wszystkich wariantów modeli."""
    prices = pd.concat({t: ohlcv[t]["Close"] for t in cfg.tickers}, axis=1).dropna().sort_index()
    gt_df = align_gt_to_trading_days(gt_series, prices.index)
    if not gt_df.empty:  # odrzucamy początek bez znanej wartości GT
        valid = gt_df.notna().all(axis=1)
        prices, gt_df = prices[valid], gt_df[valid]

    shares = {t: cfg.initial_capital * w / float(prices[t].iloc[0]) for t, w in cfg.assets.items()}
    port_val = sum(prices[t] * shares[t] for t in cfg.tickers).rename("Portfolio_Value")
    n = len(prices)
    return AlignedDataset(
        dates=prices.index, portfolio_value=port_val, asset_prices=prices, shares=shares,
        single_asset_price=prices[cfg.single_ticker], gt_features=gt_df,
        train_end=int(n * cfg.train_frac), val_end=int(n * (cfg.train_frac + cfg.val_frac)),
    )


def scale_gt_train_only(gt_df: Optional[pd.DataFrame], train_end: int) -> Optional[np.ndarray]:
    """Standaryzacja cech GT wyłącznie na zbiorze treningowym (brak wycieku)."""
    if gt_df is None or gt_df.empty:
        return None
    arr = gt_df.to_numpy(dtype=float)
    return StandardScaler().fit(arr[:train_end]).transform(arr)
