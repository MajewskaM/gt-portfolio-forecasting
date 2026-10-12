"""
Ostateczna konfiguracja słów kluczowych Google Trends.

  MACRO_KEYWORDS  - dwa hasła ogólnorynkowe, sumowane w jeden sygnał wspólny dla portfela
                    (FeatureSet.WITH_GT_MACRO),
  ASSET_KEYWORDS  - słowo per aktywo (FeatureSet.WITH_GT_ASSET).
"""

from __future__ import annotations

from typing import Dict, List

MACRO_KEYWORDS: List[str] = ["stock market", "S&P 500"]

ASSET_KEYWORDS: Dict[str, str] = {
    "MSFT": "MSFT stock",
    "AAPL": "AAPL stock",
    "GOOGL": "Google stock",
    "JPM": "JPM stock",
    "JNJ": "JNJ stock",
    "XOM": "XOM stock",
    "PG": "Procter & Gamble stock",
    "SPY": "SPY etf",
    "QQQ": "QQQ etf",
    "GLD": "gold price",
    "SLV": "silver price",
    "TLT": "treasury yield",
}


def keywords_for_assets(tickers: List[str]) -> Dict[str, str]:
    """Zwraca słowa kluczowe dla zadanych tickerów."""
    missing = [t for t in tickers if t not in ASSET_KEYWORDS]
    if missing:
        raise KeyError(f"Brak zdefiniowanego słowa kluczowego GT dla: {missing}")
    return {t: ASSET_KEYWORDS[t] for t in tickers}
