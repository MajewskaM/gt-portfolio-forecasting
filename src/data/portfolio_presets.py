"""
Ostateczna konfiguracja portfeli. Każdy preset: {ticker: udział wartościowy}, suma = 1.0.

Skład i wagi są decyzja projektową.
"""

from __future__ import annotations

from typing import Dict, List

PORTFOLIO_PRESETS: Dict[str, Dict[str, float]] = {
    # 1. DOMYŚLNY: 9 aktywów - szeroki rynek akcji USA, pięć sektorów, złoto, obligacje skarbowe.
    # Akcje 75%, złoto 15%, obligacje skarbowe 10%.
    "diversified_9": {
        "SPY": 0.20,  # szeroki rynek USA (rdzeń)
        "MSFT": 0.10,  # IT
        "AAPL": 0.10,  # IT / elektronika użytkowa
        "JPM": 0.10,  # finanse
        "JNJ": 0.10,  # ochrona zdrowia
        "XOM": 0.10,  # energia
        "PG": 0.05,  # dobra podstawowe
        "GLD": 0.15,  # złoto
        "TLT": 0.10,  # długie obligacje skarbowe USA
    },
    # 2. Koncentracja sektorowa. Nakładanie QQQ z MSFT/AAPL/GOOGL jest tu celowe.
    "tech_heavy": {
        "MSFT": 0.25,
        "AAPL": 0.25,
        "GOOGL": 0.20,
        "QQQ": 0.20,
        "SPY": 0.10,
    },
    # 3. Defensywny: niski profil ryzyka, kontrast względem tech_heavy.
    "defensive": {
        "JNJ": 0.25,
        "PG": 0.20,
        "GLD": 0.20,
        "TLT": 0.15,
        "SPY": 0.20,
    },
    # 4. Przeważony w surowce (metale szlachetne: złoto i srebro).
    "commodities_tilted": {
        "GLD": 0.35,
        "SLV": 0.25,
        "SPY": 0.20,
        "QQQ": 0.10,
        "MSFT": 0.10,
    },
}

# Presety, na których powtarzamy najważniejsze porównanie
ROBUSTNESS_PRESETS: List[str] = ["tech_heavy", "defensive", "commodities_tilted"]

# ETF-y występujące w presetach (reszta to spółki); potrzebne regule SINGLE_ASSET
ETF_TICKERS = frozenset({"SPY", "QQQ", "GLD", "TLT", "SLV"})


def get_preset(portfolio_id: str) -> Dict[str, float]:
    """Zwraca KOPIĘ presetu i sprawdza, że wagi sumują się do 1."""
    if portfolio_id not in PORTFOLIO_PRESETS:
        raise KeyError(
            f"Nieznane portfolio {portfolio_id!r}. Dostępne: {list(PORTFOLIO_PRESETS)}"
        )
    weights = PORTFOLIO_PRESETS[portfolio_id]
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Preset {portfolio_id!r} ma wagi sumujące się do {total}, nie do 1.0"
        )
    return dict(weights)


def default_single_ticker(assets: Dict[str, float]) -> str:
    """Reguła wyboru aktywa dla TargetMode.SINGLE_ASSET (ustalona z góry):
    - SPÓŁKA (nie ETF) o największej wadze w portfelu; remis rozstrzyga kolejność w definicji presetu.
    - Gdy portfel nie zawiera spółek, wybierane jest aktywo o największej wadze."""
    stocks = {t: w for t, w in assets.items() if t not in ETF_TICKERS}
    pool = stocks or assets
    best = max(pool.values())
    return next(t for t, w in pool.items() if w == best)
