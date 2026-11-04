"""
Kandydaci na słowa kluczowe Google Trends - analiza.

Warianty badane dla każdego aktywa: ticker, ticker + 'stock'/'etf', nazwa spółki, nazwa + 'stock'
oraz hasła-proxy. Ostateczny wybór słów jest w gałęzi main
(src/data/gt_dictionary.py); tu są tylko kandydaci i kolejność priorytetów.
"""

from __future__ import annotations

from typing import Dict, List

from src.data.gt_dictionary import ASSET_KEYWORDS
from src.selection.universe import get_info

# Kandydaci na hasła makro badani w Step 1 (do 5 w jednym zapytaniu GT = jedna skala)
MACRO_CANDIDATES: List[str] = [
    "stock market",
    "S&P 500",
    "recession",
    "inflation",
    "stock market crash",
]

# Hasła-proxy, których nie da się wygenerować regułą (tickery ETF-ów są rzadko
# wyszukiwane wprost, a niektóre - np. SPY - są wieloznaczne).
EXTRA_CANDIDATES: Dict[str, List[str]] = {
    "SPY": ["S&P 500", "SPY stock", "SPY etf"],
    "QQQ": ["Nasdaq 100", "QQQ etf", "QQQ stock"],
    "GLD": ["gold price", "gold etf", "GLD etf"],
    "SLV": ["silver price", "SLV stock", "silver etf"],
    # 'USO' samo jest wieloznaczne (zapaśnicy WWE The Usos, organizacja USO) - odpada w teście
    # odpornej korelacji z wariantem kontekstowym; 'USO stock' to pierwsze zapytanie
    # powiązane z 'USO' w Google Trends (weryfikacja ręczna).
    "USO": ["USO stock", "oil price", "crude oil price", "USO etf"],
    "TLT": ["treasury bonds", "treasury yield", "TLT etf"],
    "GOOGL": ["Google stock", "Alphabet stock"],
}

# Kolejność preferencji typów wariantów
PRIORITY_ORDER: List[str] = ["ticker_ctx", "ticker", "name_ctx", "name", "extra"]


def candidate_keywords(ticker: str) -> Dict[str, str]:
    """Zwraca {słowo kluczowe: typ_wariantu} dla tickera, w kolejności priorytetu.
    Typy: ticker (MSFT), ticker_ctx (MSFT stock), name (Microsoft),
    name_ctx (Microsoft stock), extra (hasła-proxy)."""
    info = get_info(ticker)
    ctx = "stock" if info.kind == "stock" else "etf"
    out: Dict[str, str] = {}

    def add(kw: str, kind: str) -> None:
        if kw.lower() not in {k.lower() for k in out}:
            out[kw] = kind

    add(f"{ticker} {ctx}", "ticker_ctx")
    add(ticker, "ticker")
    if info.kind == "stock":
        add(f"{info.short_name} stock", "name_ctx")
        add(info.short_name, "name")
    for kw in EXTRA_CANDIDATES.get(ticker, []):
        add(kw, "extra")
    final = ASSET_KEYWORDS.get(ticker)
    if final:  # wybór ostateczny zawsze podlega badaniu
        add(final, "extra")
    return out
