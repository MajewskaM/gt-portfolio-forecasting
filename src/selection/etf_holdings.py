"""
Udziały największych spółek w ETF-ach SPY i QQQ - do analizy
(ekspozycja portfela na spółkę = udział bezpośredni + udział przez ETF).

Dane pochodzą z oficjalnych publikacji emitentów (stan i źródło przy każdej pozycji). Udziały zmieniają
się codziennie wraz z cenami, więc służą do oszacowania rzędu wielkości nakładania.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

HOLDINGS: Dict[str, Dict] = {
    "SPY": {
        "as_of": "2026-10-08",
        "source": "State Street Global Advisors, strona funduszu SPY (ssga.com), sekcja Top Holdings",
        "weights_pct": {
            "NVDA": 8.34,
            "AAPL": 7.43,
            "MSFT": 5.80,
            "GOOGL": 5.50,
            "AMZN": 3.73,
        },
        "info_tech_sector_pct": 39.70,
    },
    "QQQ": {
        "as_of": "2026-03-31",
        "source": "Invesco QQQ Trust, Series 1, Form N-CSRS (SEC EDGAR), % of net assets; Alphabet = klasa A 3,43 + klasa C 3,19",
        "weights_pct": {
            "NVDA": 8.68,
            "AAPL": 7.63,
            "MSFT": 5.63,
            "GOOGL": 6.62,
            "AMZN": 4.58,
        },
        "info_tech_sector_pct": None,
    },
}

TRACKED = ["MSFT", "AAPL", "GOOGL", "NVDA"]


def look_through_exposure(weights: Dict[str, float]) -> Dict[str, float]:
    """Ekspozycja portfela [% wartości] na wybrane spółki: udział
    bezpośredni + (waga ETF x udział spółki w ETF)."""
    out = {t: 100.0 * weights.get(t, 0.0) for t in TRACKED}
    for etf, meta in HOLDINGS.items():
        w_etf = weights.get(etf, 0.0)
        for t, pct in meta["weights_pct"].items():
            out[t] = out.get(t, 0.0) + w_etf * pct
    return out


def look_through_table(portfolios: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Tabela: portfel x spółka (ekspozycja %) + sumy 'mega-cap tech'
    (MSFT+AAPL+GOOGL+NVDA) oraz część przypadająca na ekspozycję posrednia przez ETF."""
    rows = {}
    for name, w in portfolios.items():
        e = look_through_exposure(w)
        direct = {t: 100.0 * w.get(t, 0.0) for t in TRACKED}
        mega = ["MSFT", "AAPL", "GOOGL", "NVDA"]
        rows[name] = {
            **{t: round(e[t], 2) for t in TRACKED},
            "mega_tech_total_%": round(sum(e[t] for t in mega), 2),
            "mega_tech_direct_%": round(sum(direct[t] for t in mega), 2),
            "mega_tech_via_etf_%": round(sum(e[t] - direct[t] for t in mega), 2),
        }
    return pd.DataFrame(rows).T
