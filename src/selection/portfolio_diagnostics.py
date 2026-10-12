"""
Ilościowa ocena aktywów i kandydackich portfeli.

Wszystkie statystyki wyboru liczymy na części treningowej próby - pierwsze train_frac dni
Wybór składu na całej próbie wprowadzałby informację z okresu testowego do decyzji projektowych, a chcemy
te dwie rzeczy uniezależnić i nie wybeirac pod wynik.

Miary dywersyfikacji:
- średnia i maksymalna korelacja par,
- współczynnik dywersyfikacji DR = sum_i w_i*sigma_i / sigma_p
    DR=1 gdy aktywa są idealnie skorelowane, rośnie wraz z dywersyfikacją;
    1/DR to część "średniej" zmienności, która zostaje po agregacji - bezpośrednio związane z hipotezą H1 (agregacja redukuje szum),
- efektywna liczba niezależnych czynników N_eff = N^2 / sum(lambda_k^2)
    współczynnik uczestnictwa wartości własnych macierzy korelacji;
    N_eff=1 przy pełnej korelacji,
    N_eff=N przy braku korelacji,
- maksymalna waga sektora (metadane z universe.py),
- maksymalne obsunięcie (max drawdown) portfela buy-and-hold.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.selection.universe import UNIVERSE


# dane wejściowe
def close_frame(ohlcv: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Close wszystkich aktywów na wspólnych datach (inner join)."""
    return pd.concat({t: df["Close"] for t, df in ohlcv.items()}, axis=1).dropna()


def train_returns(close: pd.DataFrame, train_frac: float = 0.70) -> pd.DataFrame:
    """Dzienne proste stopy zwrotu z części treningowej."""
    n_train = int(len(close) * train_frac)
    return close.iloc[:n_train].pct_change().dropna()


# statystyki aktywów
def max_drawdown(value: pd.Series) -> float:
    return float((value / value.cummax() - 1.0).min())


def asset_statistics(
    ohlcv: Dict[str, pd.DataFrame], train_frac: float = 0.70
) -> pd.DataFrame:
    """Tabela cech aktywów: pokrycie danych, płynność, zmienność.
    Dane z części treningowej; płynność = mediana dziennego obrotu Close*Volume [USD].
    """
    close = close_frame(ohlcv)
    n_train = int(len(close) * train_frac)
    idx = close.index[:n_train]
    rows = []
    for t, df in ohlcv.items():
        d = df.loc[df.index.isin(idx)]
        r = d["Close"].pct_change().dropna()
        info = UNIVERSE.get(t)
        rows.append(
            {
                "ticker": t,
                "kind": info.kind if info else "?",
                "asset_class": info.asset_class if info else "?",
                "sector": info.sector if info else "?",
                "n_obs": len(d),
                "ann_vol_%": 100 * r.std() * np.sqrt(252),
                "median_dollar_volume_M": float(
                    (d["Close"] * d["Volume"]).median() / 1e6
                ),
                "mean_dollar_volume_M": float((d["Close"] * d["Volume"]).mean() / 1e6),
                "max_to_median_volume": float(
                    (d["Close"] * d["Volume"]).max()
                    / (d["Close"] * d["Volume"]).median()
                ),
            }
        )
    return pd.DataFrame(rows).set_index("ticker")


def liquidity_and_coverage_filter(
    stats: pd.DataFrame, min_dollar_volume_m: float = 50.0, min_obs_share: float = 0.98
) -> pd.DataFrame:
    """Filtry twarde: historia obejmuje niemal całą próbę i płynność nie
    mniejsza niż próg. Dodaje kolumny 'passes_*' oraz 'eligible'."""
    out = stats.copy()
    out["obs_share"] = out["n_obs"] / out["n_obs"].max()
    out["passes_coverage"] = out["obs_share"] >= min_obs_share
    out["passes_liquidity"] = out["median_dollar_volume_M"] >= min_dollar_volume_m
    out["eligible"] = out["passes_coverage"] & out["passes_liquidity"]
    return out


# portfele kandydujące
def _weights_vec(weights: Dict[str, float], cols: Iterable[str]) -> np.ndarray:
    w = np.array([weights.get(c, 0.0) for c in cols], dtype=float)
    return w / w.sum()


def portfolio_diagnostics(
    ret: pd.DataFrame, weights: Dict[str, float]
) -> Dict[str, float]:
    """Miary dywersyfikacji jednego portfela (wagi stałe, dzienne zwroty `ret`)."""
    cols = [t for t in weights if t in ret.columns]
    r = ret[cols]
    w = _weights_vec(weights, cols)
    cov = r.cov().to_numpy() * 252
    vol_i = np.sqrt(np.diag(cov))
    vol_p = float(np.sqrt(w @ cov @ w))
    corr = r.corr().to_numpy()
    n = len(cols)
    iu = np.triu_indices(n, k=1)
    avg_corr = float(corr[iu].mean()) if n > 1 else float("nan")
    lam = np.linalg.eigvalsh(corr)
    n_eff = float(n**2 / np.sum(lam**2))
    # buy-and-hold - stała liczba jednostek z pierwszego dnia
    growth = (1 + r).cumprod()
    value = growth @ w
    sector_w: Dict[str, float] = {}
    for t, wi in zip(cols, w):
        info = UNIVERSE.get(t)
        if info:
            sector_w[info.sector] = sector_w.get(info.sector, 0) + wi
    return {
        "n_assets": n,
        "avg_pairwise_corr": avg_corr,
        "max_pairwise_corr": float(corr[iu].max()) if n > 1 else float("nan"),
        "ann_vol_portfolio_%": 100 * vol_p,
        "diversification_ratio": float(w @ vol_i) / vol_p,
        "effective_n_factors": n_eff,
        "max_sector_weight": max(sector_w.values()) if sector_w else float("nan"),
        "max_drawdown_%": 100 * max_drawdown(value),
    }


def compare_candidates(
    ret: pd.DataFrame, candidates: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """Tabela porównawcza portfeli (wiersz = portfel)."""
    rows = {}
    for name, w in candidates.items():
        missing = [t for t in w if t not in ret.columns]
        if missing:
            rows[name] = {"missing": ",".join(missing)}
            continue
        rows[name] = portfolio_diagnostics(ret, w)
    return pd.DataFrame(rows).T


def high_correlation_pairs(ret: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """Pary aktywów o korelacji >= threshold (kandydaci do 'podwójnego liczenia')."""
    c = ret.corr()
    out = [
        {"a": a, "b": b, "corr": c.loc[a, b]}
        for i, a in enumerate(c.columns)
        for b in c.columns[i + 1 :]
        if c.loc[a, b] >= threshold
    ]
    return pd.DataFrame(out, columns=["a", "b", "corr"]).sort_values(
        "corr", ascending=False
    )


# selekcja zachłanna
def greedy_selection(
    ret: pd.DataFrame,
    pool: List[str],
    n_select: int = 9,
    seed: Optional[List[str]] = None,
    max_per_sector: int = 2,
) -> pd.DataFrame:
    """Zachłanna selekcja w przód: w każdym kroku dodaje aktywo z pool, które
    maksymalizuje współczynnik dywersyfikacji portfela, przy limicie
    max_per_sector aktywów z jednego sektora, żeby nie zdominował jej jeden sektor.

    Pokazuje, czy ręcznie zaprojektowany preset jest rozsądny względem prostego algorytmu.
    Wynik zależy od próby i nie jest jedynym uzasadnionym rozwiązaniem (DR sprzyja aktywom niskiej zmienności, np.
    obligacjom) - dlatego ostateczny skład wynika z jasnych reguł."""
    chosen = list(seed or [])
    rows = []
    cand = [t for t in pool if t in ret.columns and t not in chosen]
    while len(chosen) < n_select and cand:
        best, best_dr = None, -np.inf
        for t in sorted(cand):
            sec = UNIVERSE[t].sector
            if sum(UNIVERSE[c].sector == sec for c in chosen) >= max_per_sector:
                continue
            trial = chosen + [t]
            w = {a: 1 / len(trial) for a in trial}
            dr = portfolio_diagnostics(ret, w)["diversification_ratio"]
            if dr > best_dr:
                best, best_dr = t, dr
        if best is None:
            break
        chosen.append(best)
        cand.remove(best)
        rows.append(
            {"step": len(chosen), "added": best, "diversification_ratio": best_dr}
        )
    return pd.DataFrame(rows)


# wykresy
def plot_dollar_volume(
    ohlcv: Dict[str, pd.DataFrame],
    path: str,
    train_frac: float = 0.70,
    threshold_m: float = 50.0,
) -> None:
    """Rozkład dziennego obrotu (Close x Volume) każdego aktywa w okresie treningowym"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    close = close_frame(ohlcv)
    idx = close.index[: int(len(close) * train_frac)]
    dv = {
        t: (df.loc[df.index.isin(idx), "Close"] * df.loc[df.index.isin(idx), "Volume"])
        / 1e6
        for t, df in ohlcv.items()
    }
    order = sorted(dv, key=lambda t: dv[t].median())
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(order) + 1.8))
    ax.boxplot(
        [dv[t].clip(lower=1e-3) for t in order],
        vert=False,
        whis=(5, 95),
        showfliers=False,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black"),
        medianprops=dict(color="tab:blue"),
    )
    ax.set_yticks(range(1, len(order) + 1), order)
    ax.set_xscale("log")
    ax.axvline(
        threshold_m,
        color="tab:red",
        linestyle="--",
        label=f"próg {threshold_m:g} mln USD",
    )
    ax.set_xlabel("dzienny obrót [mln USD], skala logarytmiczna")
    ax.set_title("Dzienny obrót w okresie treningowym")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
