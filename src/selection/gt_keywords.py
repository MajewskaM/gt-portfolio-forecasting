"""
Badanie i wybór słów kluczowych Google Trends dla aktywów.

Zasady:
  1. Dla każdego aktywa generujemy warianty słowa (gt_dictionary.candidate_keywords).
  2. Każdy wariant pobieramy osobno (pełna rozdzielczość gt 0-100) co najmniej
     dwukrotnie, niezależnie (szum próbkowania GT: kolejne pobrania różnią się).
  3. Dla każdego wariantu liczymy wskaźniki jakosci danych (na tygodniach
     zbioru treningowego): udział zer, mediana/szczyt, niezawodność (korelacja
     dwóch pobrań), jednoznaczność (korelacja z wariantem kontekstowym 'X stock').
  4. Reguła wyboru: pierwszy wariant z listy priorytetów, który spełnia wszystkie
     progi. Brak spełniającego => aktywo oznaczone NO_GT.

Kryteria dotyczą tylko jakości danych GT; związek z ceną jest testowany dopiero w modelach.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, replace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.acquisition import DataAcquisition, DataAcquisitionError
from src.selection.gt_candidates import (
    MACRO_CANDIDATES,
    PRIORITY_ORDER,
    candidate_keywords,
)

logger = logging.getLogger(__name__)

CONTEXT_TYPES = ("ticker_ctx", "name_ctx")  # słowo + 'stock'/'etf'
BARE_TYPES = ("ticker", "name")  # podlegają testowi jednoznaczności


@dataclass
class KeywordThresholds:
    """Progi jakości danych. Wartości domyślne to decyzje projektowe, analiza wrażliwości: threshold_sensitivity()."""

    max_zero_share: float = (
        0.10  # powyżej: sygnał to głównie zera (rozdzielczość 0-100)
    )
    min_median_to_peak: float = 0.10  # poniżej: seria zdominowana pojedynczymi skokami
    min_reliability: float = 0.80  # korelacja dwóch niezależnych pobrań
    min_corr_with_context: float = (
        0.50  # poniżej: wariant mierzy co innego niż 'X stock'
    )
    # to samo, ale odporną miarą (Spearman tygodniowych zmian) - nie da się go "zdobyć" jednym
    # wspólnym skokiem (np. wyszlo nam tak dla USO: kwiecień 2020).
    min_corr_context_robust: float = 0.40


# wskaźniki jakości
def svi_quality(svi: pd.Series) -> Dict[str, float]:
    """Jakość pojedynczej serii SVI (jedno pobranie)."""
    s = svi.astype(float)
    peak = s.max()
    med = s.median()
    return {
        "n_points": int(len(s)),
        "zero_share": float((s <= 0.5).mean()),  # '<1' w eksporcie = 0.5
        "mean_svi": float(s.mean()),
        "median_svi": float(med),
        "median_to_peak": float(med / peak) if peak > 0 else 0.0,
        "cv": float(s.std() / s.mean()) if s.mean() > 0 else float("nan"),
    }


def reliability(a: pd.Series, b: pd.Series) -> float:
    """Niezawodność test-retest: korelacja Pearsona poziomów dwóch niezależnych
    pobrań tego samego zapytania. Przy stałej serii zwraca NaN."""
    j = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(j) < 10 or j.iloc[:, 0].std() == 0 or j.iloc[:, 1].std() == 0:
        return float("nan")
    return float(j.corr().iloc[0, 1])


def robust_corr(a: pd.Series, b: pd.Series) -> float:
    """Odporna korelacja dwóch serii SVI: Spearman tygodniowych zmian. Zwykły Pearson poziomów
    bywa wygenerowany przez jeden wspólny skok (np. USO i 'USO etf' w kwietniu 2020: 0,95 na
    zbiorze treningowym, ale ok. 0,03-0,35 po jego wycięciu); różnice usuwają poziom, a rangi
    - wpływ pojedynczych obserwacji odstających. Przy stałej serii zwraca NaN."""
    j = pd.concat([a, b], axis=1, join="inner").dropna().diff().dropna()
    if len(j) < 10 or j.iloc[:, 0].nunique() < 2 or j.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(j.corr(method="spearman").iloc[0, 1])


def _train_weeks(svi: pd.Series, train_frac: float) -> pd.Series:
    return svi.iloc[: int(len(svi) * train_frac)]


# badanie aktywa
def investigate_asset(
    acq: DataAcquisition,
    ticker: str,
    n_repeats: int = 2,
    train_frac: float = 0.70,
    allow_network: bool = True,
) -> pd.DataFrame:
    """Wiersz na wariant słowa kluczowego z metrykami jakości."""
    variants = candidate_keywords(ticker)
    pulls: Dict[str, List[pd.Series]] = {}
    for kw in variants:
        pulls[kw] = []
        for r in range(n_repeats):
            try:
                frame = acq.fetch_gt_frame([kw], repeat=r, allow_network=allow_network)
                pulls[kw].append(frame.iloc[:, 0].astype(float))
            except DataAcquisitionError as exc:
                logger.warning(
                    "[%s] wariant %r (powt. %d) niedostępny: %s", ticker, kw, r, exc
                )
                break

    # wzorce kontekstowe: warianty z członem 'stock'/'etf' (jednoznaczne z konstrukcji) - oba oznaczaja cos zawiazanego z rynkiem
    ctx_refs = {
        kw: _train_weeks(pulls[kw][0], train_frac)
        for kw, kind in variants.items()
        if kind in CONTEXT_TYPES and pulls.get(kw)
    }
    rows = []
    for rank, (kw, kind) in enumerate(variants.items()):
        s_list = pulls[kw]
        row = {
            "ticker": ticker,
            "keyword": kw,
            "variant_type": kind,
            "priority": PRIORITY_ORDER.index(kind),
            "n_pulls": len(s_list),
        }
        if not s_list:
            row["status"] = "NO_DATA"
            rows.append(row)
            continue
        s0 = _train_weeks(s_list[0], train_frac)
        row.update(svi_quality(s0))
        row["reliability"] = (
            reliability(
                _train_weeks(s_list[0], train_frac), _train_weeks(s_list[1], train_frac)
            )
            if len(s_list) > 1
            else float("nan")
        )
        # Test jednoznaczności dotyczy tylko wariantów (ticker, nazwa): najlepsza
        # korelacja z dowolnym wariantem kontekstowym. Warianty z kontekstem są jednoznaczne
        # z konstrukcji, a hasła-proxy (extra) jest traktowany jako zamiennik - bez tego testu.
        refs = [reliability(s0, r) for k2, r in ctx_refs.items() if k2 != kw]
        refs = [x for x in refs if not np.isnan(x)]
        row["corr_with_context"] = (
            max(refs) if (kind in BARE_TYPES and refs) else float("nan")
        )
        refs_r = [robust_corr(s0, r) for k2, r in ctx_refs.items() if k2 != kw]
        refs_r = [x for x in refs_r if not np.isnan(x)]
        row["corr_context_robust"] = (
            max(refs_r) if (kind in BARE_TYPES and refs_r) else float("nan")
        )
        row["n_weeks_train"] = len(s0)
        rows.append(row)
    return pd.DataFrame(rows)


def apply_thresholds(
    df: pd.DataFrame, th: KeywordThresholds, require_reliability: bool = True
) -> pd.DataFrame:
    """Dodaje kolumny z testami progów i flagę 'eligible'.
    reliability NaN => niespełnione (gdy wymagane);
    corr_with_context NaN => wariant referencyjny."""
    d = df.copy()
    ok = d.get("status", pd.Series("OK", index=d.index)).ne("NO_DATA")
    d["ok_zero_share"] = d["zero_share"] <= th.max_zero_share
    d["ok_median_to_peak"] = d["median_to_peak"] >= th.min_median_to_peak
    d["ok_reliability"] = (
        (d["reliability"] >= th.min_reliability) if require_reliability else True
    )
    d["ok_unambiguous"] = (
        d["corr_with_context"].isna()
        | (d["corr_with_context"] >= th.min_corr_with_context)
    ) & (
        d["corr_context_robust"].isna()
        | (d["corr_context_robust"] >= th.min_corr_context_robust)
    )
    d["eligible"] = (
        ok
        & d["ok_zero_share"]
        & d["ok_median_to_peak"]
        & d["ok_reliability"]
        & d["ok_unambiguous"]
    )
    return d


def select_keywords(evaluated: pd.DataFrame) -> pd.DataFrame:
    """Dla każdego tickera: pierwszy kwalifikujący się wariant wg priorytetu.
    Brak => 'NO_GT'. Na koniec zwraca tabelę decyzji."""
    rows = []
    for ticker, g in evaluated.groupby("ticker", sort=False):
        el = g[g["eligible"]].sort_values(
            ["priority", "reliability"], ascending=[True, False]
        )
        if el.empty:
            rows.append(
                {
                    "ticker": ticker,
                    "keyword": None,
                    "decision": "NO_GT",
                    "reason": "żaden wariant nie spełnia progów jakości",
                }
            )
        else:
            best = el.iloc[0]
            rows.append(
                {
                    "ticker": ticker,
                    "keyword": best["keyword"],
                    "decision": "SELECTED",
                    "reason": f"najwyższy priorytet wśród {len(el)} kwalifikujących się "
                    f"({best['variant_type']})",
                }
            )
    return pd.DataFrame(rows)


def threshold_sensitivity(
    evaluated_raw: pd.DataFrame, base: KeywordThresholds
) -> pd.DataFrame:
    """Czy wybór słów jest odporny na progi? Powtarza selekcję dla progów
    zaostrzonych/złagodzonych o czynnik i zwraca, ile aktywów zmienia wybór."""
    base_sel = select_keywords(apply_thresholds(evaluated_raw, base)).set_index(
        "ticker"
    )["keyword"]
    rows = []
    for name, th in {
        "bazowe": base,
        "łagodniejsze": replace(
            base,
            max_zero_share=base.max_zero_share * 2,
            min_median_to_peak=base.min_median_to_peak / 2,
            min_reliability=base.min_reliability - 0.1,
            min_corr_with_context=base.min_corr_with_context - 0.2,
            min_corr_context_robust=base.min_corr_context_robust - 0.2,
        ),
        "ostrzejsze": replace(
            base,
            max_zero_share=base.max_zero_share / 2,
            min_median_to_peak=base.min_median_to_peak * 1.5,
            min_reliability=min(base.min_reliability + 0.1, 0.99),
            min_corr_with_context=base.min_corr_with_context + 0.2,
            min_corr_context_robust=base.min_corr_context_robust + 0.2,
        ),
    }.items():
        sel = select_keywords(apply_thresholds(evaluated_raw, th)).set_index("ticker")[
            "keyword"
        ]
        changed = [t for t in base_sel.index if str(sel.get(t)) != str(base_sel[t])]
        rows.append(
            {
                "progi": name,
                "n_NO_GT": int(sel.isna().sum()),
                "n_zmienionych_vs_bazowe": len(changed),
                "zmienione": ",".join(changed),
            }
        )
    return pd.DataFrame(rows)


# hasła makro
def investigate_macro(
    acq: DataAcquisition,
    final_keywords: List[str],
    n_repeats: int = 2,
    train_frac: float = 0.70,
    allow_network: bool = True,
) -> pd.DataFrame:
    """Jakość danych haseł sygnału makro, tymi samymi wskaźnikami co dla aktywów
    (udział zer, mediana/szczyt, niezawodność test-retest na tygodniach zbioru treningowego).
    Wiersze:
    - każde hasło z MACRO_CANDIDATES pobrane osobno (>= n_repeats niezależnych pobrań);
    - sam sygnał makro, czyli suma haseł `final_keywords` pobranych w jednym zapytaniu (zeby miec wspólna skala z GT),
    tak jak jest używany w modelu (FeatureSet.WITH_GT_MACRO).
    - corr_with_signal mówi, jak blisko sygnału makro jest dane hasło- korelacja
    Pearsona między tygodniową serią SVI danego hasła a serią sygnału makro, czyli sumą 'stock market' i 'S&P 500' z jednego zapytania
    """

    def pull(keywords: List[str], signal: bool) -> List[pd.Series]:
        out: List[pd.Series] = []
        for r in range(n_repeats):
            try:
                if signal:
                    out.append(
                        acq.fetch_gt_series(
                            keywords, "MACRO", repeat=r, allow_network=allow_network
                        ).astype(float)
                    )
                else:
                    out.append(
                        acq.fetch_gt_frame(
                            keywords, repeat=r, allow_network=allow_network
                        )
                        .iloc[:, 0]
                        .astype(float)
                    )
            except DataAcquisitionError as exc:
                logger.warning(
                    "[MACRO] %s (powt. %d) niedostępne: %s", keywords, r, exc
                )
                break
        return out

    signal_pulls = pull(list(final_keywords), signal=True)
    signal_ref = _train_weeks(signal_pulls[0], train_frac) if signal_pulls else None

    cands = list(dict.fromkeys(list(MACRO_CANDIDATES) + list(final_keywords)))
    items = [(kw, "macro_word", [kw]) for kw in cands]
    items.append((" + ".join(final_keywords), "macro_signal", None))
    rows = []
    for kw, kind, _ in items:
        s_list = signal_pulls if kind == "macro_signal" else pull([kw], signal=False)
        row = {
            "ticker": "MACRO",
            "keyword": kw,
            "variant_type": kind,
            "in_signal": kind == "macro_signal" or kw in final_keywords,
            "n_pulls": len(s_list),
        }
        if not s_list:
            row["status"] = "NO_DATA"
            rows.append(row)
            continue
        s0 = _train_weeks(s_list[0], train_frac)
        row.update(svi_quality(s0))
        row["reliability"] = (
            reliability(s0, _train_weeks(s_list[1], train_frac))
            if len(s_list) > 1
            else float("nan")
        )
        row["corr_with_context"] = float("nan")
        row["corr_context_robust"] = float("nan")
        row["corr_with_signal"] = (
            reliability(s0, signal_ref) if signal_ref is not None else float("nan")
        )
        row["n_weeks_train"] = len(s0)
        rows.append(row)
    return pd.DataFrame(rows)


def macro_correlations(
    acq: DataAcquisition,
    final_keywords: List[str],
    train_frac: float = 0.70,
    allow_network: bool = True,
) -> pd.DataFrame:
    """Korelacje parami między hasłami makro (kandydaci + hasła sygnału)
    Służy do oceny, czy dwa hasła niosą różną informację, czy jedno dubluje drugie.
    Nie dotyczy związku z ceną (testowany dopiero w modelach). Korzysta z tych samych plików co
    investigate_macro, więc bez nowych zapytań do Google."""
    cands = list(dict.fromkeys(list(MACRO_CANDIDATES) + list(final_keywords)))
    series: Dict[str, pd.Series] = {}
    for kw in cands:
        try:
            frame = acq.fetch_gt_frame([kw], repeat=0, allow_network=allow_network)
            series[kw] = _train_weeks(frame.iloc[:, 0].astype(float), train_frac)
        except DataAcquisitionError as exc:
            logger.warning("[MACRO] %r niedostępne: %s", kw, exc)
    rows = []
    names = list(series)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            rows.append(
                {
                    "keyword_a": a,
                    "keyword_b": b,
                    "oba_w_sygnale": a in final_keywords and b in final_keywords,
                    "pearson": reliability(series[a], series[b]),
                    "spearman_diff": robust_corr(series[a], series[b]),
                }
            )
    return pd.DataFrame(rows)


def macro_report(evaluated: pd.DataFrame) -> pd.DataFrame:
    """Tabela wyniku dla haseł makro: status OK / NIE SPEŁNIA / BRAK DANYCH i lista zawodzących progów."""
    rows = []
    for r in evaluated.itertuples():
        if getattr(r, "status", None) == "NO_DATA":
            rows.append(
                {
                    "keyword": r.keyword,
                    "typ": r.variant_type,
                    "w_sygnale": r.in_signal,
                    "status": "BRAK DANYCH",
                    "failing_criteria": "brak pobrania",
                }
            )
            continue
        if pd.isna(r.reliability) and int(r.n_pulls) < 2:
            rows.append(
                {
                    "keyword": r.keyword,
                    "typ": r.variant_type,
                    "w_sygnale": r.in_signal,
                    "status": "BRAK DANYCH",
                    "failing_criteria": "brak 2. pobrania (reliability)",
                }
            )
            continue
        failing = [
            n
            for n, ok in [
                ("zero_share", r.ok_zero_share),
                ("median_to_peak", r.ok_median_to_peak),
                ("reliability", r.ok_reliability),
            ]
            if not bool(ok)
        ]
        rows.append(
            {
                "keyword": r.keyword,
                "typ": r.variant_type,
                "w_sygnale": r.in_signal,
                "status": "OK" if r.eligible else "NIE SPEŁNIA",
                "failing_criteria": ",".join(failing) or "-",
                "zero_share": r.zero_share,
                "median_to_peak": r.median_to_peak,
                "reliability": r.reliability,
                "corr_with_signal": r.corr_with_signal,
            }
        )
    return pd.DataFrame(rows)


def fetch_related_queries(config, keyword: str) -> Optional[pd.DataFrame]:
    """Powiązane zapytania (top) do manualnej weryfikacji wieloznaczności słowa
    (np. 'SPY' vs szpieg), przy bledzie zwraca None."""
    try:
        from pytrends.request import TrendReq

        pt = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
        pt.build_payload(
            [keyword],
            timeframe=f"{config.start_date} {config.end_date}",
            geo=config.gt_geo,
        )
        top = pt.related_queries().get(keyword, {}).get("top")
        return None if top is None else top.assign(keyword=keyword)
    except Exception as exc:  # sieć/limit/zmiana API
        logger.warning("related_queries(%r) niedostępne: %s", keyword, exc)
        return None


def fetch_related_cached(
    config, keywords, cache_dir: str, pause_s: float = 30.0, max_failures: int = 2
) -> pd.DataFrame:
    """Powiązane zapytania dla listy słów: z pauzą, z cache na dysku (jeden plik na słowo)
    po `max_failures` kolejnych błędach (zwykle 429) przerywa, a to, co już pobrane, zostaje zapisane.
    Kolejne uruchomienie dociąga tylko brakujące słowa."""
    import time

    from src.data.acquisition import slugify

    os.makedirs(cache_dir, exist_ok=True)
    frames, failures = [], 0
    for i, kw in enumerate(keywords):
        path = os.path.join(cache_dir, f"{slugify([kw])}.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
            continue
        if failures >= max_failures:
            logger.warning(
                "related_queries: przerwano po %d błędach - uruchom ponownie później",
                failures,
            )
            break
        df = fetch_related_queries(config, kw)
        if df is None:
            failures += 1
        else:
            failures = 0
            df.to_csv(path, index=False)
            frames.append(df)
        if i < len(keywords) - 1:
            time.sleep(pause_s)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def validate_final_choice(
    evaluated: pd.DataFrame, final: Dict[str, str]
) -> pd.DataFrame:
    """Weryfikacja ostatecznego wyboru słów (gt_dictionary.ASSET_KEYWORDS) na danych.

    Dla każdego aktywa: czy wybrane słowo spełnia progi jakości, które z progów
    zawodzą i jaki jest zamiennik (pierwszy kwalifikujący się wariant wg priorytetu;
    'NO_GT' gdy żaden). Status: OK / ZAMIENIĆ (zamiennik istnieje) / NO_GT."""
    rows = []
    for ticker, kw in final.items():
        g = evaluated[evaluated["ticker"] == ticker]
        if g.empty:
            continue
        row = g[g["keyword"] == kw]
        alt = g[g["eligible"] & (g["keyword"] != kw)].sort_values(
            ["priority", "reliability"], ascending=[True, False]
        )
        if row.empty or row.iloc[0].get("status") == "NO_DATA":
            rows.append(
                {
                    "ticker": ticker,
                    "final_keyword": kw,
                    "status": "BRAK DANYCH",
                    "failing_criteria": "brak pobrania",
                    "replacement": alt.iloc[0]["keyword"] if len(alt) else None,
                }
            )
            continue
        r = row.iloc[0]
        # reliability wymaga dwoch niezależnych pobrań; bledu 429 nie wolno tego raportować jako NO_GT
        if pd.isna(r.get("reliability")) and int(r.get("n_pulls", 0)) < 2:
            rows.append(
                {
                    "ticker": ticker,
                    "final_keyword": kw,
                    "status": "BRAK DANYCH",
                    "failing_criteria": "brak 2. pobrania (reliability)",
                    "replacement": None,
                }
            )
            continue
        failing = [
            name
            for name, col in [
                ("zero_share", "ok_zero_share"),
                ("median_to_peak", "ok_median_to_peak"),
                ("reliability", "ok_reliability"),
                ("jednoznaczność", "ok_unambiguous"),
            ]
            if not bool(r[col])
        ]
        if r["eligible"]:
            status, repl = "OK", None
        elif len(alt):
            status, repl = "ZAMIENIĆ", alt.iloc[0]["keyword"]
        else:
            status, repl = "NO_GT", None
        rows.append(
            {
                "ticker": ticker,
                "final_keyword": kw,
                "status": status,
                "failing_criteria": ",".join(failing) or "-",
                "replacement": repl,
                "zero_share": r.get("zero_share"),
                "median_to_peak": r.get("median_to_peak"),
                "reliability": r.get("reliability"),
                "corr_with_context": r.get("corr_with_context"),
                "corr_context_robust": r.get("corr_context_robust"),
            }
        )
    return pd.DataFrame(rows)
