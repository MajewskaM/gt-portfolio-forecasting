"""Testy pobierania danych (z podstawionymi modułami yfinance/pytrends)
- Ponawianie zapytań: czy po nieudanym pobraniu kod próbuje ponownie, zwłaszcza dla pojedynczych tickerów.
- Niepełne dane z Google Trends: czy kod odrzuca niepełny punkt i zapisuje wynik w cache.
- Błąd DNS i całkowita niepowodzenie pobierania: czy kod kończy się szybko, zamiast czekać na kolejne ponowienia.

"""

from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.data import acquisition as acq_mod
from src.data.acquisition import DataAcquisition, DataAcquisitionError
from tests.helpers import synthetic_config

FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def _ohlcv(n=300, tz=None, seed=0):
    idx = pd.bdate_range("2019-01-01", periods=n, tz=tz)
    c = 100 + np.cumsum(np.random.default_rng(seed).normal(0, 1, n))
    return pd.DataFrame(
        {"Open": c, "High": c + 1, "Low": c - 1, "Close": c, "Volume": 1e6}, index=idx
    )


def test_yfinance_retries_failed_tickers_individually(tmp_path, monkeypatch):
    cfg = synthetic_config(str(tmp_path))
    calls = []

    def download(tickers, **kw):
        calls.append(tickers)
        if isinstance(tickers, list):  # zbiorczo: tylko pierwszy ticker się udaje
            return pd.concat({tickers[0]: _ohlcv(seed=1)}, axis=1)
        return pd.concat({tickers: _ohlcv(seed=2)}, axis=1)

    monkeypatch.setitem(
        sys.modules, "yfinance", types.SimpleNamespace(download=download)
    )
    monkeypatch.setattr(acq_mod.time, "sleep", lambda s: None)
    out = DataAcquisition(cfg).fetch_ohlcv_for(["AAA", "BBB"])
    assert set(out) == {"AAA", "BBB"} and len(calls) == 2
    # drugie wywołanie korzysta z cache
    calls.clear()
    DataAcquisition(cfg).fetch_ohlcv_for(["AAA", "BBB"])
    assert calls == []


def test_yfinance_total_failure_raises_clear_error(tmp_path, monkeypatch):
    cfg = synthetic_config(str(tmp_path))
    monkeypatch.setitem(
        sys.modules,
        "yfinance",
        types.SimpleNamespace(download=lambda *a, **k: pd.DataFrame()),
    )
    monkeypatch.setattr(acq_mod.time, "sleep", lambda s: None)
    with pytest.raises(DataAcquisitionError):
        DataAcquisition(cfg).fetch_ohlcv_for(["AAA"])


def _fake_pytrends(monkeypatch, fail_times=0, error="429 Too Many Requests"):
    state = {"attempts": 0, "init_kwargs": [], "fail_times": fail_times}
    weeks = pd.date_range("2019-01-06", periods=60, freq="W-SUN")

    class FakeReq:
        def __init__(self, **kw):
            state["init_kwargs"].append(kw)

        def build_payload(self, kws, timeframe, geo):
            self.kws = kws

        def interest_over_time(self):
            state["attempts"] += 1
            if state["attempts"] <= state["fail_times"]:
                raise RuntimeError(error)
            df = pd.DataFrame(
                {k: np.linspace(10, 90, len(weeks)) for k in self.kws}, index=weeks
            )
            df["isPartial"] = False
            df.iloc[-1, df.columns.get_loc("isPartial")] = True
            return df

    mod = types.ModuleType("pytrends.request")
    mod.TrendReq = FakeReq
    monkeypatch.setitem(sys.modules, "pytrends", types.ModuleType("pytrends"))
    monkeypatch.setitem(sys.modules, "pytrends.request", mod)
    monkeypatch.setattr(acq_mod.time, "sleep", lambda s: None)
    return state


def test_pytrends_retries_drops_partial_and_caches(tmp_path, monkeypatch):
    cfg = synthetic_config(
        str(tmp_path), start_date="2019-01-01", end_date="2020-03-01"
    )
    state = _fake_pytrends(monkeypatch, fail_times=2)
    acq = DataAcquisition(cfg)
    frame = acq.fetch_gt_frame(["MSFT stock"])
    assert (
        state["attempts"] == 3 and len(frame) == 59
    )  # ostatni, niepełny punkt usunięty
    assert (
        "retries" not in state["init_kwargs"][-1]
    )  # omija błąd method_whitelist/urllib3 2.x
    state["attempts"], state["fail_times"] = 0, 0
    acq.fetch_gt_frame(["MSFT stock"])  # z pliku
    assert state["attempts"] == 0
    # niezależne powtórzenie = osobny plik i osobne zapytanie
    acq.fetch_gt_frame(["MSFT stock"], repeat=1)
    assert state["attempts"] == 1


def test_pytrends_dns_failure_fails_fast(tmp_path, monkeypatch):
    cfg = synthetic_config(str(tmp_path))
    state = _fake_pytrends(
        monkeypatch, fail_times=99, error="NameResolutionError: Could not resolve"
    )
    with pytest.raises(DataAcquisitionError, match="manual"):
        DataAcquisition(cfg).fetch_gt_frame(["x"])
    assert state["attempts"] == 1
