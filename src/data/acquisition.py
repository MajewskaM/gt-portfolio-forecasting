"""
Pobieranie danych: OHLCV (Yahoo Finance przez yfinance) i Google Trends.

OHLCV jest cache'owane per ticker w config.cache_dir - przełączanie
presetów portfeli używa ponownie cache dla tickerów wspólnych.

GOOGLE TRENDS - kolejność źródeł dla jednego zapytania:
  1. plik w config.gt_dir  (wcześniejsze pobranie; wczesniej zapisany
     zbiór danych - kolejne pobrania GT różnią się szumem próbkowania Google, więc
     bez 'zamrożenia' eksperymenty nie byłyby powtarzalne),
  2. ręczny eksport z interfejsu trends.google.com w config.gt_dir/manual/<slug>.csv
     (zalecana ścieżka awaryjna, gdy Google zablokuje zapytania),
  3. pytrends (sieć), z pauzą między zapytaniami; wynik zapisywany jak w pkt 1.

Uwaga: Google zwraca dane dzienne dla okien < ~9 miesięcy,
tygodniowe dla 9 miesięcy - 5 lat i miesięczne dla okien dłuższych niż 5 lat.
"""

from __future__ import annotations

import io
import logging
import os
import re
import time
from typing import Dict, List, Optional

import pandas as pd

from config.portfolio_config import FeatureSet, PortfolioConfig

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


class DataAcquisitionError(RuntimeError):
    """Error wyrzucany, gdy dane (OHLCV lub GT) są niedostępne."""


def slugify(keywords: List[str]) -> str:
    """Nazwa pliku z listy słów kluczowych: ['S&P 500'] -> 's_p_500'."""
    raw = "+".join(k.lower() for k in keywords)
    return re.sub(r"[^a-z0-9+]+", "_", raw).strip("_")


def infer_resolution(index: pd.DatetimeIndex) -> str:
    """'daily' | 'weekly' | 'monthly' | 'unknown' wg mediany odstępów między punktami.
    Google zwraca dane dzienne (okna do ok. 9 mies.), tygodniowe (9 mies.-5 lat)
    i miesięczne (powyżej 5 lat); logika wyrównania i opóźnienia zakłada
    okres równy odstępowi między punktami."""
    if len(index) < 3:
        return "unknown"
    gap = pd.Series(index).diff().dt.days.median()
    if gap <= 1.5:
        return "daily"
    if 6 <= gap <= 8:
        return "weekly"
    if 27 <= gap <= 32:
        return "monthly"
    return "unknown"


def parse_trends_export_frame(source) -> pd.DataFrame:
    """Parsuje plik CSV wyeksportowany z interfejsu Google Trends.

    Format eksportu: kilka linii nagłówkowych ("Category: ..."), pusta linia, potem
    nagłówek 'Week,<słowo>: (United States)' (lub 'Day'/'Month'/'Time') i wiersze
    'YYYY-MM-DD,wartość'. Wartość '<1' oznacza < 1% szczytu - zamieniamy na 0.5.
    Zwraca DataFrame z jedną kolumną na słowo kluczowe.
    """
    text = (
        source.read()
        if hasattr(source, "read")
        else open(source, encoding="utf-8-sig").read()
    )
    lines = text.splitlines()
    header_idx = next(
        (i for i, ln in enumerate(lines) if re.match(r"^(Week|Day|Month|Time)\b", ln)),
        None,
    )
    if header_idx is None:
        raise ValueError("Nie znaleziono nagłówka 'Week/Day/Month' w eksporcie GT.")
    body = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(body), index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.apply(
        lambda col: pd.to_numeric(
            col.astype(str).str.replace("<1", "0.5", regex=False), errors="coerce"
        )
    )
    df.columns = [re.sub(r":\s*\(.*\)\s*$", "", str(c)).strip() for c in df.columns]
    return df.sort_index()


def parse_trends_export_csv(source) -> pd.Series:
    """Jak parse_trends_export_frame, ale kilka słów jest sumowane w jeden sygnał
    (tak samo jak w fetch_gt_series)."""
    return parse_trends_export_frame(source).sum(axis=1)


class DataAcquisition:
    def __init__(self, config: PortfolioConfig) -> None:
        self.config = config
        os.makedirs(config.cache_dir, exist_ok=True)
        os.makedirs(config.gt_dir, exist_ok=True)

    def fetch_ohlcv(self) -> Dict[str, pd.DataFrame]:
        """Zwraca {ticker: DataFrame(Open, High, Low, Close, Volume)}.

        Ceny są skorygowane o splity i dywidendy (auto_adjust=True), więc wartość
        portfela odpowiada zwrotowi całkowitemu. Cache sprawdzany osobno per ticker."""
        return self.fetch_ohlcv_for(self.config.tickers)

    # OHLCV (Yahoo Finance)
    def fetch_ohlcv_for(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        result: Dict[str, pd.DataFrame] = {}
        missing: List[str] = []
        for ticker in tickers:
            cached = self._load_ohlcv_cache(self._ohlcv_cache_path(ticker))
            if cached is not None:
                result[ticker] = cached
            else:
                missing.append(ticker)

        if not missing:
            logger.info("OHLCV: wszystkie %d aktywów z cache.", len(tickers))
            return result

        logger.info(
            "OHLCV: %d/%d z cache, pobieram brakujące: %s",
            len(result),
            len(tickers),
            missing,
        )
        try:
            frames = self._download_yf(missing)
        except Exception as exc:
            raise DataAcquisitionError(f"Nie udało się pobrać OHLCV: {exc}") from exc
        failed = [t for t in missing if t not in frames]
        if failed:
            raise DataAcquisitionError(f"Nie udało się pobrać OHLCV dla: {failed}")
        for ticker, df in frames.items():
            result[ticker] = df
            self._save_ohlcv_cache(self._ohlcv_cache_path(ticker), df)
        return result

    def _download_yf(
        self, tickers: List[str], attempts: int = 3
    ) -> Dict[str, pd.DataFrame]:
        """Pobranie zbiorcze, a dla tickerów, które się nie udały - pojedyncze próby
        z rosnącą pauzą (poniewaz podobnie jak GT, Yahoo bywa ograniczający przy większej liczbie zapytań).
        """
        import yfinance as yf

        out: Dict[str, pd.DataFrame] = {}
        kw = dict(
            start=self.config.start_date,
            end=self.config.end_date,
            auto_adjust=True,
            progress=False,
        )
        todo = list(tickers)
        for attempt in range(attempts):
            if not todo:
                break
            try:
                raw = (
                    yf.download(todo, group_by="ticker", **kw) if attempt == 0 else None
                )
            except Exception as exc:
                logger.warning("yf.download(%s) błąd: %s", todo, exc)
                raw = None
            for t in list(todo):
                try:
                    if raw is None:
                        raw_t = yf.download(t, group_by="ticker", **kw)
                        out[t] = self.extract_ticker_frame(raw_t, t)
                    else:
                        out[t] = self.extract_ticker_frame(raw, t)
                    todo.remove(t)
                except Exception as exc:
                    logger.warning(
                        "OHLCV %s (próba %d/%d): %s", t, attempt + 1, attempts, exc
                    )
            if todo and attempt < attempts - 1:
                time.sleep(2.0 * (attempt + 1))
        return out

    @staticmethod
    def extract_ticker_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Wyciąga OHLCV jednego tickera z wyniku yf.download."""
        cols = raw.columns
        if isinstance(cols, pd.MultiIndex):
            if ticker in cols.get_level_values(0):
                df = raw[ticker]
            elif ticker in cols.get_level_values(1):
                df = raw.xs(ticker, axis=1, level=1)
            else:
                raise KeyError(f"Brak tickera {ticker} w danych z Yahoo Finance.")
        else:
            df = raw
        df = df[OHLCV_COLUMNS].copy()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:  # yfinance potrafi zwrócić indeks ze strefą czasową
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()
        df.index.name = "Date"
        df = df.dropna()
        if df.empty:
            raise ValueError(f"Puste dane OHLCV dla {ticker}.")
        return df

    # Google Trends
    def gt_cache_path(self, keywords: List[str], repeat: int = 0) -> str:
        c = self.config
        suffix = f"__r{repeat}" if repeat else ""
        name = (
            f"{slugify(keywords)}__{c.gt_geo}__{c.start_date}__{c.end_date}{suffix}.csv"
        )
        return os.path.join(c.gt_dir, name)

    def gt_manual_path(self, keywords: List[str]) -> str:
        return os.path.join(self.config.gt_dir, "manual", f"{slugify(keywords)}.csv")

    def fetch_gt_frame(
        self, keywords: List[str], repeat: int = 0, allow_network: bool = True
    ) -> pd.DataFrame:
        """SVI per słowo kluczowe (kolumny = słowa). Słowa w jednym zapytaniu mają
        wspólną skalę 0-100 (normalizacja do maksimum całego zapytania), więc
        wolumeny słów z jednego zapytania są porównywalne, a z różnych - nie.

        repeat>0 wymusza niezależne pobranie - do pomiaru szumu
        próbkowania GT (test-retest)."""
        cache = self.gt_cache_path(keywords, repeat)
        df = self._load_gt_file(cache)
        if df is not None:
            logger.info("GT%s: z pliku %s", keywords, cache)
            return df

        manual = self.gt_manual_path(keywords)
        if repeat == 0 and os.path.exists(manual):
            logger.info("GT%s: z ręcznego eksportu %s", keywords, manual)
            return self._crop(parse_trends_export_frame(manual))

        if not allow_network:
            raise DataAcquisitionError(
                f"Brak danych GT dla {keywords} (szukano: {cache}, {manual})."
            )
        raw = self._pytrends_interest_over_time(keywords)
        # ostatni, niepełny punkt (isPartial=True) jest niewiarygodny
        if "isPartial" in raw.columns:
            raw = raw[~raw["isPartial"].astype(bool)].drop(columns=["isPartial"])
        raw.index = pd.to_datetime(raw.index)
        res = infer_resolution(raw.index)
        if res != "weekly":
            logger.warning(
                "GT%s: rozdzielczość '%s' (oczekiwano 'weekly' dla okna 9 mies.-5 lat)",
                keywords,
                res,
            )
        if (raw.std() == 0).any():
            logger.warning("GT%s: stała seria (brak wolumenu zapytań).", keywords)
        self._save_gt_file(cache, raw)
        time.sleep(self.config.gt_request_pause_s)
        return raw

    def _pytrends_interest_over_time(
        self, keywords: List[str], attempts: int = 4
    ) -> pd.DataFrame:
        """Zapytanie do Google z własnym ponawianiem (wykładnicza pauza przy 429/błędach)"""
        manual = self.gt_manual_path(keywords)
        last: Optional[Exception] = None
        # bezpiecznik: po 2 kolejnych słowach zablokowanych przez 429 nie ponawiamy
        # kolejne próby tylko wydłużają blokadę; reszta jest pomijana, a wynik zostaje w cache
        if getattr(self, "_gt_blocked", 0) >= 2:
            raise DataAcquisitionError(
                f"Pominięto GT dla {keywords}: Google zwraca 429 (limit zapytań). Awaryjnie wyeksportuj CSV z trends.google.com i zapisz jako {manual}"
            )
        for k in range(attempts):
            try:
                from pytrends.request import TrendReq

                logger.info(
                    "Pobieranie Google Trends (pytrends) dla %s, próba %d/%d",
                    keywords,
                    k + 1,
                    attempts,
                )
                pt = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
                pt.build_payload(
                    keywords,
                    timeframe=f"{self.config.start_date} {self.config.end_date}",
                    geo=self.config.gt_geo,
                )
                raw = pt.interest_over_time()
                if raw.empty:
                    raise ValueError("Pusta odpowiedź Google Trends.")
                self._gt_blocked = 0
                return raw
            except Exception as exc:
                last = exc
                if "NameResolution" in str(exc) or "Could not resolve" in str(exc):
                    break  # brak sieci/DNS - ponawianie tez nie ma sensu
                wait = self.config.gt_request_pause_s * (2**k)
                logger.warning("GT%s: %s - pauza %.0fs", keywords, str(exc)[:120], wait)
                if k < attempts - 1:
                    time.sleep(wait)
        if "429" in str(last):
            self._gt_blocked = getattr(self, "_gt_blocked", 0) + 1
        raise DataAcquisitionError(
            f"Nie udało się pobrać GT dla {keywords}: {last}. Awaryjnie wyeksportuj "
            f"CSV z trends.google.com i zapisz jako {manual}"
        ) from last

    def fetch_gt_series(
        self,
        keywords: List[str],
        label: str,
        repeat: int = 0,
        allow_network: bool = True,
    ) -> pd.Series:
        """SVI dla listy słów kluczowych sumowany w jeden sygnał (np. dwa hasła makro)."""
        frame = self.fetch_gt_frame(keywords, repeat, allow_network)
        return frame.sum(axis=1).rename(label)

    def fetch_gt_for_feature_set(self) -> Dict[str, pd.Series]:
        """Zwraca {nazwa_cechy: seria SVI} dla config.feature_set:
        BASELINE -> {} ;
        WITH_GT_MACRO -> {'Trend_SVI_MACRO': ...} ;
        WITH_GT_ASSET -> {'Trend_SVI_<TICKER>': ...} (po jednym na aktywo)."""
        from src.data.gt_dictionary import keywords_for_assets

        fs = self.config.feature_set
        if fs == FeatureSet.BASELINE:
            return {}
        if fs == FeatureSet.WITH_GT_MACRO:
            kws = self.config.gt_macro_keywords
            return {"Trend_SVI_MACRO": self.fetch_gt_series(kws, "Trend_SVI_MACRO")}
        if fs == FeatureSet.WITH_GT_ASSET:
            out: Dict[str, pd.Series] = {}
            for ticker, kw in keywords_for_assets(self.config.tickers).items():
                label = f"Trend_SVI_{ticker}"
                out[label] = self.fetch_gt_series([kw], label)
            return out
        else:
            raise NotImplementedError()

    # pomocnicze funkcje do cache'owania i wczytywania plików OHLCV i GT
    def _crop(self, s):
        return s.loc[self.config.start_date : self.config.end_date]

    def _ohlcv_cache_path(self, ticker: str) -> str:
        c = self.config
        return os.path.join(
            c.cache_dir, f"ohlcv_{ticker}_{c.start_date}_{c.end_date}.csv"
        )

    @staticmethod
    def _load_ohlcv_cache(path: str) -> Optional[pd.DataFrame]:
        if not os.path.exists(path):
            return None
        try:
            return pd.read_csv(path, index_col=0, parse_dates=True)
        except Exception:
            return None

    @staticmethod
    def _save_ohlcv_cache(path: str, df: pd.DataFrame) -> None:
        df.to_csv(path)

    @staticmethod
    def _load_gt_file(path: str) -> Optional[pd.DataFrame]:
        if not os.path.exists(path):
            return None
        try:
            return pd.read_csv(path, index_col=0, parse_dates=True)
        except Exception:
            return None

    @staticmethod
    def _save_gt_file(path: str, frame: pd.DataFrame) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        frame.to_csv(path)
