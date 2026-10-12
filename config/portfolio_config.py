"""
Centralna konfiguracja projektu: jedna klasa PortfolioConfig opisuje WSZYSTKIE osie
eksperymentu

Trafiają tu tylko parametry wspólne dla wszystkich modeli (dane, portfel,
split, GT). Parametry konkretnej rodziny modeli (np. przestrzeń przeszukiwania
ARIMA, hiperparametry LSTM) leżą przy samym modelu w src/models/, żeby dodanie
nowego modelu nie rozrastało tego pliku.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List

from src.data.portfolio_presets import default_single_ticker, get_preset


class FeatureSet(enum.Enum):
    """
    Oś eksperymentu: jakie dane Google Trends trafiają do modelu.

    BASELINE       - brak sygnału GT
    WITH_GT_MACRO  - jeden wspólny sygnał dla całego portfela (suma SVI dwóch
                     ogólnorynkowych haseł); podejście top-down
    WITH_GT_ASSET  - osobny sygnał GT na aktywo (słowa z gt_dictionary.py);
                     podejście bottom-up
    WITH_GT_SECTOR - (rozszerzenie, jeszcze nie zaimplementowane)
    WITH_GT_HYBRID - (rozszerzenie, jeszcze nie zaimplementowane) MACRO + ASSET
    """

    BASELINE = "baseline"
    WITH_GT_MACRO = "with_gt_macro"
    WITH_GT_ASSET = "with_gt_asset"
    WITH_GT_SECTOR = "with_gt_sector"
    WITH_GT_HYBRID = "with_gt_hybrid"


class TargetMode(enum.Enum):
    """Cel predykcji."""

    PORTFOLIO = "portfolio"  # y_t = wartość dolarowa portfela
    SINGLE_ASSET = "single_asset"  # y_t = Close pojedynczej spółki/ETF-u


class AggregationStrategy(enum.Enum):
    """
    Jak dojść do prognozy portfela.

    DIRECT    - jeden model, prognoza wartości portfela wprost.
    BOTTOM_UP - N niezależnych modeli (po jednym na aktywo), prognozy łączone
                przez shares (te same, które definiują prawdziwą wartość portfela).
                Ignoruje korelacje między aktywami - hipoteza badana od Step 2 (ARIMA).
    """

    DIRECT = "direct"
    BOTTOM_UP = "bottom_up"


class FeatureRichness(enum.Enum):
    """
    Zestaw cech per aktywo (istotny dla modeli konsumujących macierz cech).

    MINIMAL - tylko Open i Close (2 cechy/aktywo)
    FULL    - pełny rejestr z FeatureEngineer (9 cech/aktywo)
    """

    MINIMAL = "minimal"
    FULL = "full"


@dataclass
class PortfolioConfig:
    # Skład portfela: domyślnie preset 'diversified_9'
    assets: Dict[str, float] = field(
        default_factory=lambda: get_preset("diversified_9")
    )

    # Aktywo dla TargetMode.SINGLE_ASSET (musi należeć do portfela).
    single_ticker: str = ""

    # Etykieta portfela do rejestru wyników
    portfolio_id: str = "diversified_9"

    # Kapitał początkowy do wyliczenia stałej liczby jednostek (buy-and-hold)
    initial_capital: float = 100_000.0

    # 'manual' | 'equal' | 'min_variance'
    weighting_method: str = "manual"

    # Prognoza portfela: DIRECT vs BOTTOM_UP (tylko TargetMode.PORTFOLIO)
    aggregation_strategy: AggregationStrategy = AggregationStrategy.DIRECT

    # Bogactwo cech per aktywo - niezależne od feature_set
    feature_richness: FeatureRichness = FeatureRichness.FULL

    # Zakres dat. Google Trends zwraca dane TYGODNIOWE dla okien
    # 9 miesięcy-5 lat, a MIESIĘCZNE dla okien dłuższych niż 5 lat -
    # wydłużenie zakresu powyżej 5 lat zmienia więc rozdzielczość sygnału GT.
    start_date: str = "2019-01-01"
    end_date: str = "2024-01-01"

    # Google Trends
    feature_set: FeatureSet = FeatureSet.BASELINE
    gt_macro_keywords: List[str] = field(
        default_factory=lambda: ["stock market", "S&P 500"]
    )
    gt_geo: str = "US"
    # pauza między zapytaniami do Google (ochrona przed błędem 429)
    gt_request_pause_s: float = 15.0

    # cache OHLCV i dane GT (zbiór danych, bo kolejne pobrania GT różnią się szumem próbkowania)
    cache_dir: str = "data/cache"
    gt_dir: str = "data/gt"

    # Okno wsteczne (ile dni widzi model)
    window_size: int = 10

    # Podział chronologiczny: train/val/test
    train_frac: float = 0.70
    val_frac: float = 0.15

    # Poziom istotności testów statystycznych
    alpha: float = 0.05

    # Reprodukowalność
    random_seed: int = 42

    output_dir: str = "results"

    # podstawowe testy zabezpieczajace
    def __post_init__(self) -> None:
        if not self.single_ticker:
            self.single_ticker = default_single_ticker(self.assets)
        if self.single_ticker not in self.assets:
            raise ValueError(
                f"single_ticker={self.single_ticker!r} nie jest w assets "
                f"({list(self.assets.keys())})."
            )
        total_w = sum(self.assets.values())
        if abs(total_w - 1.0) > 1e-6:
            raise ValueError(f"Wagi assets sumują się do {total_w}, nie do 1.0.")
        if self.train_frac + self.val_frac >= 1.0:
            raise ValueError(
                f"train_frac + val_frac = {self.train_frac + self.val_frac} "
                f"musi być < 1.0, żeby zostało coś na test."
            )
        if self.start_date >= self.end_date:
            raise ValueError("start_date musi być wcześniejsza niż end_date.")

    @property
    def test_frac(self) -> float:
        return 1.0 - self.train_frac - self.val_frac

    @property
    def tickers(self) -> List[str]:
        return list(self.assets.keys())

    @property
    def n_assets(self) -> int:
        return len(self.assets)
