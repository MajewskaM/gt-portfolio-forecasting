"""
Uniwersum kandydatów na aktywa portfeli - dobor aktywow.

Metadane (klasa aktywów, sektor, rodzaj) służą do:
- ograniczeń pokrycia przy doborze portfela (np. min. 1 aktywo na klasę),
- reguły wyboru aktywa dla TargetMode.SINGLE_ASSET (spółka, nie ETF),
- generowania wariantów słów kluczowych Google Trends (nazwa spółki).

Sektory wg klasyfikacji GICS (stan na 2018+: Alphabet należy do Communication
Services, nie do Information Technology).
Lista jest szersza niż jakikolwiek portfel - zawiera aktywa, które odpadną w filtrach (płynność, GT).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class AssetInfo:
    ticker: str
    name: str  # pełna nazwa instrumentu
    short_name: str  # nazwa potoczna do wyszukiwania (wariant GT)
    kind: str  # "stock" | "etf"
    asset_class: str  # "equity" | "bond" | "commodity" | "real_estate"
    sector: str  # sektor GICS albo "broad" / "gold" / "oil" / "treasuries" itp.
    region: str = "US"


_ASSETS: List[AssetInfo] = [
    # --- pojedyncze spółki (USA) ---
    AssetInfo(
        "MSFT",
        "Microsoft Corp.",
        "Microsoft",
        "stock",
        "equity",
        "Information Technology",
    ),
    AssetInfo(
        "AAPL", "Apple Inc.", "Apple", "stock", "equity", "Information Technology"
    ),
    AssetInfo(
        "GOOGL",
        "Alphabet Inc. (Class A)",
        "Google",
        "stock",
        "equity",
        "Communication Services",
    ),
    AssetInfo(
        "AMZN", "Amazon.com Inc.", "Amazon", "stock", "equity", "Consumer Discretionary"
    ),
    AssetInfo(
        "JPM", "JPMorgan Chase & Co.", "JPMorgan", "stock", "equity", "Financials"
    ),
    AssetInfo(
        "JNJ",
        "Johnson & Johnson",
        "Johnson & Johnson",
        "stock",
        "equity",
        "Health Care",
    ),
    AssetInfo("XOM", "Exxon Mobil Corp.", "Exxon", "stock", "equity", "Energy"),
    AssetInfo(
        "PG",
        "Procter & Gamble Co.",
        "Procter & Gamble",
        "stock",
        "equity",
        "Consumer Staples",
    ),
    AssetInfo(
        "KO", "Coca-Cola Co.", "Coca-Cola", "stock", "equity", "Consumer Staples"
    ),
    AssetInfo("WMT", "Walmart Inc.", "Walmart", "stock", "equity", "Consumer Staples"),
    # --- ETF-y: akcje ---
    AssetInfo("SPY", "SPDR S&P 500 ETF", "S&P 500", "etf", "equity", "broad"),
    AssetInfo(
        "QQQ", "Invesco QQQ (Nasdaq-100)", "Nasdaq 100", "etf", "equity", "broad-growth"
    ),
    AssetInfo(
        "DIA",
        "SPDR Dow Jones Industrial Average ETF",
        "Dow Jones",
        "etf",
        "equity",
        "broad",
    ),
    AssetInfo(
        "IWM", "iShares Russell 2000 ETF", "Russell 2000", "etf", "equity", "small-cap"
    ),
    AssetInfo(
        "EFA",
        "iShares MSCI EAFE ETF",
        "MSCI EAFE",
        "etf",
        "equity",
        "broad",
        "Developed ex-US",
    ),
    AssetInfo(
        "EEM",
        "iShares MSCI Emerging Markets ETF",
        "emerging markets",
        "etf",
        "equity",
        "broad",
        "Emerging",
    ),
    AssetInfo(
        "XLU", "Utilities Select Sector SPDR", "utilities", "etf", "equity", "Utilities"
    ),
    # --- ETF-y: obligacje ---
    AssetInfo(
        "TLT",
        "iShares 20+ Year Treasury Bond ETF",
        "treasury bonds",
        "etf",
        "bond",
        "treasuries",
    ),
    AssetInfo(
        "IEF",
        "iShares 7-10 Year Treasury Bond ETF",
        "treasury bonds",
        "etf",
        "bond",
        "treasuries",
    ),
    AssetInfo(
        "AGG",
        "iShares Core U.S. Aggregate Bond ETF",
        "bond market",
        "etf",
        "bond",
        "aggregate",
    ),
    AssetInfo(
        "LQD",
        "iShares iBoxx $ Investment Grade Corporate Bond ETF",
        "corporate bonds",
        "etf",
        "bond",
        "credit",
    ),
    # --- ETF-y: surowce / nieruchomości ---
    AssetInfo("GLD", "SPDR Gold Shares", "gold", "etf", "commodity", "gold"),
    AssetInfo("SLV", "iShares Silver Trust", "silver", "etf", "commodity", "silver"),
    AssetInfo("USO", "United States Oil Fund", "oil", "etf", "commodity", "oil"),
    AssetInfo(
        "VNQ", "Vanguard Real Estate ETF", "real estate", "etf", "real_estate", "REIT"
    ),
]

UNIVERSE: Dict[str, AssetInfo] = {a.ticker: a for a in _ASSETS}


def get_info(ticker: str) -> AssetInfo:
    if ticker not in UNIVERSE:
        raise KeyError(f"Ticker {ticker!r} nie ma opisu w src/data/universe.py")
    return UNIVERSE[ticker]


def tickers_of_kind(kind: str) -> List[str]:
    return [t for t, a in UNIVERSE.items() if a.kind == kind]
