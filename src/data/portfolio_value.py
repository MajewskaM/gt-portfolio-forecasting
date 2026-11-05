"""
Oblicza historyczną wartość portfela metodą buy-and-hold.

ohlcv - dane OHLCV dla poszczególnych aktywów
assets - ticker i jego udział w portfelu
initial_capital - początkowy kapitał
"""
import pandas as pd


def calculate_portfolio_value(ohlcv, assets, initial_capital=100_000):


    prices = pd.concat(
        {ticker: ohlcv[ticker]["Close"] for ticker in assets},
        axis=1
    ).dropna()

    if prices.empty:
        raise ValueError("Brak wspólnych danych cenowych dla aktywów.")

    shares = {}

    for ticker, weight in assets.items():
        shares[ticker] = initial_capital * weight / prices[ticker].iloc[0]

    portfolio_value = sum(
        prices[ticker] * shares[ticker]
        for ticker in assets
    )

    portfolio_value.name = "Portfolio_Value"

    return portfolio_value
