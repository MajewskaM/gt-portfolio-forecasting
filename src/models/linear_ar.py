
"""
linear_ar.py

Autoregressive AR(p) model for investment portfolio forecasting.

The module implements:
- AR(p) model estimated using Ordinary Least Squares (OLS),
- historical portfolio data acquisition from Yahoo Finance,
- portfolio value calculation using a buy-and-hold strategy,
- chronological train/validation/test split (70% / 15% / 15%),
- one-step-ahead forecasting,
- forecast evaluation using MAE, RMSE and Directional Accuracy.

The model uses historical portfolio values only.
Google Trends data is not used.

Run from the main project directory:

    python -m src.models.linear_ar
"""

import numpy as np
import statsmodels.api as sm
import pandas as pd

from config.portfolio_config import PortfolioConfig, FeatureSet
from src.data.acquisition import DataAcquisition
from src.data.portfolio_value import calculate_portfolio_value


class LinearAR:
    """Autoregressive AR(p) model estimated using OLS."""

    def __init__(self, p=5):
        self.p = p
        self.model = None

    def fit(self, y_train, y_val=None, X_train=None, X_val=None):
        """Fits the model using lagged observations."""

        y = np.asarray(y_train, dtype=float)

        if len(y) <= self.p:
            raise ValueError("Not enough observations to fit AR model.")

        X = np.array([
            y[i-self.p:i][::-1]
            for i in range(self.p, len(y))
        ])

        target = y[self.p:]

        self.model = sm.OLS(target, sm.add_constant(X)).fit()

        return self

    def predict_one_step(self, y_full, start, X_full=None):
        """Predicts the next value using the previous p observations."""

        if self.model is None:
            raise ValueError("Model must be fitted before prediction.")

        y = np.asarray(y_full, dtype=float)

        if start < self.p or start >= len(y):
            raise ValueError("Invalid prediction index.")

        lags = y[start-self.p:start][::-1]
        X = np.r_[1, lags].reshape(1, -1)

        return float(self.model.predict(X)[0])


def run_experiment():
    """Runs the complete AR(p) portfolio forecasting experiment."""

    config = PortfolioConfig(
        portfolio_id="diversified_9",
        feature_set=FeatureSet.BASELINE
    )

    # Download market data
    acquisition = DataAcquisition(config)
    ohlcv = acquisition.fetch_ohlcv()

    # Calculate historical portfolio value
    portfolio = calculate_portfolio_value(
        ohlcv,
        config.assets,
        config.initial_capital
    )

    y = portfolio.to_numpy(dtype=float)

    # Chronological split: 70% / 15% / 15%
    n = len(y)
    train_end = int(n * config.train_frac)
    val_end = int(n * (config.train_frac + config.val_frac))

    y_train = y[:train_end]

    print("Portfolio:", config.portfolio_id)
    print("Observations:", n)
    print("Training:", train_end)
    print("Validation:", val_end - train_end)
    print("Test:", n - val_end)

    # Fit AR(5)
    model = LinearAR(p=5)
    model.fit(y_train)

    # One-step-ahead predictions
    predictions = []

    for i in range(val_end, n):
        predictions.append(
            model.predict_one_step(y, start=i)
        )

    predictions = np.asarray(predictions)
    actual = y[val_end:]

    # Evaluation
    mae = np.mean(np.abs(actual - predictions))
    rmse = np.sqrt(np.mean((actual - predictions) ** 2))

    actual_direction = np.sign(np.diff(actual))
    predicted_direction = np.sign(np.diff(predictions))

    da = np.mean(actual_direction == predicted_direction) * 100

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Directional Accuracy: {da:.2f}%")

    # Save forecasts
    results = pd.DataFrame({
        "Date": portfolio.index[val_end:],
        "Actual": actual,
        "Predicted": predictions
    })

    results.to_csv("results_ar.csv", index=False)

    print("\nResults saved to results_ar.csv")

    return model, results


if __name__ == "__main__":
    run_experiment()
