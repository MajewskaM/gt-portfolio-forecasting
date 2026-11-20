import numpy as np
import pandas as pd
import pytest
from src.data.alignment import align_gt_to_trading_days
from src.models.linear_ar import LinearAR, diebold_mariano_test, evaluate_forecasts


def test_gt_alignment_no_leakage():
    gt = {"Trend_TEST": pd.Series([10.0, 50.0], index=pd.to_datetime(["2019-01-06", "2019-01-13"]))}
    aligned = align_gt_to_trading_days(gt, pd.bdate_range("2019-01-07", "2019-01-18"))
    assert aligned.loc["2019-01-07":"2019-01-11", "Trend_TEST"].isna().all()
    assert (aligned.loc["2019-01-14":"2019-01-18", "Trend_TEST"] == 10.0).all()


def test_metrics_and_dm():
    m = evaluate_forecasts(np.array([102., 101., 105.]), np.array([101., 100.5, 106.]), np.array([100., 102., 101.]))
    assert m["DA_%"] == pytest.approx(100.0)
    rng = np.random.default_rng(7)
    y = 100.0 + np.cumsum(rng.normal(0, 1, 200))
    dm, p = diebold_mariano_test(y, y + rng.normal(0, 2, 200), y + rng.normal(0, 0.5, 200))
    assert dm < 0 and p < 0.01


def test_linear_ar_fit_predict():
    rng = np.random.default_rng(42)
    y = 100_000.0 * np.cumprod(1 + rng.normal(0.0004, 0.01, 250))
    X = rng.normal(0, 1, size=(250, 2))
    model = LinearAR(p=None, p_grid=(1, 2, 5)).fit(y[:175], y[175:210], X[:175], X[175:210])
    assert model.selected_p_ in (1, 2, 5)
    assert np.isfinite(model.predict_one_step(y, 210, X))