"""Modele AR(p) / ARX(p) na log-zwrotach (OLS), metryki i test Diebolda-Mariano."""

from typing import Dict, Optional, Tuple
import numpy as np
from scipy import stats


def evaluate_forecasts(actual: np.ndarray, predicted: np.ndarray, prev_actual: np.ndarray) -> Dict[str, float]:
    """MAE, RMSE, MAPE [%] i Directional Accuracy [%] (kierunek względem y_{t-1})."""
    err = actual - predicted
    da = np.mean(np.sign(actual - prev_actual) == np.sign(predicted - prev_actual)) * 100.0
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_%": float(np.mean(np.abs(err / actual)) * 100.0),
        "DA_%": float(da),
    }


def diebold_mariano_test(actual: np.ndarray, pred_base: np.ndarray, pred_cand: np.ndarray) -> Tuple[float, float]:
    """Test Diebolda-Mariano (DM < 0 oznacza niższy błąd modelu kandydującego)."""
    d = (actual - pred_cand) ** 2 - (actual - pred_base) ** 2
    var = float(np.var(d, ddof=1))
    if var <= 1e-18:
        return 0.0, 1.0
    dm = float(np.mean(d) / np.sqrt(var / len(d)))
    return dm, float(2.0 * stats.t.sf(np.abs(dm), df=len(d) - 1))


class NaiveRandomWalk:
    """Model naiwny (błądzenie losowe): y_hat_t = y_{t-1}."""
    def predict_range(self, y_full: np.ndarray, start: int, end: int) -> np.ndarray:
        return np.asarray(y_full, dtype=float)[start - 1 : end - 1].copy()


def _log_returns(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return np.r_[np.nan, np.log(y[1:] / y[:-1])]


def _design(r: np.ndarray, X: Optional[np.ndarray], idx: np.ndarray, p: int) -> np.ndarray:
    """Macierz cech dla chwil idx: [1, r_{t-1}..r_{t-p}, X_t]."""
    cols = [np.ones(len(idx))] + [r[idx - k] for k in range(1, p + 1)]
    if X is not None:
        cols += list(np.asarray(X, dtype=float)[idx].T)
    return np.column_stack(cols)


class LinearAR:
    """AR(p) / ARX(p) estymowany OLS na log-zwrotach; prognoza cen: y_t = y_{t-1} * exp(r_hat)."""
    def __init__(self, p: Optional[int] = None, p_grid=(1, 2, 3, 5, 10)) -> None:
        self.p = p
        self.p_grid = tuple(p_grid)
        self.selected_p_ = p
        self.coef_ = None

    def _fit_p(self, y, X, p: int) -> "LinearAR":
        r = _log_returns(y)
        idx = np.arange(p + 1, len(r))  # r[0] to NaN, więc pierwszy użyteczny wiersz ma indeks p+1
        self.coef_ = np.linalg.lstsq(_design(r, X, idx, p), r[idx], rcond=None)[0]
        self.selected_p_ = p
        return self

    def fit(self, y_train, y_val=None, X_train=None, X_val=None) -> "LinearAR":
        """Gdy p=None i podano y_val: dobiera p na walidacji, potem uczy na train+val."""
        if self.p is not None or y_val is None:
            return self._fit_p(y_train, X_train, self.p or 5)

        y_tv = np.r_[y_train, y_val]
        X_tv = None if X_train is None else np.vstack([X_train, X_val])
        n = len(y_train)

        def val_rmse(p: int) -> float:
            try:
                self._fit_p(y_train, X_train, p)
                pred = [self.predict_one_step(y_tv, i, X_tv) for i in range(n, len(y_tv))]
                rmse = float(np.sqrt(np.mean((np.asarray(y_val) - pred) ** 2)))
                return rmse if np.isfinite(rmse) else np.inf
            except Exception:
                return np.inf

        return self._fit_p(y_tv, X_tv, min(self.p_grid, key=val_rmse))

    def predict_one_step(self, y_full, start: int, X_full=None) -> float:
        """Prognoza 1-krokowa dla y[start]."""
        y = np.asarray(y_full, dtype=float)
        feat = _design(_log_returns(y), X_full, np.array([start]), self.selected_p_)
        return float(y[start - 1] * np.exp((feat @ self.coef_)[0]))