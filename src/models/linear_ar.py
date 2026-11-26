"""Modele AR(p) / ARX(p) na log-zwrotach (OLS), metryki i test Diebolda-Mariano."""

from typing import Dict, Optional, Tuple
import numpy as np
from scipy import stats


def evaluate_forecasts(actual, predicted, prev_actual):
    """MAE, RMSE, MAPE [%] i Directional Accuracy [%] (czy zgadliśmy kierunek zmiany)."""
    error = actual - predicted
    same_direction = np.sign(actual - prev_actual) == np.sign(predicted - prev_actual)
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAPE_%": float(np.mean(np.abs(error / actual)) * 100),
        "DA_%": float(np.mean(same_direction) * 100),
    }

def diebold_mariano_test(actual, pred_base, pred_cand):
    """Czy model kandydujący różni się istotnie od bazowego? Zwraca (statystykę DM, wartość p).
    DM < 0 oznacza, że kandydat ma mniejszy błąd."""
    d = (actual - pred_cand) ** 2 - (actual - pred_base) ** 2  # różnica kwadratów błędów w każdym dniu
    var = float(np.var(d, ddof=1))
    if var <= 1e-18:
        return 0.0, 1.0
    dm = float(np.mean(d) / np.sqrt(var / len(d)))
    p_value = 2 * stats.t.sf(abs(dm), df=len(d) - 1)
    return float(dm), float(p_value)

def log_returns(y):
    y = np.asarray(y, dtype=float)
    return np.r_[np.nan, np.log(y[1:] / y[:-1])]

def features_for_day(returns, X, t, p):
    """Cechy modelu dla dnia t: stała, p ostatnich zwrotów i (opcjonalnie) cechy GT z dnia t."""
    features = [1.0]
    features += [returns[t - k] for k in range(1, p + 1)]
    if X is not None:
        features += list(X[t])
    return features

class NaiveRandomWalk:
    """Model naiwny (błądzenie losowe): y_hat_t = y_{t-1}."""
    def predict_range(self, y_full, start, end):
        return np.asarray(y_full, dtype=float)[start - 1 : end - 1].copy()

class LinearAR:
    """AR(p) bez cech GT albo ARX(p) z cechami GT. Przewiduje zwrot, z którego liczy cenę."""
    def __init__(self, p=None, p_grid=(1, 2, 3, 5, 10)):
        self.p = p # stałe p
        self.p_grid = tuple(p_grid)
        self.selected_p_ = p
        self.coef_ = None

    def _fit_p(self, y, X, p):
        """Uczy model z p opóźnieniami metodą najmniejszych kwadratów (OLS)."""
        r = log_returns(y)
        days = range(p + 1, len(r))  # r[0] jest NaN, więc zaczynamy od dnia p+1
        features = np.array([features_for_day(r, X, t, p) for t in days])
        self.coef_ = np.linalg.lstsq(features, r[p + 1:], rcond=None)[0]
        self.selected_p_ = p

    def fit(self, y_train, y_val=None, X_train=None, X_val=None):
        # stałe p: uczymy od razu
        if self.p is not None or y_val is None:
            self._fit_p(y_train, X_train, self.p or 5)
            return self

        # p=None: dla każdego p uczymy na train i sprawdzamy błąd na val
        y_all = np.r_[y_train, y_val]
        X_all = None if X_train is None else np.vstack([X_train, X_val])
        n_train = len(y_train)

        best_p, best_rmse = self.p_grid[0], np.inf
        for p in self.p_grid:
            self._fit_p(y_train, X_train, p)
            preds = [self.predict_one_step(y_all, t, X_all) for t in range(n_train, len(y_all))]
            rmse = np.sqrt(np.mean((np.asarray(y_val) - preds) ** 2))
            if rmse < best_rmse:
                best_p, best_rmse = p, rmse

        # ostateczny model: wybrane p, uczenie na train + val
        self._fit_p(y_all, X_all, best_p)
        return self

    def predict_one_step(self, y, t, X=None):
        """Prognoza ceny na dzień t na podstawie danych sprzed dnia t."""
        r = log_returns(y)
        features = features_for_day(r, X, t, self.selected_p_)
        predicted_return = float(np.dot(features, self.coef_))
        return float(y[t - 1] * np.exp(predicted_return))