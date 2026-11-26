"""
Wspólna logika dla wszystkich modeli prognozujących (etap 3).

Każdy model jest uczony i testowany dokładnie tak samo jak AR z etapu 2:
  1. Cechy dla dnia t: W ostatnich log-zwrotów (r_{t-1}, ..., r_{t-W}) oraz cechy GT znane w dniu t
     (GT jest już opóźnione o tydzień i standaryzowane na zbiorze uczącym - robi to skrypt).
  2. Model przewiduje log-zwrot r_t, z którego liczona jest cena: y_hat_t = y_{t-1} * exp(r_hat_t).
  3. Ustawienia z listy `grid` (w tym długość okna W) są wybierane na zbiorze walidacyjnym
     (najmniejsze RMSE cen), model jest uczony na train, a prognozuje val.
  4. Model z wybranymi ustawieniami jest uczony ponownie na train+val i prognozuje 1 krok naprzód na teście.

Jak dodać nowy model:
  - skopiuj src/models/ar.py (model AR z etapu 2 w formie templatki) do src/models/<nazwa>.py,
  - klasa musi nazywać się Model i dziedziczyć po WindowModel,
  - ustaw `name`, `grid` (lista słowników; klucz "window" = długość okna W, reszta trafia do make_estimator),
  - napisz make_estimator(params): ma zwrócić obiekt z metodami fit(F, target) i predict(F),
    np. model ze scikit-learn albo własną klasę (np. sieć neuronową).
    W params są wszystkie ustawienia z grid OPRÓCZ "window"; długość okna jest w self.window_.
    Ziarno losowości jest w self.seed - przekaż je do modelu (np. random_state=self.seed,
    torch.manual_seed(self.seed)), żeby wyniki były powtarzalne,
  - jeśli model potrzebuje skalowania danych (np. SVR, sieci), ustaw `scale = True`,
  - uruchom: python -m scripts.run_model --model <nazwa> --synthetic

Format danych dla make_estimator:
  F - tablica (liczba dni, W + m): pierwsze W kolumn to r_{t-1}, ..., r_{t-W}, kolejne m kolumn to cechy GT;
  target - tablica (liczba dni,) z log-zwrotami r_t.
  Modele sekwencyjne (np. sieci rekurencyjne) mogą zamienić F na sekwencję funkcją as_sequence(F, W).
"""

import numpy as np

from src.models.linear_ar import log_returns


def make_features(returns, X, days, window):
    """Wiersz dla każdego dnia t z `days`: [r_{t-1}, ..., r_{t-W}, cechy GT z dnia t]."""
    columns = [returns[days - k] for k in range(1, window + 1)]
    if X is not None:
        columns += list(np.asarray(X, dtype=float)[days].T)
    return np.column_stack(columns)


def as_sequence(F, window):
    """Zamienia F na sekwencję (liczba dni, W, 1 + m) w kolejności od najstarszego dnia do najnowszego.
    Cechy GT (stałe w obrębie okna) są powielone na każdy krok sekwencji."""
    lags = F[:, :window][:, ::-1, None]                      # r_{t-W}, ..., r_{t-1}
    gt = np.repeat(F[:, None, window:], window, axis=1)      # te same cechy GT w każdym kroku
    return np.concatenate([lags, gt], axis=2)


class WindowModel:
    """Baza dla wszystkich modeli. W podklasie ustaw: name, grid, make_estimator (opcjonalnie scale)."""

    name = "model"
    grid = [{"window": 10}]
    scale = False  # True: standaryzacja cech i celu na danych uczących danego dopasowania

    def __init__(self, seed=42):
        self.seed = seed
        self.params_ = None
        self.window_ = None
        self.estimator_ = None
        self._scaling = None

    def make_estimator(self, params):
        """Zwraca obiekt z metodami fit(F, target) i predict(F). Do nadpisania w podklasie."""
        raise NotImplementedError

    def label(self, has_gt):
        """Opis modelu w tabeli wyników, np. SVR(window=10, C=0.1)."""
        settings = ", ".join(f"{k}={v}" for k, v in self.params_.items())
        return f"{self.name}({settings})"

    def _fit_params(self, y, X, params):
        """Uczy model z danymi ustawieniami na całym szeregu y (i cechach X)."""
        params = dict(params)
        window = params.pop("window")
        r = log_returns(y)
        days = np.arange(window + 1, len(r))  # r[0] jest NaN, więc pierwszy pełny wiersz ma indeks window+1
        F = make_features(r, X, days, window)
        target = r[days]

        self._scaling = None
        if self.scale:
            f_mean, f_std = F.mean(axis=0), F.std(axis=0)
            f_std[f_std == 0] = 1.0
            t_mean, t_std = target.mean(), target.std() or 1.0
            self._scaling = (f_mean, f_std, t_mean, t_std)
            F, target = (F - f_mean) / f_std, (target - t_mean) / t_std

        self.window_ = window
        self.estimator_ = self.make_estimator(params)
        self.estimator_.fit(F, target)

    def predict_range(self, y, start, end, X=None):
        """Prognozy 1-krokowe cen dla dni start..end-1; dzień t korzysta tylko z danych sprzed dnia t."""
        y = np.asarray(y, dtype=float)
        r = log_returns(y)
        days = np.arange(start, end)
        F = make_features(r, X, days, self.window_)
        if self._scaling is not None:
            f_mean, f_std, t_mean, t_std = self._scaling
            predicted_returns = self.estimator_.predict((F - f_mean) / f_std) * t_std + t_mean
        else:
            predicted_returns = self.estimator_.predict(F)
        return y[start - 1:end - 1] * np.exp(np.asarray(predicted_returns, dtype=float).ravel())

    def fit(self, y_train, y_val, X_train=None, X_val=None):
        """Wybór ustawień na val (uczenie na train), potem ponowne uczenie na train+val."""
        y_all = np.r_[y_train, y_val]
        X_all = None if X_train is None else np.vstack([X_train, X_val])
        n_train = len(y_train)

        best_params, best_rmse = self.grid[0], np.inf
        for params in self.grid:
            self._fit_params(y_train, X_train, params)
            preds = self.predict_range(y_all, n_train, len(y_all), X_all)
            rmse = np.sqrt(np.mean((np.asarray(y_val) - preds) ** 2))
            if rmse < best_rmse:
                best_params, best_rmse = params, rmse

        self._fit_params(y_all, X_all, best_params)
        self.params_ = dict(best_params)
        return self
