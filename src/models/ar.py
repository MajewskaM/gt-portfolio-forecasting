"""
Model AR(p) / ARX(p) z etapu 2 w szkielecie etapu 3 - templatka dla wszystkich nowych modeli.

To ten sam model co LinearAR w src/models/linear_ar.py (etap 2, plik bez zmian): regresja liniowa
log-zwrotu r_t na stałej, p ostatnich zwrotach i cechach GT z dnia t, metoda najmniejszych kwadratów.
Rząd p (= długość okna "window") jest wybierany na zbiorze walidacyjnym z {1, 2, 3, 5, 10}.
Puszczony przez szkielet daje identyczne wyniki jak w etapie 2.

Nowy model: skopiuj ten plik do src/models/<nazwa>.py i zmień name, grid i make_estimator
(klasa musi nazywać się Model). Uruchomienie: python -m scripts.run_model --model <nazwa> --synthetic
"""

import numpy as np

from src.models.window_model import WindowModel


class OLS:
    """Metoda najmniejszych kwadratów ze stałą (np.linalg.lstsq), tak jak w LinearAR."""

    def fit(self, F, target):
        design = np.column_stack([np.ones(len(F)), F])
        self.coef_ = np.linalg.lstsq(design, target, rcond=None)[0]
        return self

    def predict(self, F):
        return np.column_stack([np.ones(len(F)), F]) @ self.coef_


class Model(WindowModel):
    name = "AR"
    grid = [{"window": p} for p in (1, 2, 3, 5, 10)]  # window = rząd p

    def make_estimator(self, params):
        return OLS()

    def label(self, has_gt):
        # taki sam opis jak w wynikach etapu 2: AR(p) bez GT, ARX(p) z GT
        return f"{'ARX' if has_gt else 'AR'}({self.window_})"
