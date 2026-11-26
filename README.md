# gt-portfolio-forecasting

## Etap 2 - model autoregresyjny

```
python -m scripts.step2_autoregressive --all-presets
```

## Etap 3 - kolejne modele

Wspólny szkielet: każdy model jest uczony i testowany tak samo jak AR z etapu 2
(okno zwrotów + GT, dobór ustawień na val, uczenie na train+val, prognoza na teście, metryki, test DM).

```
python -m scripts.run_model --model ar --synthetic          # test szkieletu (AR z etapu 2)
python -m scripts.run_model --model NAZWA --all-presets     # model z pliku src/models/NAZWA.py
python -m scripts.collect_step3                             # wszystkie modele + porównanie z AR
```

Nowy model: skopiuj `src/models/ar.py` (AR z etapu 2 jako templatka) do `src/models/NAZWA.py`
i zmień `name`, `grid` i `make_estimator` (opis w `src/models/window_model.py`).

Wyniki: `results/step3/NAZWA/PRZEBIEG/`, np. `results/step3/xgb/s42/`.
