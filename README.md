# gt-portfolio-forecasting

Uruchomienie modelu autoregresyjnego (step2):
    python -m scripts.step2_autoregressive --all-presets

Etap 3 - kolejne modele (wspólny szkielet, każdy model testowany tak samo jak AR z etapu 2):
    python -m scripts.run_model --model ar --synthetic        # test szkieletu (AR z etapu 2)
    python -m scripts.run_model --model <nazwa> --all-presets # model z pliku src/models/<nazwa>.py
    python -m scripts.collect_step3                           # wszystkie modele + porównanie z AR

Nowy model: skopiuj src/models/ar.py (AR z etapu 2 jako templatka) do src/models/<nazwa>.py
i zmień name, grid i make_estimator (opis w src/models/window_model.py).
Wyniki: results/step3/<nazwa>/<przebieg>/.
