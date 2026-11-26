"""
Skleja wyniki wszystkich modeli z etapu 3 z wynikami AR z etapu 2 (results/step2, już w repo) w jedną tabelę
i porównuje każdy model z AR (ten sam portfel, cel i wariant GT).

Wymaga: results/step2/ (python -m scripts.step2_autoregressive --all-presets)
        results/step3/<model>/<przebieg>/ (python -m scripts.run_model --model <nazwa> ...)

    python -m scripts.collect_step3

Wyniki:
    results/step3/all_runs.csv        - wszystkie przebiegi wszystkich modeli + AR,
                                        z kolumnami RMSE_vs_AR, DM_stat_vs_AR, DM_pvalue_vs_AR
    results/step3/summary_by_model.csv - średnia i odchylenie po przebiegach (np. po ziarnach)
"""

import glob
import os

import numpy as np
import pandas as pd

from src.models.linear_ar import diebold_mariano_test

STEP2_DIR = os.path.join("results", "step2")
STEP3_DIR = os.path.join("results", "step3")
KEYS = ["portfolio", "target_mode", "target_name", "aggregation", "feature_set"]


def prediction_file(run_dir, row):
    """Nazwa pliku z prognozami dla wiersza tabeli wyników (taka sama jak w scripts/run_model.py)."""
    pid, fs = row["portfolio"], row["feature_set"]
    if row["aggregation"] == "direct":
        name = f"pred_{pid}_portfolio_direct_{fs}.csv"
    elif row["aggregation"] == "bottom_up":
        name = f"pred_{pid}_portfolio_bottomup_{fs}.csv"
    else:
        name = f"pred_{pid}_single_{row['target_name']}_{fs}.csv"
    return os.path.join(run_dir, name)


def load_runs():
    tables = []
    step2_file = os.path.join(STEP2_DIR, "step2_metrics_summary.csv")
    if not os.path.exists(step2_file):
        raise SystemExit("Brak results/step2 - najpierw: python -m scripts.step2_autoregressive --all-presets")
    ar = pd.read_csv(step2_file)
    ar.insert(0, "run", "step2")
    ar.insert(0, "model", "ar")
    ar["run_dir"] = STEP2_DIR
    tables.append(ar)

    for summary_file in sorted(glob.glob(os.path.join(STEP3_DIR, "*", "*", "metrics_summary.csv"))):
        run_dir = os.path.dirname(summary_file)
        model = os.path.basename(os.path.dirname(run_dir))
        table = pd.read_csv(summary_file)
        table.insert(0, "run", os.path.basename(run_dir))
        table.insert(0, "model", model)
        table["run_dir"] = run_dir
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def compare_with_ar(runs):
    """RMSE względem AR i test DM względem AR dla tego samego celu i wariantu GT."""
    ar = runs[runs["model"] == "ar"].set_index(KEYS)
    out = []
    for _, row in runs.iterrows():
        row = row.copy()
        row["RMSE_vs_AR"], row["DM_stat_vs_AR"], row["DM_pvalue_vs_AR"] = np.nan, np.nan, np.nan
        key = tuple(row[k] for k in KEYS)
        if row["model"] != "ar" and row["feature_set"] != "naive_rw" and key in ar.index:
            ar_row = ar.loc[key]
            row["RMSE_vs_AR"] = row["RMSE"] / ar_row["RMSE"]
            own_file = prediction_file(row["run_dir"], row)
            ar_file = prediction_file(ar_row["run_dir"], row)
            if os.path.exists(own_file) and os.path.exists(ar_file):
                own, ar_pred = pd.read_csv(own_file), pd.read_csv(ar_file)
                if not own["Date"].astype(str).equals(ar_pred["Date"].astype(str)):
                    raise SystemExit(f"Różne dni testowe w {own_file} i {ar_file} - porównanie z AR niemożliwe")
                dm, p = diebold_mariano_test(own["Actual"].to_numpy(), ar_pred["Predicted"].to_numpy(),
                                             own["Predicted"].to_numpy())
                row["DM_stat_vs_AR"], row["DM_pvalue_vs_AR"] = dm, p
        out.append(row)
    return pd.DataFrame(out)


def main():
    runs = compare_with_ar(load_runs())
    os.makedirs(STEP3_DIR, exist_ok=True)
    runs.drop(columns="run_dir").round(4).to_csv(os.path.join(STEP3_DIR, "all_runs.csv"), index=False)

    metrics = ["RMSE", "MAPE_%", "DA_%", "RMSE_vs_RW", "RMSE_vs_AR", "DM_pvalue_vs_RW", "DM_pvalue_vs_AR"]
    by_model = runs.groupby(["model"] + KEYS, dropna=False)[metrics].agg(["mean", "std"])
    by_model.columns = [f"{m}_{s}" for m, s in by_model.columns]
    by_model["n_runs"] = runs.groupby(["model"] + KEYS, dropna=False).size()
    by_model.round(4).to_csv(os.path.join(STEP3_DIR, "summary_by_model.csv"))

    direct = runs[(runs["aggregation"] == "direct") & (runs["feature_set"] != "naive_rw")]
    print("\n=== RMSE względem modelu naiwnego, portfel bezpośrednio (średnia po przebiegach) ===")
    print(direct.pivot_table(index=["portfolio", "feature_set"], columns="model", values="RMSE_vs_RW").round(4))
    print(f"\nZapisano: {STEP3_DIR}/all_runs.csv i {STEP3_DIR}/summary_by_model.csv")


if __name__ == "__main__":
    main()
