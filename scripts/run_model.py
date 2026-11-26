"""
Wspólny skrypt uruchamiający dowolny model na tych samych danych i w ten sam sposób co AR z etapu 2.

Dla każdego portfela liczy:
  1. portfel - prognoza bezpośrednia (warianty: bez GT, GT makro, GT aktywa),
  2. portfel - prognoza oddolna (osobny model dla każdego aktywa; bez GT i GT aktywa),
  3. pojedyncza spółka (bez GT, GT makro, GT aktywa),
oraz model naiwny, miary błędów i testy Diebolda-Mariano (vs model naiwny, vs ten sam model bez GT).

Model wskazuje się nazwą pliku z src/models/ (klasa Model), np.:
    python -m scripts.run_model --model ar --synthetic          # test na danych sztucznych
    python -m scripts.run_model --model ar --all-presets        # wszystkie portfele, prawdziwe dane

Wyniki: results/step3/<model>/<przebieg>/ (metrics_summary.csv, config.json, pred_*.csv, wykresy),
gdzie <przebieg> = s<seed> albo s<seed>_<tag>. Każdy przebieg ma osobny folder, więc nic się nie nadpisuje.
"""

import argparse
import importlib
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.portfolio_config import FeatureSet, PortfolioConfig
from src.data.acquisition import DataAcquisition
from src.data.alignment import build_aligned_dataset, scale_gt_train_only
from src.data.portfolio_presets import ROBUSTNESS_PRESETS, get_preset
from src.data.synthetic import fake_ohlcv_and_gt
from src.models.linear_ar import NaiveRandomWalk, diebold_mariano_test, evaluate_forecasts


def load_model_class(model_name):
    """Wczytuje klasę Model z pliku src/models/<model_name>.py."""
    return importlib.import_module(f"src.models.{model_name}").Model


def fit_predict(model_class, seed, y, gt, train_end, val_end):
    """GT standaryzowane na train, ustawienia dobrane na val, prognozy 1-krokowe na teście."""
    X = scale_gt_train_only(gt, train_end)  # None, gdy brak GT
    if X is None:
        X_train, X_val = None, None
    else:
        X_train, X_val = X[:train_end], X[train_end:val_end]

    model = model_class(seed=seed)
    model.fit(y[:train_end], y[train_end:val_end], X_train, X_val)
    predictions = model.predict_range(y, val_end, len(y), X)
    return np.asarray(predictions, dtype=float), model.label(gt is not None)


def make_row(info, feature_set, label, actual, pred, prev, base_pred, rw_pred, alpha):
    """Jeden wiersz tabeli wyników: miary błędów i testy Diebolda-Mariano.
    base_pred - prognoza tego samego modelu bez GT, rw_pred - prognoza modelu naiwnego."""
    row = dict(info)
    row["feature_set"] = feature_set
    row["selected_p"] = label
    row.update(evaluate_forecasts(actual, pred, prev))
    if feature_set == "naive_rw":
        row["DA_%"] = np.nan

    dm_base, p_base = np.nan, np.nan
    if base_pred is not None:
        dm_base, p_base = diebold_mariano_test(actual, base_pred, pred)
    dm_rw, p_rw = np.nan, np.nan
    if rw_pred is not None:
        dm_rw, p_rw = diebold_mariano_test(actual, rw_pred, pred)

    row["DM_stat_vs_base"] = dm_base
    row["DM_pvalue"] = p_base
    row["significant_005"] = bool(p_base < alpha)
    row["DM_stat_vs_RW"] = dm_rw
    row["DM_pvalue_vs_RW"] = p_rw
    return row


def save_predictions(out_dir, file_name, dates, actual, pred):
    table = pd.DataFrame({"Date": dates, "Actual": actual, "Predicted": pred, "Error": actual - pred})
    table.to_csv(os.path.join(out_dir, file_name), index=False)


def forecast_target(run, info, y, variants, file_tag, ds, alpha, out_dir):
    """Prognozy dla jednego celu: model naiwny oraz model w każdym wariancie GT.
    variants = {nazwa: tabela cech GT albo None (model bez GT)}."""
    test = y[ds.val_end:]
    prev = y[ds.val_end - 1:-1]
    rw = NaiveRandomWalk().predict_range(y, ds.val_end, len(y))

    rows = [make_row(info, "naive_rw", "RW", test, rw, prev, None, None, alpha)]
    preds = {}
    for name, gt in variants.items():
        pred, label = fit_predict(run["model_class"], run["seed"], y, gt, ds.train_end, ds.val_end)
        preds[name] = pred
        base_pred = None if gt is None else preds["baseline"]
        rows.append(make_row(info, name, label, test, pred, prev, base_pred, rw, alpha))
        save_predictions(out_dir, f"pred_{file_tag}_{name}.csv", ds.dates[ds.val_end:], test, pred)
    return rows, preds, rw


def save_forecast_plot(out_dir, pid, dates, actual, preds):
    plt.figure(figsize=(11, 4.5))
    plt.plot(dates, actual, color="black", linewidth=1.6, label="Rzeczywista")
    for name, pred in preds.items():
        plt.plot(dates, pred, linestyle="--", linewidth=1.2, alpha=0.85, label=name)
    plt.title(f"Prognozy portfela ({pid}) - zbiór testowy")
    plt.ylabel("Wartość [USD]")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"forecast_{pid}_portfolio.png"), dpi=150)
    plt.close()


def run_for_portfolio(run, pid, out_dir):
    assets = get_preset(pid)
    cfg_macro = PortfolioConfig(portfolio_id=pid, assets=assets, feature_set=FeatureSet.WITH_GT_MACRO)
    cfg_asset = PortfolioConfig(portfolio_id=pid, assets=assets, feature_set=FeatureSet.WITH_GT_ASSET)

    # dane: ceny oraz Google Trends (jeden sygnał makro i po jednym na aktywo)
    if run["synthetic"]:
        ohlcv, gt_macro = fake_ohlcv_and_gt(cfg_macro)
        _, gt_asset = fake_ohlcv_and_gt(cfg_asset)
    else:
        ohlcv = DataAcquisition(cfg_macro).fetch_ohlcv()
        gt_macro = DataAcquisition(cfg_macro).fetch_gt_for_feature_set()
        gt_asset = DataAcquisition(cfg_asset).fetch_gt_for_feature_set()

    ds = build_aligned_dataset(cfg_asset, ohlcv, {**gt_macro, **gt_asset})
    gt = ds.gt_features
    alpha = cfg_asset.alpha
    short = run["short_name"]

    # 1. portfel - jeden model dla wartości całego portfela
    info = {"portfolio": pid, "target_mode": "portfolio", "target_name": pid, "aggregation": "direct"}
    variants = {
        "baseline": None,
        "with_gt_macro": gt[list(gt_macro)],
        "with_gt_asset": gt[list(gt_asset)],
    }
    y = ds.portfolio_value.to_numpy(dtype=float)
    rows, preds, rw = forecast_target(run, info, y, variants, f"{pid}_portfolio_direct", ds, alpha, out_dir)

    test = y[ds.val_end:]
    prev = y[ds.val_end - 1:-1]
    save_forecast_plot(out_dir, pid, ds.dates[ds.val_end:], test, {
        "RW": rw, short: preds["baseline"],
        f"{short}X macro" if short == "AR" else f"{short} GT macro": preds["with_gt_macro"],
        f"{short}X asset" if short == "AR" else f"{short} GT asset": preds["with_gt_asset"],
    })

    # 2. portfel - osobny model dla każdego aktywa, prognozy dodane z wagami (liczbą jednostek)
    if run["bottom_up"]:
        info_bottom_up = dict(info, aggregation="bottom_up")
        for name in ("baseline", "with_gt_asset"):
            total = 0
            for ticker in cfg_asset.tickers:
                gt_ticker = gt[[f"Trend_SVI_{ticker}"]] if name == "with_gt_asset" else None
                prices = ds.asset_prices[ticker].to_numpy(dtype=float)
                pred_ticker, _ = fit_predict(run["model_class"], run["seed"], prices, gt_ticker,
                                             ds.train_end, ds.val_end)
                total = total + pred_ticker * ds.shares[ticker]
            rows.append(make_row(info_bottom_up, name, "per_asset", test, total, prev, preds["baseline"], rw, alpha))
            save_predictions(out_dir, f"pred_{pid}_portfolio_bottomup_{name}.csv", ds.dates[ds.val_end:], test, total)

    # 3. pojedyncza spółka
    ticker = cfg_asset.single_ticker
    info_single = {"portfolio": pid, "target_mode": "single_asset", "target_name": ticker, "aggregation": "single"}
    variants = {
        "baseline": None,
        "with_gt_macro": gt[list(gt_macro)],
        "with_gt_asset": gt[[f"Trend_SVI_{ticker}"]],
    }
    y_single = ds.single_asset_price.to_numpy(dtype=float)
    rows += forecast_target(run, info_single, y_single, variants, f"{pid}_single_{ticker}", ds, alpha, out_dir)[0]

    return pd.DataFrame(rows)


def add_relative_rmse(summary):
    """Dodaje kolumnę RMSE_vs_RW = RMSE modelu / RMSE modelu naiwnego (poniżej 1 = lepszy od naiwnego).
    Jest bezwymiarowa, więc można porównywać portfel z pojedynczą spółką."""
    naive = summary[summary["feature_set"] == "naive_rw"][["portfolio", "target_mode", "RMSE"]]
    naive = naive.rename(columns={"RMSE": "RMSE_RW"})
    summary = summary.merge(naive, on=["portfolio", "target_mode"], how="left")
    summary["RMSE_vs_RW"] = summary["RMSE"] / summary["RMSE_RW"]
    return summary.drop(columns="RMSE_RW")


def save_rmse_chart(summary, out_dir, model_name):
    direct = summary[(summary["target_mode"] == "portfolio") & (summary["aggregation"] == "direct")]
    table = direct.pivot(index="portfolio", columns="feature_set", values="RMSE")
    table = table[["naive_rw", "baseline", "with_gt_macro", "with_gt_asset"]]
    table.plot(kind="bar", figsize=(9, 4.5), rot=0)
    plt.title(f"RMSE modeli {model_name} (PORTFOLIO, DIRECT)")
    plt.ylabel("RMSE [USD]")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rmse_comparison_portfolios.png"), dpi=150)
    plt.close()


def run_model(model_name, out_dir, portfolios, synthetic=False, bottom_up=True, seed=42,
              summary_file="metrics_summary.csv", chart_title=None):
    """Uruchamia model na wskazanych portfelach i zapisuje wyniki w out_dir. Zwraca tabelę wyników."""
    np.random.seed(seed)
    model_class = load_model_class(model_name)
    run = {
        "model_class": model_class,
        "short_name": model_class.name,
        "seed": seed,
        "synthetic": synthetic,
        "bottom_up": bottom_up,
    }
    os.makedirs(out_dir, exist_ok=True)

    results = [run_for_portfolio(run, pid, out_dir) for pid in portfolios]
    summary = add_relative_rmse(pd.concat(results, ignore_index=True))
    summary.round(4).to_csv(os.path.join(out_dir, summary_file), index=False)
    save_rmse_chart(summary, out_dir, chart_title or model_class.name)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="nazwa pliku z src/models/ bez .py, np. ar")
    parser.add_argument("--portfolio", default="diversified_9")
    parser.add_argument("--all-presets", action="store_true", help="powtórz dla wszystkich portfeli")
    parser.add_argument("--synthetic", action="store_true", help="dane sztuczne - tylko test działania kodu")
    parser.add_argument("--no-bottom-up", action="store_true", help="pomiń prognozę oddolną (szybciej)")
    parser.add_argument("--seed", type=int, default=42, help="ziarno losowości (dla modeli losowych)")
    parser.add_argument("--tag", default="", help="dopisek do nazwy przebiegu, np. hidden64")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    portfolios = [args.portfolio]
    if args.all_presets:
        portfolios = ["diversified_9"] + list(ROBUSTNESS_PRESETS)

    run_id = f"s{args.seed}" + (f"_{args.tag}" if args.tag else "")
    base_dir = os.path.join("results", "step3_synthetic" if args.synthetic else "step3")
    out_dir = os.path.join(base_dir, args.model, run_id)

    summary = run_model(args.model, out_dir, portfolios, synthetic=args.synthetic,
                        bottom_up=not args.no_bottom_up, seed=args.seed)

    config = {"model": args.model, "run_id": run_id, "seed": args.seed, "tag": args.tag,
              "portfolios": portfolios, "synthetic": args.synthetic, "bottom_up": not args.no_bottom_up,
              "grid": load_model_class(args.model).grid}
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n=== WYNIKI: {args.model} ({run_id}) -> {out_dir} ===")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
