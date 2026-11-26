"""
Etap 2: modele autoregresyjne AR / ARX.

"""

import argparse
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
from src.models.linear_ar import LinearAR, NaiveRandomWalk, diebold_mariano_test, evaluate_forecasts


def fit_predict(y, gt, train_end, val_end):
    """GT standaryzowane na train, p dobrane na val, prognozy 1-krokowe na teście."""
    X = scale_gt_train_only(gt, train_end)  # standaryzacja GT na zbiorze uczącym; None, gdy brak GT
    if X is None:
        X_train, X_val = None, None
    else:
        X_train, X_val = X[:train_end], X[train_end:val_end]

    model = LinearAR(p=None)
    model.fit(y[:train_end], y[train_end:val_end], X_train, X_val)
    predictions = [model.predict_one_step(y, t, X) for t in range(val_end, len(y))]
    return np.array(predictions), model.selected_p_

def make_row(info, feature_set, label, actual, pred, prev, base_pred, rw_pred, alpha):
    """Jeden wiersz tabeli wyników: miary błędów i testy Diebolda-Mariano.
    base_pred - prognoza modelu bez GT (czy GT pomaga), rw_pred - prognoza modelu naiwnego."""
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


def forecast_target(info, y, variants, file_tag, ds, alpha, out_dir):
    """Prognozy dla jednego celu (portfel albo pojedyncza spółka): model naiwny oraz AR/ARX.
    variants = {nazwa: tabela cech GT albo None (model bez GT)}."""
    test = y[ds.val_end:]
    prev = y[ds.val_end - 1:-1]
    rw = NaiveRandomWalk().predict_range(y, ds.val_end, len(y))

    rows = [make_row(info, "naive_rw", "RW", test, rw, prev, None, None, alpha)]
    preds = {}
    for name, gt in variants.items():
        pred, p = fit_predict(y, gt, ds.train_end, ds.val_end)
        preds[name] = pred
        if gt is None:
            label, base_pred = f"AR({p})", None
        else:
            label, base_pred = f"ARX({p})", preds["baseline"]
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


def run_for_portfolio(pid, out_dir, synthetic=False):
    assets = get_preset(pid)
    cfg_macro = PortfolioConfig(portfolio_id=pid, assets=assets, feature_set=FeatureSet.WITH_GT_MACRO)
    cfg_asset = PortfolioConfig(portfolio_id=pid, assets=assets, feature_set=FeatureSet.WITH_GT_ASSET)

    # dane: ceny oraz Google Trends (jeden sygnał makro i po jednym na aktywo)
    if synthetic:
        ohlcv, gt_macro = fake_ohlcv_and_gt(cfg_macro)
        _, gt_asset = fake_ohlcv_and_gt(cfg_asset)
    else:
        ohlcv = DataAcquisition(cfg_macro).fetch_ohlcv()
        gt_macro = DataAcquisition(cfg_macro).fetch_gt_for_feature_set()
        gt_asset = DataAcquisition(cfg_asset).fetch_gt_for_feature_set()

    ds = build_aligned_dataset(cfg_asset, ohlcv, {**gt_macro, **gt_asset})
    gt = ds.gt_features
    alpha = cfg_asset.alpha

    # 1. portfel - jeden model dla wartości całego portfela
    info = {"portfolio": pid, "target_mode": "portfolio", "target_name": pid, "aggregation": "direct"}
    variants = {
        "baseline": None,
        "with_gt_macro": gt[list(gt_macro)],
        "with_gt_asset": gt[list(gt_asset)],
    }
    y = ds.portfolio_value.to_numpy(dtype=float)
    rows, preds, rw = forecast_target(info, y, variants, f"{pid}_portfolio_direct", ds, alpha, out_dir)

    test = y[ds.val_end:]
    prev = y[ds.val_end - 1:-1]
    save_forecast_plot(out_dir, pid, ds.dates[ds.val_end:], test, {
        "RW": rw, "AR": preds["baseline"], "ARX macro": preds["with_gt_macro"], "ARX asset": preds["with_gt_asset"],
    })

    # 2. portfel - osobny model dla każdego aktywa, prognozy dodane z wagami (liczbą jednostek)
    info_bottom_up = dict(info, aggregation="bottom_up")
    for name in ("baseline", "with_gt_asset"):
        total = 0
        for ticker in cfg_asset.tickers:
            gt_ticker = gt[[f"Trend_SVI_{ticker}"]] if name == "with_gt_asset" else None
            prices = ds.asset_prices[ticker].to_numpy(dtype=float)
            pred_ticker, _ = fit_predict(prices, gt_ticker, ds.train_end, ds.val_end)
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
    rows += forecast_target(info_single, y_single, variants, f"{pid}_single_{ticker}", ds, alpha, out_dir)[0]

    return pd.DataFrame(rows)


def add_relative_rmse(summary):
    """Dodaje kolumnę RMSE_vs_RW = RMSE modelu / RMSE modelu naiwnego (poniżej 1 = lepszy od naiwnego).
    Jest bezwymiarowa, więc można porównywać portfel z pojedynczą spółką."""
    naive = summary[summary["feature_set"] == "naive_rw"][["portfolio", "target_mode", "RMSE"]]
    naive = naive.rename(columns={"RMSE": "RMSE_RW"})
    summary = summary.merge(naive, on=["portfolio", "target_mode"], how="left")
    summary["RMSE_vs_RW"] = summary["RMSE"] / summary["RMSE_RW"]
    return summary.drop(columns="RMSE_RW")


def save_rmse_chart(summary, out_dir):
    direct = summary[(summary["target_mode"] == "portfolio") & (summary["aggregation"] == "direct")]
    table = direct.pivot(index="portfolio", columns="feature_set", values="RMSE")
    table = table[["naive_rw", "baseline", "with_gt_macro", "with_gt_asset"]]
    table.plot(kind="bar", figsize=(9, 4.5), rot=0)
    plt.title("RMSE modeli autoregresyjnych (PORTFOLIO, DIRECT)")
    plt.ylabel("RMSE [USD]")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rmse_comparison_portfolios.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--portfolio", default="diversified_9")
    parser.add_argument("--all-presets", action="store_true", help="powtórz dla wszystkich portfeli")
    parser.add_argument("--synthetic", action="store_true", help="dane sztuczne - tylko test działania kodu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    out_dir = os.path.join("results", "step2_synthetic" if args.synthetic else "step2")
    os.makedirs(out_dir, exist_ok=True)

    portfolios = [args.portfolio]
    if args.all_presets:
        portfolios = ["diversified_9"] + list(ROBUSTNESS_PRESETS)

    results = [run_for_portfolio(pid, out_dir, args.synthetic) for pid in portfolios]
    summary = add_relative_rmse(pd.concat(results, ignore_index=True))
    summary.round(4).to_csv(os.path.join(out_dir, "step2_metrics_summary.csv"), index=False)
    save_rmse_chart(summary, out_dir)

    print("\n=== PODSUMOWANIE WYNIKÓW ETAPU 2 ===")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()