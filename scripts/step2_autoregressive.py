"""
Etap 2: modele autoregresyjne AR / ARX.
Uruchomienie:
    python -m scripts.step2_autoregressive --all-presets
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


def fit_predict(y, gt_df, train_end, val_end):
    """GT standaryzowane na train, p dobrane na val, prognozy 1-krokowe na teście."""
    X = scale_gt_train_only(gt_df, train_end)
    split = lambda a, b: None if X is None else X[a:b]
    model = LinearAR(p=None).fit(y[:train_end], y[train_end:val_end], split(0, train_end), split(train_end, val_end))
    preds = np.array([model.predict_one_step(y, i, X) for i in range(val_end, len(y))])
    return preds, model.selected_p_


def dm(actual, ref, pred):
    """Test Diebolda-Mariano względem prognozy ref (NaN, gdy brak ref)."""
    return diebold_mariano_test(actual, ref, pred) if ref is not None else (np.nan, np.nan)


def make_record(meta, fset, label, actual, pred, prev, base=None, rw=None, alpha=0.05):
    """Wiersz wyników: metryki + DM vs baseline (czy GT pomaga) + DM vs RW (czy pobijamy RW)."""
    metrics = evaluate_forecasts(actual, pred, prev)
    if fset == "naive_rw":
        metrics["DA_%"] = np.nan  # RW zawsze przewiduje 'bez zmiany'
    (s_b, p_b), (s_r, p_r) = dm(actual, base, pred), dm(actual, rw, pred)
    return {
        **dict(zip(["portfolio", "target_mode", "target_name", "aggregation"], meta)),
        "feature_set": fset, "selected_p": label, **metrics,
        "DM_stat_vs_base": s_b, "DM_pvalue": p_b, "significant_005": bool(p_b < alpha),
        "DM_stat_vs_RW": s_r, "DM_pvalue_vs_RW": p_r,
    }


def run_block(meta, y, variants, tag, ds, alpha, out_dir):
    """RW + AR/ARX dla jednego celu. variants = {feature_set: DataFrame GT albo None}."""
    test, prev = y[ds.val_end:], y[ds.val_end - 1:-1]
    rw = NaiveRandomWalk().predict_range(y, ds.val_end, len(y))
    rows, preds = [make_record(meta, "naive_rw", "RW", test, rw, prev)], {}
    for name, gt in variants.items():
        preds[name], p = fit_predict(y, gt, ds.train_end, ds.val_end)
        label = f"{'AR' if gt is None else 'ARX'}({p})"
        rows.append(make_record(meta, name, label, test, preds[name], prev, preds.get("baseline") if gt is not None else None, rw, alpha))
        save_csv(out_dir, f"pred_{tag}_{name}.csv", ds.dates[ds.val_end:], test, preds[name])
    return rows, preds, test, prev, rw


def save_csv(out_dir, fname, dates, actual, pred):
    pd.DataFrame({"Date": dates, "Actual": actual, "Predicted": pred, "Error": actual - pred}).to_csv(
        os.path.join(out_dir, fname), index=False)


def save_plot(out_dir, pid, dates, actual, preds):
    plt.figure(figsize=(11, 4.5))
    plt.plot(dates, actual, "k", lw=1.6, label="Rzeczywista")
    for name, p in preds.items():
        plt.plot(dates, p, "--", lw=1.2, alpha=0.85, label=name)
    plt.title(f"Prognozy portfela ({pid}) - zbiór testowy")
    plt.ylabel("Wartość [USD]")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"forecast_{pid}_portfolio.png"), dpi=150)
    plt.close()


def run_for_portfolio(pid, out_dir, synthetic=False):
    assets = get_preset(pid)
    cfg_m = PortfolioConfig(portfolio_id=pid, assets=assets, feature_set=FeatureSet.WITH_GT_MACRO)
    cfg_a = PortfolioConfig(portfolio_id=pid, assets=assets, feature_set=FeatureSet.WITH_GT_ASSET)
    if synthetic:
        ohlcv, gt_macro = fake_ohlcv_and_gt(cfg_m)
        _, gt_asset = fake_ohlcv_and_gt(cfg_a)
    else:
        ohlcv = DataAcquisition(cfg_m).fetch_ohlcv()
        gt_macro = DataAcquisition(cfg_m).fetch_gt_for_feature_set()
        gt_asset = DataAcquisition(cfg_a).fetch_gt_for_feature_set()

    ds = build_aligned_dataset(cfg_a, ohlcv, {**gt_macro, **gt_asset})
    gt, kw = ds.gt_features, dict(ds=ds, alpha=cfg_a.alpha, out_dir=out_dir)

    # 1. portfel, prognoza wprost
    variants = {"baseline": None, "with_gt_macro": gt[list(gt_macro)], "with_gt_asset": gt[list(gt_asset)]}
    rows, preds, test, prev, rw = run_block((pid, "portfolio", pid, "direct"), ds.portfolio_value.to_numpy(float), variants, f"{pid}_portfolio_direct", **kw)
    save_plot(out_dir, pid, ds.dates[ds.val_end:], test, {"RW": rw, "AR": preds["baseline"],"ARX macro": preds["with_gt_macro"], "ARX asset": preds["with_gt_asset"]})

    # 2. portfel, bottom-up: osobny model na aktywo, prognozy ważone liczbą jednostek
    for name in ("baseline", "with_gt_asset"):
        bu = sum(fit_predict(ds.asset_prices[t].to_numpy(float),gt[[f"Trend_SVI_{t}"]] if name == "with_gt_asset" else None,ds.train_end, ds.val_end)[0] * ds.shares[t] for t in cfg_a.tickers)
        rows.append(make_record((pid, "portfolio", pid, "bottom_up"), name, "per_asset", test, bu, prev,preds["baseline"], rw, cfg_a.alpha))
        save_csv(out_dir, f"pred_{pid}_portfolio_bottomup_{name}.csv", ds.dates[ds.val_end:], test, bu)

    # 3. pojedyncza spółka
    t = cfg_a.single_ticker
    variants = {"baseline": None, "with_gt_macro": gt[list(gt_macro)], "with_gt_asset": gt[[f"Trend_SVI_{t}"]]}
    rows += run_block((pid, "single_asset", t, "single"), ds.single_asset_price.to_numpy(float),variants, f"{pid}_single_{t}", **kw)[0]
    return pd.DataFrame(rows)


def add_relative_rmse(summary):
    """RMSE_vs_RW = RMSE / RMSE modelu naiwnego (<1: lepszy od RW); bezwymiarowe, więc porównywalne między celami."""
    rw = (summary[summary["feature_set"] == "naive_rw"][["portfolio", "target_mode", "RMSE"]].rename(columns={"RMSE": "RMSE_RW"}))
    out = summary.merge(rw, on=["portfolio", "target_mode"], how="left")
    out["RMSE_vs_RW"] = out["RMSE"] / out["RMSE_RW"]
    return out.drop(columns="RMSE_RW")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--portfolio", default="diversified_9")
    ap.add_argument("--all-presets", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    out_dir = os.path.join("results", "step2_synthetic" if args.synthetic else "step2")
    os.makedirs(out_dir, exist_ok=True)
    portfolios = ["diversified_9", *ROBUSTNESS_PRESETS] if args.all_presets else [args.portfolio]

    summary = add_relative_rmse(pd.concat([run_for_portfolio(p, out_dir, args.synthetic) for p in portfolios],ignore_index=True))
    summary.round(4).to_csv(os.path.join(out_dir, "step2_metrics_summary.csv"), index=False)

    sub = summary[(summary["target_mode"] == "portfolio") & (summary["aggregation"] == "direct")]
    sub.pivot(index="portfolio", columns="feature_set", values="RMSE")[
        ["naive_rw", "baseline", "with_gt_macro", "with_gt_asset"]].plot(kind="bar", figsize=(9, 4.5), rot=0,
        title="RMSE modeli autoregresyjnych (PORTFOLIO, DIRECT)", ylabel="RMSE [USD]")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rmse_comparison_portfolios.png"), dpi=150)
    plt.close()

    print("\n=== PODSUMOWANIE WYNIKÓW ETAPU 2 ===")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
