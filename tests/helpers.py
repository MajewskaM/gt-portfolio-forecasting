"""Konfiguracja dla testów pobierania danych"""

from __future__ import annotations

from config.portfolio_config import PortfolioConfig


def synthetic_config(
    tmp_dir: str = "/tmp/gt_portfolio_test", **overrides
) -> PortfolioConfig:
    defaults = dict(
        start_date="2019-01-01",
        end_date="2021-01-01",
        cache_dir=f"{tmp_dir}/cache",
        gt_dir=f"{tmp_dir}/gt",
    )
    defaults.update(overrides)
    return PortfolioConfig(**defaults)
