from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# def visualize_ic_analysis(ic_df, summary_df, lag_analysis_results=None):
#     if summary_df is None or len(summary_df) == 0:
#         return

#     fig, axes = plt.subplots(len(summary_df), 2, figsize=(15, 4 * len(summary_df)))
#     if len(summary_df) == 1:
#         axes = axes.reshape(1, -1)

#     for i, (_, row) in enumerate(summary_df.iterrows()):
#         factor = row["factor"]
#         ic_col = factor + "_ic"

#         valid_data = ic_df[[ic_col, "trade_date"]].dropna()
#         axes[i, 0].plot(valid_data["trade_date"], valid_data[ic_col], alpha=0.5, label="IC")
#         ic_ma = valid_data[ic_col].rolling(window=22, min_periods=1).mean()
#         axes[i, 0].plot(valid_data["trade_date"], ic_ma, color="orange", linewidth=1.2, label="MA22")
#         axes[i, 0].set_title(f"{factor} IC (IR={row['ir']:.3f})")
#         axes[i, 0].grid(True)
#         axes[i, 0].axhline(y=0, color="r", linestyle="--", alpha=0.4)
#         axes[i, 0].legend()
#         axes[i, 0].tick_params(axis="x", rotation=45)

#         valid_ic = valid_data[ic_col]
#         axes[i, 1].hist(valid_ic, bins=50, alpha=0.7, edgecolor="black")
#         axes[i, 1].set_title(f"{factor} IC Distribution")
#         axes[i, 1].grid(True)
#         axes[i, 1].axvline(x=row["ic_mean"], color="r", linestyle="--", label=f"mean={row['ic_mean']:.3f}")
#         axes[i, 1].legend()

#     plt.tight_layout()
#     plt.show()

#     if lag_analysis_results:
#         fig, axes = plt.subplots((len(lag_analysis_results) + 1) // 2, 2, figsize=(15, 4 * ((len(lag_analysis_results) + 1) // 2)))
#         axes = np.array(axes).reshape(-1)

#         for i, result in enumerate(lag_analysis_results):
#             factor = result["factor"]
#             lag_ic_values = result["lag_ic_values"]
#             half_life = result["half_life"]
#             ax = axes[i]

#             x = range(len(lag_ic_values))
#             ax.bar(x, lag_ic_values, color="skyblue", alpha=0.7)
#             ax.set_title(f"{factor} IC Decay")
#             ax.grid(True, axis="y")
#             if half_life is not None:
#                 ax.axvline(x=half_life, color="red", linestyle="--", label=f"half-life={half_life}")
#                 ax.legend()

#             valid_values = [v for v in lag_ic_values if v is not None and not math.isnan(v) and not math.isinf(v)]
#             if valid_values:
#                 y_min, y_max = min(valid_values), max(valid_values)
#                 y_range = y_max - y_min if y_max != y_min else 1
#                 ax.set_ylim(y_min - y_range * 0.15, y_max + y_range * 0.15)

#         for i in range(len(lag_analysis_results), len(axes)):
#             axes[i].set_visible(False)

#         plt.tight_layout()
#         plt.show()


def visualize_ic_analysis(ic_df, summary_df, lag_analysis_results=None):
    if summary_df is None or len(summary_df) == 0:
        return

    fig, axes = plt.subplots(len(summary_df), 3, figsize=(18, 4 * len(summary_df)))
    if len(summary_df) == 1:
        axes = axes.reshape(1, -1)

    for i, (_, row) in enumerate(summary_df.iterrows()):
        factor = row["factor"]
        ic_col = factor + "_ic"

        valid_data = ic_df[[ic_col, "trade_date"]].dropna()
        axes[i, 0].plot(
            valid_data["trade_date"],
            valid_data[ic_col],
            linewidth=0.7,
            alpha=0.6,
            label="IC",
        )
        ic_ma = valid_data[ic_col].rolling(window=22, min_periods=1).mean()
        axes[i, 0].plot(valid_data["trade_date"], ic_ma, color="orange", linewidth=1.1, label="MA22")
        axes[i, 0].set_title(f"{factor} IC (IR={row['ir']:.3f})")
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].axhline(y=0, color="r", linestyle="--", alpha=0.4)
        axes[i, 0].legend()
        axes[i, 0].tick_params(axis="x", rotation=45)

        valid_ic = valid_data[ic_col]
        axes[i, 1].hist(valid_ic, bins=50, alpha=0.7, edgecolor="black")
        axes[i, 1].set_title(f"{factor} IC Distribution")
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].axvline(
            x=row["ic_mean"],
            color="r",
            linestyle="--",
            label=f"mean={row['ic_mean']:.3f}",
        )
        axes[i, 1].legend()

        cumulative_ic = valid_ic.cumsum()
        axes[i, 2].plot(valid_data["trade_date"], cumulative_ic, color="green", linewidth=1.2)
        axes[i, 2].set_title(f"{factor} Cumulative IC")
        axes[i, 2].grid(True, alpha=0.3)
        axes[i, 2].axhline(
            y=cumulative_ic.iloc[-1],
            color="gray",
            linestyle=":",
            alpha=0.6,
            label=f"total={cumulative_ic.iloc[-1]:.3f}",
        )
        axes[i, 2].legend()
        axes[i, 2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

    if lag_analysis_results:
        fig, axes = plt.subplots(
            (len(lag_analysis_results) + 1) // 2,
            2,
            figsize=(12, 4 * ((len(lag_analysis_results) + 1) // 2)),
        )
        axes = np.array(axes).reshape(-1)

        for i, result in enumerate(lag_analysis_results):
            factor = result["factor"]
            lag_ic_values = result["lag_ic_values"]
            half_life = result["half_life"]
            ic_decay_rank_corr = result.get("ic_decay_rank_corr", np.nan)
            ax = axes[i]

            x = range(len(lag_ic_values))
            ax.bar(x, lag_ic_values, color="skyblue", alpha=0.7)
            if ic_decay_rank_corr is not None and np.isfinite(ic_decay_rank_corr):
                ax.set_title(f"{factor} IC Decay (Spearman={ic_decay_rank_corr:.3f})")
            else:
                ax.set_title(f"{factor} IC Decay")
            ax.grid(True, axis="y")
            if half_life is not None:
                ax.axvline(
                    x=half_life,
                    color="red",
                    linestyle="--",
                    label=f"half-life={half_life}",
                )
                ax.legend()

            valid_values = [v for v in lag_ic_values if v is not None and not math.isnan(v) and not math.isinf(v)]
            if valid_values:
                y_min, y_max = min(valid_values), max(valid_values)
                y_range = y_max - y_min if y_max != y_min else 1
                ax.set_ylim(y_min - y_range * 0.15, y_max + y_range * 0.15)

        for i in range(len(lag_analysis_results), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()


def visualize_layer_analysis(layer_results, show_long_short=True, show_long_short_separate=False):
    n_factors = len(layer_results)
    if n_factors == 0:
        return

    long_short_curves: list[tuple[str, pd.Series]] = []

    fig, axes = plt.subplots(n_factors, 1, figsize=(12, 5 * n_factors))
    if n_factors == 1:
        axes = [axes]

    for i, (factor, data) in enumerate(layer_results.items()):
        return_col = [col for col in data.columns if col not in ["trade_date", "layer"]][0]
        data_wide = data.pivot(index="trade_date", columns="layer", values=return_col)
        long_short_curve = data_wide["long_short"].cumsum().copy() if "long_short" in data_wide.columns else None
        if not show_long_short:
            data_wide = data_wide.loc[:, data_wide.columns != "long_short"]

        cumulative_returns_by_layer = data_wide.cumsum()
        if long_short_curve is not None:
            long_short_curves.append((factor, long_short_curve))
        for layer in cumulative_returns_by_layer.columns:
            if layer == "long_short":
                if show_long_short:
                    axes[i].plot(
                        cumulative_returns_by_layer.index,
                        cumulative_returns_by_layer[layer],
                        label="long_short",
                        color="black",
                        linewidth=1.2,
                    )
            else:
                color = plt.cm.tab10(int(layer) % 10)
                axes[i].plot(
                    cumulative_returns_by_layer.index,
                    cumulative_returns_by_layer[layer],
                    label=f"L{layer}",
                    color=color,
                    linewidth=0.7,
                )

        axes[i].set_title(f"{factor} Layer Cumulative Returns")
        axes[i].legend(loc="upper left", bbox_to_anchor=(0, 1), ncol=1)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

    if show_long_short_separate and long_short_curves:
        fig, axes = plt.subplots(len(long_short_curves), 1, figsize=(12, 4 * len(long_short_curves)))
        if len(long_short_curves) == 1:
            axes = [axes]

        for i, (factor, long_short_curve) in enumerate(long_short_curves):
            axes[i].plot(
                long_short_curve.index,
                long_short_curve.values,
                color="black",
                linewidth=1.3,
                label="long_short",
            )
            axes[i].set_title(f"{factor} Long-Short Cumulative Returns")
            axes[i].legend(loc="upper left")
            axes[i].grid(True, alpha=0.3)
            axes[i].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()


def visualize_turnover_analysis(turnover_results):
    n_factors = len(turnover_results)
    if n_factors == 0:
        return

    fig, axes = plt.subplots(n_factors, 1, figsize=(12, 5 * n_factors))
    if n_factors == 1:
        axes = [axes]

    for i, (factor, data) in enumerate(turnover_results.items()):
        dates = data["trade_date"]
        x_ticks = range(0, len(dates), max(1, len(dates) // 30))
        x_labels = [str(dates.iloc[i])[:10] for i in x_ticks]
        ax1 = axes[i]
        ax2 = ax1.twinx()

        ax1.plot(
            range(len(dates)),
            data["min_layer_turnover"] * 100,
            label="min turnover",
            color="blue",
            linewidth=0.8,
        )
        ax2.plot(
            range(len(dates)),
            data["max_layer_turnover"] * 100,
            label="max turnover",
            color="red",
            linewidth=0.8,
        )

        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_labels, rotation=45, ha="right")
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"{factor} Turnover")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.show()


def visualize_factor_correlation(corr_matrix, figsize=(12, 10)):
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Factor Correlation Matrix", fontsize=14, pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def visualize_backtest_results(portfolio_cumulative_returns):
    if portfolio_cumulative_returns is None or len(portfolio_cumulative_returns) == 0:
        return

    plt.figure(figsize=(14, 7))
    for column in portfolio_cumulative_returns.columns:
        color = plt.cm.tab10(int(column) % 10)
        plt.plot(
            portfolio_cumulative_returns.index,
            portfolio_cumulative_returns[column],
            label=f"{column}",
            color=color,
        )

    plt.title("Combined Factor Backtest")
    plt.xlabel("trade_date")
    plt.ylabel("cumulative return")
    plt.legend()
    plt.grid(True)
    plt.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()


def visualize_ewma_backtest_results(portfolio_cumulative_returns):
    if portfolio_cumulative_returns is None or len(portfolio_cumulative_returns) == 0:
        return

    plt.figure(figsize=(14, 7))
    for column in portfolio_cumulative_returns.columns:
        if column != "trade_date":
            color = plt.cm.tab10(int(column) % 10)
            plt.plot(
                portfolio_cumulative_returns["trade_date"],
                portfolio_cumulative_returns[column],
                label=f"{column}",
                color=color,
            )

    plt.title("EWMA Backtest")
    plt.xlabel("trade_date")
    plt.ylabel("cumulative return")
    plt.legend()
    plt.grid(True)
    plt.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()


def visualize_factor_distribution(df: pd.DataFrame, factor_cols: list[str], bins: int = 50):
    """Visualize cross-sectional distributions of factor values."""
    if not factor_cols:
        return
    fig, axes = plt.subplots(len(factor_cols), 1, figsize=(12, 3 * len(factor_cols)))
    if len(factor_cols) == 1:
        axes = [axes]
    for i, factor in enumerate(factor_cols):
        if factor not in df.columns:
            continue
        s = df[factor].dropna()
        sns.histplot(s, bins=bins, kde=True, ax=axes[i], color="steelblue")
        axes[i].set_title(f"{factor} Distribution")
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_ic_yearly_bar(ic_df: pd.DataFrame, factor_cols: list[str] | None = None):
    """Plot yearly IC mean bar chart for each factor."""
    if ic_df is None or ic_df.empty or "trade_date" not in ic_df.columns:
        return
    work = ic_df.copy()
    work["trade_date"] = pd.to_datetime(work["trade_date"])
    work["year"] = work["trade_date"].dt.year

    if factor_cols is None:
        factor_cols = [c[:-3] for c in work.columns if c.endswith("_ic")]
    if not factor_cols:
        return

    fig, axes = plt.subplots(len(factor_cols), 1, figsize=(12, 3 * len(factor_cols)))
    if len(factor_cols) == 1:
        axes = [axes]

    for i, factor in enumerate(factor_cols):
        col = f"{factor}_ic"
        if col not in work.columns:
            continue
        year_ic = work.groupby("year")[col].mean().dropna()
        axes[i].bar(year_ic.index.astype(str), year_ic.values, color="teal", alpha=0.8)
        axes[i].axhline(0, color="black", linewidth=0.8)
        axes[i].set_title(f"{factor} Yearly IC Mean")
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_layer_terminal_values(layer_results: dict[str, pd.DataFrame]):
    """Plot final cumulative return by layer for each factor."""
    if not layer_results:
        return
    n = len(layer_results)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n))
    if n == 1:
        axes = [axes]

    for i, (factor, data) in enumerate(layer_results.items()):
        return_col = data.columns[-1]
        daily_layer_returns = data.groupby(["trade_date", "layer"])[return_col].mean().reset_index()
        wide = daily_layer_returns.pivot(index="trade_date", columns="layer", values=return_col).sort_index()
        cum = wide.cumsum()
        terminal = cum.iloc[-1].dropna()
        axes[i].bar(terminal.index.astype(str), terminal.values, color="slateblue", alpha=0.85)
        spearman = (
            stats.spearmanr(terminal.index.astype(float), terminal.values, nan_policy="omit").correlation
            if len(terminal) > 1
            else np.nan
        )
        axes[i].set_title(f"{factor} Layer Terminal Return (Spearman={spearman:.3f})")
        axes[i].grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_period_comparison(period_comparison_df: pd.DataFrame, metric: str = "ir", factor: str | None = None):
    """Visualize holding-period robustness metric across periods."""
    if period_comparison_df is None or period_comparison_df.empty:
        return
    if metric not in period_comparison_df.columns:
        raise ValueError(f"Metric '{metric}' not in period_comparison_df")

    data = period_comparison_df.copy()
    if factor is not None and "factor" in data.columns:
        data = data[data["factor"] == factor]
    if data.empty:
        return

    plt.figure(figsize=(12, 5))
    if "factor" in data.columns and factor is None:
        for f, g in data.groupby("factor"):
            g = g.sort_values("period")
            plt.plot(g["period"], g[metric], marker="o", label=f)
        plt.legend()
    else:
        data = data.sort_values("period")
        plt.plot(data["period"], data[metric], marker="o")
    plt.title(f"Holding Period Comparison: {metric}")
    plt.xlabel("period")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
