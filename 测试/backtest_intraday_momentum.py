from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 全年交易日、风险厌恶系数和权重边界
ANNUAL_TRADING_DAYS = 242
RISK_AVERSION = 5.0
WEIGHT_BOUNDS = (-0.5, 1.5)
MIN_TRAIN_DAYS = 60


def locate_default_csv() -> Path:
    matches = glob.glob(r"平安银行_KLINE.csv")
    if not matches:
        raise FileNotFoundError("No *KLINE.csv file found on Desktop. ")
    return Path(matches[0])


def load_intraday_csv(csv_path: Path) -> pd.DataFrame:
    # 统一字段名和时间字段格式
    df = pd.read_csv(csv_path, encoding="gbk")
    df.columns = [
        "dt",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "chg",
        "pct",
        "amp",
        "oi",
        "ma1",
        "ma2",
        "ma3",
        "ma4",
    ]
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)
    df["trade_date"] = df["dt"].dt.normalize()
    df["bar_time"] = df["dt"].dt.strftime("%H:%M")
    return df


def build_daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    # 只保留 8 根 30 分钟 bar 完整的交易日，避免午后缺失影响信号计算
    full_days = df.groupby("trade_date").filter(lambda x: len(x) == 8).copy()
    if full_days.empty:
        raise ValueError("No full 8-bar trading days found in the CSV.")

    # 将日内收盘价展开成矩阵
    bar_matrix = (
        full_days.pivot(index="trade_date", columns="bar_time", values="close")
        .rename_axis(columns=None)
        .sort_index()
    )
    required_bars = ["10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30", "15:00"]
    missing = [bar for bar in required_bars if bar not in bar_matrix.columns]
    if missing:
        raise ValueError(f"Missing required 30-minute bars: {missing}")

    day_first = full_days.groupby("trade_date").first()
    day_last = full_days.groupby("trade_date").last()
    first_bar = (
        full_days.sort_values("dt")
        .groupby("trade_date")
        .nth(0)[["open", "high", "low", "close", "volume", "amount"]]
        .add_prefix("first_bar_")
    )
    total_day = full_days.groupby("trade_date")[["volume", "amount"]].sum().add_prefix("day_")

    # 日级基础价格和成交信息，构造全部回测信号
    daily = pd.DataFrame(index=bar_matrix.index)
    daily["prev_close"] = day_last["close"].shift(1)
    daily["open_0930"] = first_bar["first_bar_open"]
    daily["close_1000"] = bar_matrix["10:00"]
    daily["close_1400"] = bar_matrix["14:00"]
    daily["close_1430"] = bar_matrix["14:30"]
    daily["close_1500"] = bar_matrix["15:00"]
    daily["first_bar_high"] = first_bar["first_bar_high"]
    daily["first_bar_low"] = first_bar["first_bar_low"]
    daily["first_bar_volume"] = first_bar["first_bar_volume"]
    daily["day_volume"] = total_day["day_volume"]
    daily["day_amount"] = total_day["day_amount"]
    daily["day_close"] = daily["close_1500"]
    daily["day_open"] = daily["open_0930"]
    daily["bar_count"] = 8

    # 将研报中的 r1、r12、r13 映射到本地市场的 30 分钟分时结构
    daily["r1"] = daily["close_1000"] / daily["prev_close"] - 1.0
    daily["r12"] = daily["close_1430"] / daily["close_1400"] - 1.0
    daily["r13"] = daily["close_1500"] / daily["close_1430"] - 1.0
    daily["overnight_gap"] = daily["open_0930"] / daily["prev_close"] - 1.0
    daily["open_to_1000"] = daily["close_1000"] / daily["open_0930"] - 1.0
    daily["day_return"] = daily["day_close"] / daily["prev_close"] - 1.0

    # 额外补充开盘波动、量能占比和趋势状态
    daily["first_bar_range"] = (daily["first_bar_high"] - daily["first_bar_low"]) / daily["prev_close"]
    daily["first_bar_volume_share"] = daily["first_bar_volume"] / daily["day_volume"]
    daily["close_prev"] = daily["day_close"].shift(1)
    daily["sma20_prev"] = daily["day_close"].shift(1).rolling(20).mean()
    daily["trend_gap20"] = daily["close_prev"] / daily["sma20_prev"] - 1.0
    daily["abs_r1_median20"] = daily["r1"].abs().shift(1).rolling(20).median()
    daily["first_bar_volume_median20"] = daily["first_bar_volume"].shift(1).rolling(20).median()
    daily["first_bar_range_mean20"] = daily["first_bar_range"].shift(1).rolling(20).mean()
    daily["volume_share_mean20"] = daily["first_bar_volume_share"].shift(1).rolling(20).mean()
    daily["high_vol_regime"] = daily["r1"].abs() > daily["abs_r1_median20"]
    daily["high_first_bar_volume"] = daily["first_bar_volume"] > daily["first_bar_volume_median20"]
    daily["range_z20"] = daily["first_bar_range"] / daily["first_bar_range_mean20"] - 1.0
    daily["volume_share_z20"] = daily["first_bar_volume_share"] / daily["volume_share_mean20"] - 1.0
    daily["trend_dir20"] = np.where(
        daily["close_prev"] > daily["sma20_prev"],
        1.0,
        np.where(daily["close_prev"] < daily["sma20_prev"], -1.0, 0.0),
    )

    # 统一丢弃缺失值
    daily = daily.dropna(subset=["prev_close", "r1", "r12", "r13"]).copy()
    daily.index.name = "trade_date"
    return daily


def expanding_ols_prediction(
    daily: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "r13",
    min_train_days: int = MIN_TRAIN_DAYS,
) -> pd.Series:
    
    x = daily[feature_cols].to_numpy(dtype=float)
    y = daily[target_col].to_numpy(dtype=float)
    pred = np.full(len(daily), np.nan)

    # 每一天的预测都只使用 t-1 及之前的数据
    for i in range(min_train_days, len(daily)):
        train_x = x[:i]
        train_y = y[:i]
        mask = np.isfinite(train_y) & np.isfinite(train_x).all(axis=1)
        if mask.sum() < len(feature_cols) + 10:
            continue
        design = np.column_stack([np.ones(mask.sum()), train_x[mask]])
        beta = np.linalg.lstsq(design, train_y[mask], rcond=None)[0]
        x_now = x[i]
        if np.isfinite(x_now).all():
            pred[i] = np.concatenate(([1.0], x_now)) @ beta

    return pd.Series(pred, index=daily.index, name=f"pred_{'_'.join(feature_cols)}")


def calc_drawdown(returns: pd.Series) -> pd.Series:
    # 净值曲线，相对历史峰值的回撤
    nav = (1.0 + returns.fillna(0.0)).cumprod()
    peak = nav.cummax()
    return nav / peak - 1.0


def performance_metrics(
    returns: pd.Series,
    positions: pd.Series,
    annual_days: int = ANNUAL_TRADING_DAYS,
) -> dict[str, float]:
    # 将空值补成 0 后计算总收益、年化、Sharpe、最大回撤等
    returns = returns.fillna(0.0).astype(float)
    positions = positions.fillna(0.0).astype(float)
    n = len(returns)
    nav = (1.0 + returns).cumprod()
    total_return = nav.iloc[-1] - 1.0 if n else np.nan
    ann_return = (nav.iloc[-1] ** (annual_days / n) - 1.0) if n else np.nan
    ann_vol = returns.std(ddof=1) * np.sqrt(annual_days) if n > 1 else np.nan
    sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(annual_days) if returns.std(ddof=1) > 0 else np.nan
    drawdown = calc_drawdown(returns)
    max_drawdown = drawdown.min() if not drawdown.empty else np.nan
    calmar = ann_return / abs(max_drawdown) if max_drawdown and max_drawdown < 0 else np.nan
    win_rate = (returns > 0).mean() if n else np.nan
    trade_rate = (positions != 0).mean() if n else np.nan
    avg_abs_position = positions.abs().mean() if n else np.nan

    return {
        "sample_days": n,
        "total_return": total_return,
        "annual_return": ann_return,
        "annual_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "trade_rate": trade_rate,
        "avg_abs_position": avg_abs_position,
    }


def yearly_metrics(
    returns: pd.Series,
    positions: pd.Series,
    annual_days: int = ANNUAL_TRADING_DAYS,
) -> pd.DataFrame:
    # 按自然年拆分收益
    out = []
    years = sorted(set(returns.index.year))
    for year in years:
        mask = returns.index.year == year
        metrics = performance_metrics(returns.loc[mask], positions.loc[mask], annual_days=annual_days)
        metrics["year"] = year
        out.append(metrics)
    return pd.DataFrame(out)


def build_baseline_strategies(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = pd.DataFrame(index=daily.index)
    # 信号为正则尾盘做多，为负则尾盘做空
    positions["sign_r1"] = np.where(daily["r1"] > 0.0, 1.0, -1.0)
    positions["sign_r12"] = np.where(daily["r12"] > 0.0, 1.0, -1.0)
    # 双信号策略只有在 r1 和 r12 同向时才交易
    positions["sign_r1_r12"] = np.where(
        (daily["r1"] > 0.0) & (daily["r12"] > 0.0),
        1.0,
        np.where((daily["r1"] <= 0.0) & (daily["r12"] <= 0.0), -1.0, 0.0),
    )
    returns = positions.mul(daily["r13"], axis=0)
    return positions, returns


def build_weighted_strategies(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_specs = {
        "weight_r1": ["r1"],
        "weight_r12": ["r12"],
        "weight_r1_r12": ["r1", "r12"],
    }
    positions = pd.DataFrame(index=daily.index)
    predictions = pd.DataFrame(index=daily.index)
    sigma2 = daily["r13"].shift(1).expanding().var(ddof=1)

    # 三种权重策略分别对应 r1、r12、r1+r12 三套预测模型
    for name, feats in model_specs.items():
        pred = expanding_ols_prediction(daily, feats)
        # 研报 4.2 的核心是用预测收益和历史方差决定仓位，再按上下限截断杠杆。
        weight = (pred / sigma2) / RISK_AVERSION
        positions[name] = weight.clip(*WEIGHT_BOUNDS).fillna(0.0)
        predictions[name] = pred

    returns = positions.mul(daily["r13"], axis=0)
    return positions, returns, predictions


def search_enhanced_strategies(
    daily: pd.DataFrame,
    top_n: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 在入场前叠加更透明的过滤条件
    base_signals = {
        "r1": pd.Series(np.where(daily["r1"] > 0.0, 1.0, -1.0), index=daily.index),
        "r12": pd.Series(np.where(daily["r12"] > 0.0, 1.0, -1.0), index=daily.index),
        "r1_r12": pd.Series(
            np.where(
                (daily["r1"] > 0.0) & (daily["r12"] > 0.0),
                1.0,
                np.where((daily["r1"] <= 0.0) & (daily["r12"] <= 0.0), -1.0, 0.0),
            ),
            index=daily.index,
        ),
    }
    # 过滤器：纯趋势、开盘波动放大、开盘量能放大，以及二者同时满足
    extra_filters = {
        "trend_only": pd.Series(True, index=daily.index),
        "trend_r1_abs": daily["high_vol_regime"].fillna(False),
        "trend_bar_vol": daily["high_first_bar_volume"].fillna(False),
        "trend_both": (daily["high_vol_regime"] & daily["high_first_bar_volume"]).fillna(False),
    }

    candidate_rows: list[dict[str, float | str]] = []
    all_positions: dict[str, pd.Series] = {}
    all_returns: dict[str, pd.Series] = {}
    predictions = pd.DataFrame(index=daily.index)

    # 遍历均线窗口和过滤器
    for signal_name, signal in base_signals.items():
        for ma_window in [5, 10, 20, 30, 40]:
            sma = daily["day_close"].shift(1).rolling(ma_window).mean()
            trend_dir = pd.Series(
                np.where(
                    daily["close_prev"] > sma,
                    1.0,
                    np.where(daily["close_prev"] < sma, -1.0, 0.0),
                ),
                index=daily.index,
            )
            for filter_name, filter_mask in extra_filters.items():
                strategy_name = f"enh_{signal_name}_ma{ma_window}_{filter_name}"
                position = pd.Series(
                    np.where((trend_dir == signal) & filter_mask, signal, 0.0),
                    index=daily.index,
                    name=strategy_name,
                ).fillna(0.0)
                strategy_return = position * daily["r13"]
                metrics = performance_metrics(strategy_return, position)
                metrics["strategy"] = strategy_name
                metrics["family"] = "trend_filter"
                candidate_rows.append(metrics)
                all_positions[strategy_name] = position
                all_returns[strategy_name] = strategy_return

    # 多因子加权版本，观察解释变量扩充后是否优于原始 4.2 配置。
    enh_features = ["r1", "r12", "trend_gap20", "range_z20", "volume_share_z20"]
    pred = expanding_ols_prediction(daily, enh_features)
    sigma2 = daily["r13"].shift(1).expanding().var(ddof=1)
    multi_name = "enh_weighted_multifactor"
    multi_pos = ((pred / sigma2) / RISK_AVERSION).clip(*WEIGHT_BOUNDS).fillna(0.0)
    multi_ret = multi_pos * daily["r13"]
    multi_metrics = performance_metrics(multi_ret, multi_pos)
    multi_metrics["strategy"] = multi_name
    multi_metrics["family"] = "weighted_multifactor"
    candidate_rows.append(multi_metrics)
    all_positions[multi_name] = multi_pos
    all_returns[multi_name] = multi_ret
    predictions[multi_name] = pred

    # 输出交易频率不太低的策略
    candidates = pd.DataFrame(candidate_rows)
    candidates = candidates.sort_values(
        ["sharpe_ratio", "annual_return", "trade_rate"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    eligible = candidates[candidates["trade_rate"] >= 0.05].copy()
    selected_names = eligible.head(top_n)["strategy"].tolist()

    positions = pd.DataFrame({name: all_positions[name] for name in selected_names}, index=daily.index)
    returns = pd.DataFrame({name: all_returns[name] for name in selected_names}, index=daily.index)
    return positions, returns, predictions, candidates


def make_overall_metrics(returns: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    # 对每条策略逐一汇总总体样本指标，按 Sharpe 和年化收益排序
    rows = []
    for col in returns.columns:
        metrics = performance_metrics(returns[col], positions[col])
        metrics["strategy"] = col
        rows.append(metrics)
    result = pd.DataFrame(rows)
    return result[
        [
            "strategy",
            "sample_days",
            "total_return",
            "annual_return",
            "annual_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "calmar_ratio",
            "win_rate",
            "trade_rate",
            "avg_abs_position",
        ]
    ].sort_values(["sharpe_ratio", "annual_return"], ascending=False)


def make_yearly_metrics(returns: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    # 输出成长表结构
    pieces = []
    for col in returns.columns:
        yearly = yearly_metrics(returns[col], positions[col])
        yearly.insert(0, "strategy", col)
        pieces.append(yearly)
    result = pd.concat(pieces, ignore_index=True)
    return result[
        [
            "strategy",
            "year",
            "sample_days",
            "total_return",
            "annual_return",
            "annual_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "calmar_ratio",
            "win_rate",
            "trade_rate",
            "avg_abs_position",
        ]
    ]


def plot_overview(
    daily: pd.DataFrame,
    sign_returns: pd.DataFrame,
    weight_returns: pd.DataFrame,
    enhanced_returns: pd.DataFrame,
    output_path: Path,
) -> None:
    # 总览图：价格、原始策略、权重策略、增强策略”四个面板
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    price_nav = daily["day_close"] / daily["day_close"].iloc[0]
    axes[0, 0].plot(daily.index, price_nav, label="平安银行", color="black", linewidth=1.5)
    axes[0, 0].set_title("归一化价格走势")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.2)

    # 原始策略净值图
    for col in sign_returns.columns:
        axes[0, 1].plot(sign_returns.index, (1.0 + sign_returns[col]).cumprod(), label=col)
    axes[0, 1].set_title("原始择时策略净值")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.2)

    # 动态权重图展示研报 4.2 的均值-方差仓位版本
    for col in weight_returns.columns:
        axes[1, 0].plot(weight_returns.index, (1.0 + weight_returns[col]).cumprod(), label=col)
    axes[1, 0].set_title("4.2节动态权重净值")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.2)

    # 增强策略图
    for col in enhanced_returns.columns:
        axes[1, 1].plot(enhanced_returns.index, (1.0 + enhanced_returns[col]).cumprod(), label=col)
    axes[1, 1].set_title("增强策略净值")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.2)

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_annual_returns(yearly_df: pd.DataFrame, strategy_names: list[str], output_path: Path) -> None:
    # 逐年柱状图展示靠前策略
    pivot = yearly_df[yearly_df["strategy"].isin(strategy_names)].pivot(index="year", columns="strategy", values="annual_return")
    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.12
    x = np.arange(len(pivot.index))
    for i, col in enumerate(pivot.columns):
        ax.bar(x + i * width, pivot[col].fillna(0.0), width=width, label=col)
    ax.set_xticks(x + width * (len(pivot.columns) - 1) / 2)
    ax.set_xticklabels(pivot.index.astype(str))
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("逐年年化收益率")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    # 默认使用当前目录原地覆盖。
    parser = argparse.ArgumentParser(description="回测华安金工日内动量择时策略。")
    parser.add_argument("--csv-path", type=Path, default=None, help="30 分钟 K 线 CSV 路径。")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent, help="输出目录。")
    args = parser.parse_args()

    # 读取原始分时数据
    csv_path = args.csv_path or locate_default_csv()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    intraday = load_intraday_csv(csv_path)
    daily = build_daily_frame(intraday)

    # 依次回测原始三类策略、4.2 动态权重策略和增强策略
    sign_positions, sign_returns = build_baseline_strategies(daily)
    weight_positions, weight_returns, weight_predictions = build_weighted_strategies(daily)
    enhanced_positions, enhanced_returns, enhanced_predictions, enhanced_candidates = search_enhanced_strategies(daily)

    # 中间结果保留
    all_positions = pd.concat([sign_positions, weight_positions, enhanced_positions], axis=1)
    all_returns = pd.concat([sign_returns, weight_returns, enhanced_returns], axis=1)
    all_predictions = pd.concat([weight_predictions, enhanced_predictions], axis=1)

    # 总体指标和逐年指标分别输出
    overall_metrics = make_overall_metrics(all_returns, all_positions)
    yearly_df = make_yearly_metrics(all_returns, all_positions)

    # 结果文件原地覆盖
    overall_metrics.to_csv(output_dir / "overall_metrics.csv", index=False, encoding="utf-8-sig")
    yearly_df.to_csv(output_dir / "yearly_metrics.csv", index=False, encoding="utf-8-sig")
    enhanced_candidates.to_csv(output_dir / "enhanced_candidate_metrics.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(output_dir / "daily_signal_frame.csv", encoding="utf-8-sig")
    all_positions.to_csv(output_dir / "positions.csv", encoding="utf-8-sig")
    all_returns.to_csv(output_dir / "strategy_returns.csv", encoding="utf-8-sig")
    all_predictions.to_csv(output_dir / "predictions.csv", encoding="utf-8-sig")

    # 图表
    plot_overview(
        daily=daily,
        sign_returns=sign_returns,
        weight_returns=weight_returns,
        enhanced_returns=enhanced_returns,
        output_path=output_dir / "overview.png",
    )
    plot_annual_returns(
        yearly_df=yearly_df,
        strategy_names=overall_metrics.head(6)["strategy"].tolist(),
        output_path=output_dir / "yearly_annual_returns.png",
    )

    # 终端摘要
    print("回测完成。")
    print(f"完整交易日数量：{len(daily)}")
    print(f"样本区间：{daily.index.min().date()} 至 {daily.index.max().date()}")
    print("\n按 Sharpe 排序的前几名策略：")
    print(
        overall_metrics[["strategy", "annual_return", "sharpe_ratio", "max_drawdown"]]
        .head(8)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
