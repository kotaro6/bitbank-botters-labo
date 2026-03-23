from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

try:
    from scipy.stats import binomtest
except Exception:  # noqa: BLE001
    binomtest = None


ANNUALIZATION_DAYS = 365
_JP_FONT_INITIALIZED = False


def _ensure_japanese_font() -> None:
    """図の日本語文字化けを避けるため、利用可能な日本語フォントを優先設定する。"""
    global _JP_FONT_INITIALIZED
    if _JP_FONT_INITIALIZED:
        return

    candidates = [
        "IPAexGothic",
        "IPAGothic",
        "Noto Sans CJK JP",
        "Noto Sans JP",
        "Source Han Sans JP",
        "Hiragino Sans",
        "Hiragino Kaku Gothic ProN",
        "Yu Gothic",
        "YuGothic",
        "Meiryo",
        "MS Gothic",
    ]
    available = {str(f.name) for f in fm.fontManager.ttflist}
    current_sans = list(plt.rcParams.get("font.sans-serif", []))

    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = [name, "sans-serif"]
            plt.rcParams["font.sans-serif"] = [name, *current_sans]
            plt.rcParams["axes.unicode_minus"] = False
            _JP_FONT_INITIALIZED = True
            return

    # 最低限、負号崩れだけは防ぐ
    plt.rcParams["axes.unicode_minus"] = False
    _JP_FONT_INITIALIZED = True


DEFAULT_CONFIG: dict[str, Any] = {
    "curve_start_day": 14,
    "curve_end_day": 59,
    "month1_end_day": 29,
    "month2_end_day": 59,
    "worst_n": 20,
    "min_obs": 20,
    "window_bins": [14, 30, 60, 90, 120, 150, 180],
    "window_labels": ["14-29", "30-59", "60-89", "90-119", "120-149", "150-179"],
    "early_windows": ["30-59", "60-89"],
    "late_windows": ["120-149", "150-179"],
    "win_rate_threshold_pct": 55.0,
    "input_source": "unknown",
}


def _merge_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_CONFIG)
    if config:
        merged.update(config)
    return merged


def _validate_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} に必要列がありません: {missing}")


def _to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    return s.astype(str).str.lower().map(mapping).fillna(False)


def _iqr(values: pd.Series) -> float:
    clean = values.dropna()
    if clean.empty:
        return float("nan")
    return float(clean.quantile(0.75) - clean.quantile(0.25))


def _classification(row: pd.Series) -> str:
    if bool(row.get("is_cost_killed", False)):
        return "コスト負け型"
    peak_cum = row.get("peak_cum_net", np.nan)
    if pd.notna(peak_cum) and peak_cum <= 0:
        return "初期から苦戦型"
    tail_ratio = row.get("tail_loss_ratio", np.nan)
    if pd.notna(tail_ratio) and tail_ratio >= 0.5:
        return "テール主導型"
    p2t_days = row.get("peak_to_trough_days", np.nan)
    if pd.notna(p2t_days) and pd.notna(tail_ratio) and p2t_days >= 90 and tail_ratio < 0.5:
        return "じわじわ下落型"
    corr = row.get("corr_zscore_next", np.nan)
    if pd.notna(corr) and corr > 0:
        return "方向不一致型"
    return "その他"


def _build_loss_tags(row: pd.Series) -> str:
    tags: list[str] = []
    corr = row.get("corr_zscore_next", np.nan)
    tail_ratio = row.get("tail_loss_ratio", np.nan)
    p2t_days = row.get("peak_to_trough_days", np.nan)
    peak_cum = row.get("peak_cum_net", np.nan)

    if pd.notna(corr) and corr > 0:
        tags.append("方向不一致")
    if pd.notna(tail_ratio) and tail_ratio >= 0.5:
        tags.append("テール主導")
    if pd.notna(p2t_days) and pd.notna(tail_ratio) and p2t_days >= 90 and tail_ratio < 0.5:
        tags.append("じわじわ下落")
    if pd.notna(peak_cum) and peak_cum <= 0:
        tags.append("初期から苦戦")
    if not tags:
        tags.append("その他")
    return "/".join(tags)


def _prepare_inputs(summary_df: pd.DataFrame, daily_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = summary_df.copy()
    daily = daily_df.copy()

    _validate_columns(summary, ["symbol", "sharpe_net", "is_listing_censored"], "summary_df")
    _validate_columns(daily, ["symbol", "date", "event_day", "pnl_net"], "daily_df")

    summary["symbol"] = summary["symbol"].astype(str)
    daily["symbol"] = daily["symbol"].astype(str)
    summary["is_listing_censored"] = _to_bool_series(summary["is_listing_censored"])

    daily["date"] = pd.to_datetime(daily["date"], utc=True, errors="coerce")
    daily["event_day"] = pd.to_numeric(daily["event_day"], errors="coerce")
    daily["pnl_net"] = pd.to_numeric(daily["pnl_net"], errors="coerce")
    daily = daily.dropna(subset=["date", "event_day", "pnl_net"]).copy()
    daily["event_day"] = daily["event_day"].astype(int)

    if "is_listing_censored" not in daily.columns:
        flags = summary[["symbol", "is_listing_censored"]].drop_duplicates()
        daily = daily.merge(flags, on="symbol", how="left")
    else:
        daily["is_listing_censored"] = _to_bool_series(daily["is_listing_censored"])

    return summary, daily


def _build_curve_tables(summary: pd.DataFrame, daily: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    start_day = int(cfg["curve_start_day"])
    end_day = int(cfg["curve_end_day"])
    m1_end = int(cfg["month1_end_day"])
    m2_end = int(cfg["month2_end_day"])

    curve = daily[daily["event_day"].between(start_day, end_day)].copy()
    curve = curve.sort_values(["symbol", "event_day", "date"]).copy()
    curve["cum_net_event"] = curve.groupby("symbol", as_index=False)["pnl_net"].cumsum()

    curve_eventday_summary = (
        curve.groupby("event_day", observed=True)
        .agg(
            n_symbols=("symbol", "nunique"),
            median=("cum_net_event", "median"),
            mean=("cum_net_event", "mean"),
            q10=("cum_net_event", lambda x: x.quantile(0.10)),
            q25=("cum_net_event", lambda x: x.quantile(0.25)),
            q75=("cum_net_event", lambda x: x.quantile(0.75)),
            q90=("cum_net_event", lambda x: x.quantile(0.90)),
        )
        .reset_index()
    )

    ret_m1 = (
        curve[curve["event_day"].between(start_day, m1_end)]
        .groupby("symbol", observed=True)["pnl_net"]
        .sum()
        .rename("ret_m1_14_29")
    )
    ret_m2 = (
        curve[curve["event_day"].between(m1_end + 1, m2_end)]
        .groupby("symbol", observed=True)["pnl_net"]
        .sum()
        .rename("ret_m2_30_59")
    )
    cum_m2_end = (
        curve[curve["event_day"] == m2_end]
        .set_index("symbol")["cum_net_event"]
        .rename("cum_m2_end_14_59")
    )

    curve_symbol_month = pd.concat([ret_m1, ret_m2, cum_m2_end], axis=1).reset_index()
    curve_symbol_month = curve_symbol_month.merge(
        summary[["symbol", "is_listing_censored"]].drop_duplicates(),
        on="symbol",
        how="left",
    )

    return {
        "curve_symbol_daily": curve,
        "curve_eventday_summary": curve_eventday_summary,
        "curve_symbol_month": curve_symbol_month,
    }


def _build_window_tables(summary: pd.DataFrame, daily: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    bins = list(cfg["window_bins"])
    labels = list(cfg["window_labels"])
    min_obs = int(cfg["min_obs"])

    daily_event = daily[daily["event_day"] >= bins[0]].copy()
    daily_event["window"] = pd.cut(
        daily_event["event_day"],
        bins=bins,
        labels=labels,
        right=False,
    )
    daily_event = daily_event[daily_event["window"].notna()].copy()

    sym_flags = summary[["symbol", "is_listing_censored"]].drop_duplicates()
    daily_event = daily_event.drop(columns=["is_listing_censored"], errors="ignore").merge(
        sym_flags,
        on="symbol",
        how="left",
    )

    symbol_window = (
        daily_event.groupby(["symbol", "window", "is_listing_censored"], observed=True)["pnl_net"]
        .agg(obs="size", mean_pnl="mean", std_pnl="std", return_net="sum")
        .reset_index()
    )
    symbol_window["std_pnl"] = symbol_window["std_pnl"].fillna(0.0)
    symbol_window["sharpe_net"] = np.where(
        (symbol_window["obs"] >= min_obs) & (symbol_window["std_pnl"] > 0),
        symbol_window["mean_pnl"] / symbol_window["std_pnl"] * np.sqrt(ANNUALIZATION_DAYS),
        np.nan,
    )
    symbol_window["full_30_obs"] = symbol_window["obs"] >= 30

    s180_symbols = symbol_window[
        (symbol_window["window"] == labels[-1]) & (symbol_window["full_30_obs"])
    ]["symbol"].drop_duplicates()

    sample_sets = {
        "S_all": symbol_window,
        "S_nc": symbol_window[symbol_window["is_listing_censored"] == False],
        "S_180": symbol_window[symbol_window["symbol"].isin(s180_symbols)],
    }

    summary_rows: list[pd.DataFrame] = []
    paired_rows: list[pd.DataFrame] = []
    paired_stats_rows: list[dict[str, Any]] = []

    early_windows = list(cfg["early_windows"])
    late_windows = list(cfg["late_windows"])

    for sample_set, sub in sample_sets.items():
        g = (
            sub.groupby("window", observed=True)
            .agg(
                n_symbols=("symbol", "nunique"),
                n_valid_sharpe=("sharpe_net", lambda s: int(s.notna().sum())),
                n_full_30=("full_30_obs", "sum"),
                median_sharpe=("sharpe_net", "median"),
                mean_sharpe=("sharpe_net", "mean"),
                iqr_sharpe=("sharpe_net", _iqr),
                median_return=("return_net", "median"),
            )
            .reset_index()
        )
        g.insert(0, "sample_set", sample_set)
        summary_rows.append(g)

        pivot = sub.pivot_table(
            index="symbol",
            columns="window",
            values="sharpe_net",
            observed=True,
        )
        for w in early_windows + late_windows:
            if w not in pivot.columns:
                pivot[w] = np.nan
        pair = pd.DataFrame(index=pivot.index)
        pair["sample_set"] = sample_set
        pair["symbol"] = pair.index.astype(str)
        pair["early"] = pivot[early_windows].mean(axis=1)
        pair["late"] = pivot[late_windows].mean(axis=1)
        pair = pair.dropna(subset=["early", "late"]).copy()
        pair["diff"] = pair["early"] - pair["late"]
        paired_rows.append(pair.reset_index(drop=True))

        n = int(len(pair))
        wins = int((pair["diff"] > 0).sum()) if n > 0 else 0
        median_diff = float(pair["diff"].median()) if n > 0 else float("nan")
        mean_diff = float(pair["diff"].mean()) if n > 0 else float("nan")
        win_rate = float(wins / n * 100) if n > 0 else float("nan")
        pvalue = float("nan")
        if n > 0 and binomtest is not None:
            try:
                pvalue = float(binomtest(wins, n, p=0.5, alternative="greater").pvalue)
            except Exception:  # noqa: BLE001
                pvalue = float("nan")

        paired_stats_rows.append(
            {
                "sample_set": sample_set,
                "n_pairs": n,
                "wins": wins,
                "win_rate_diff": win_rate,
                "median_diff": median_diff,
                "mean_diff": mean_diff,
                "sign_test_pvalue": pvalue,
            }
        )

    window_summary = pd.concat(summary_rows, ignore_index=True)
    paired_diff = pd.concat(paired_rows, ignore_index=True)
    paired_stats = pd.DataFrame(paired_stats_rows)

    return {
        "window_symbol_metrics": symbol_window,
        "window_summary": window_summary,
        "paired_diff": paired_diff,
        "paired_stats": paired_stats,
    }


def _build_worst_diagnostics(summary: pd.DataFrame, daily: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    worst_n = int(cfg["worst_n"])
    worst = summary.nsmallest(worst_n, "sharpe_net").copy()

    daily_sorted = daily.sort_values(["symbol", "date"]).copy()
    extra_rows: list[dict[str, Any]] = []
    for sym in worst["symbol"].astype(str).tolist():
        sub = daily_sorted[daily_sorted["symbol"] == sym]
        if sub.empty:
            continue
        sub = sub.sort_values("date").copy()
        sub["cum_net"] = sub["pnl_net"].cumsum()
        peak_pos = int(sub["cum_net"].values.argmax())
        trough_pos = int(sub["cum_net"].values.argmin())
        peak = sub.iloc[peak_pos]
        trough = sub.iloc[trough_pos]
        worst_1d = float(sub["pnl_net"].min())
        worst_5d = float(sub["pnl_net"].rolling(5, min_periods=1).sum().min())
        worst_1d_pos = int(sub["pnl_net"].values.argmin())
        worst_1d_event_day = float(sub.iloc[worst_1d_pos]["event_day"])
        total_ret = float(sub["pnl_net"].sum())
        tail_ratio = abs(worst_5d) / abs(total_ret) if total_ret < 0 else 0.0
        peak_to_trough_days = float((trough["date"] - peak["date"]).days)
        extra_rows.append(
            {
                "symbol": sym,
                "worst_1d_net": worst_1d,
                "worst_5d_net": worst_5d,
                "worst_1d_event_day": worst_1d_event_day,
                "tail_loss_ratio": tail_ratio,
                "peak_event_day": float(peak["event_day"]),
                "trough_event_day": float(trough["event_day"]),
                "peak_to_trough_days": peak_to_trough_days,
                "peak_cum_net": float(peak["cum_net"]),
            }
        )

    extra = pd.DataFrame(extra_rows)
    worst = worst.merge(extra, on="symbol", how="left")

    for col in ["sharpe_gross", "sharpe_net"]:
        if col not in worst.columns:
            worst[col] = np.nan
    if "corr_zscore_next" not in worst.columns:
        worst["corr_zscore_next"] = np.nan
    for col in ["tail_loss_ratio", "peak_to_trough_days", "peak_cum_net"]:
        if col not in worst.columns:
            worst[col] = np.nan

    worst["is_cost_killed"] = (worst["sharpe_gross"] > 0) & (worst["sharpe_net"] <= 0)
    worst["primary_loss_type"] = worst.apply(_classification, axis=1)
    worst["loss_tags"] = worst.apply(_build_loss_tags, axis=1)
    # backward compatibility
    worst["failure_type"] = worst["primary_loss_type"]
    worst = worst.sort_values("sharpe_net", ascending=True).reset_index(drop=True)
    return worst


def _build_final_decision(artifacts: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    paired_stats = artifacts["paired_stats"]
    summary = artifacts["input_summary"]

    s_nc = paired_stats[paired_stats["sample_set"] == "S_nc"]
    s_all = paired_stats[paired_stats["sample_set"] == "S_all"]
    s_180 = paired_stats[paired_stats["sample_set"] == "S_180"]

    median_diff_nc = float(s_nc["median_diff"].iloc[0]) if len(s_nc) else float("nan")
    win_rate_nc = float(s_nc["win_rate_diff"].iloc[0]) if len(s_nc) else float("nan")
    sign_test_p_nc = float(s_nc["sign_test_pvalue"].iloc[0]) if len(s_nc) else float("nan")
    win_rate_threshold = float(cfg.get("win_rate_threshold_pct", 55.0))

    robust_positive = False
    for s in [s_all, s_180]:
        if len(s) and pd.notna(s["median_diff"].iloc[0]) and s["median_diff"].iloc[0] > 0:
            robust_positive = True

    cond_main = pd.notna(median_diff_nc) and median_diff_nc > 0
    cond_win = pd.notna(win_rate_nc) and win_rate_nc >= win_rate_threshold

    if cond_main and cond_win and robust_positive:
        verdict = "採択"
    elif cond_main and cond_win and not robust_positive:
        verdict = "グレー"
    else:
        verdict = "棄却"

    reasons = []
    if not cond_main:
        if pd.notna(median_diff_nc):
            reasons.append(f"上場初期−後半の中央値差がプラスにならなかった（実測 {median_diff_nc:+.4f}）")
        else:
            reasons.append("上場初期−後半の中央値差を算出できなかった")
    if not cond_win:
        if pd.notna(win_rate_nc):
            reasons.append(
                f"上場初期が優位だった銘柄割合が基準未達（実測 {win_rate_nc:.2f}% < 基準 {win_rate_threshold:.0f}%）"
            )
        else:
            reasons.append(f"上場初期が優位だった銘柄割合を算出できず、{win_rate_threshold:.0f}%基準を判定できなかった")
    if cond_main and cond_win and not robust_positive:
        reasons.append("対象母集団を変えると同じ傾向が弱かった")

    if verdict == "採択":
        verdict_human = "仮説を支持（上場直後優位の傾向を確認）"
    elif verdict == "グレー":
        verdict_human = "一部支持（主分析は支持だが、頑健性は弱い）"
    else:
        verdict_human = "仮説は支持されず（上場直後優位を恒常的とは判断できない）"

    payload = {
        "verdict": verdict,
        "verdict_human": verdict_human,
        "target_hypothesis": "上場直後（early=30-89日）の方が後半（late=120-179日）より有利",
        "input_source": cfg.get("input_source", "unknown"),
        "symbols_total": int(summary["symbol"].nunique()),
        "symbols_non_censored": int((~summary["is_listing_censored"]).sum()),
        "curve_window": f"event_day {cfg['curve_start_day']}-{cfg['curve_end_day']}",
        "worst_definition": f"sharpe_net下位{cfg['worst_n']}銘柄",
        "median_diff_s_nc": median_diff_nc,
        "win_rate_diff_s_nc": win_rate_nc,
        "sign_test_pvalue_s_nc": sign_test_p_nc,
        "win_rate_threshold_pct": win_rate_threshold,
        "rule1_median_diff_positive": bool(cond_main),
        "rule2_win_rate_reached": bool(cond_win),
        "robust_positive": bool(robust_positive),
        "reasons": reasons,
        "not_confirmed": [
            "未確認: walk-forwardのOOS検証で同結論か",
            "未確認: FRコスト込み実現PnLで同結論か",
            "未確認: 流動性・ボラ統制後も同結論か",
        ],
    }
    return payload


def run_homework_report(summary_df: pd.DataFrame, daily_df: pd.DataFrame, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """宿題1/2レポートで使う集計・診断を一括で実行する。"""
    cfg = _merge_config(config)
    summary, daily = _prepare_inputs(summary_df, daily_df)

    curve_tables = _build_curve_tables(summary, daily, cfg)
    window_tables = _build_window_tables(summary, daily, cfg)
    worst = _build_worst_diagnostics(summary, daily, cfg)

    artifacts: dict[str, Any] = {
        "input_summary": summary,
        "input_daily": daily,
        "curve_symbol_daily": curve_tables["curve_symbol_daily"],
        "curve_eventday_summary": curve_tables["curve_eventday_summary"],
        "curve_symbol_month": curve_tables["curve_symbol_month"],
        "window_symbol_metrics": window_tables["window_symbol_metrics"],
        "window_summary": window_tables["window_summary"],
        "paired_diff": window_tables["paired_diff"],
        "paired_stats": window_tables["paired_stats"],
        "worst_diagnostics": worst,
        "config": cfg,
    }
    artifacts["final_decision_payload"] = _build_final_decision(artifacts, cfg)
    return artifacts


def _safe_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _safe_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_value(v) for v in value]
    return value


def export_artifacts(artifacts: dict[str, Any], out_dir: str | Path) -> dict[str, str]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    file_map: dict[str, str] = {}
    csv_targets: dict[str, str] = {
        "input_summary": "homework_input_summary.csv",
        "input_daily": "homework_input_daily.csv",
        "curve_eventday_summary": "altcoin_basis_hw_post_listing_curve_14_59_eventday_summary.csv",
        "curve_symbol_daily": "altcoin_basis_hw_post_listing_curve_14_59_symbol_daily.csv",
        "curve_symbol_month": "altcoin_basis_hw_post_listing_curve_14_59_symbol_month.csv",
        "window_symbol_metrics": "altcoin_basis_hw_post_listing_window_symbol_metrics.csv",
        "window_summary": "altcoin_basis_hw_post_listing_window_summary.csv",
        "paired_diff": "altcoin_basis_hw_post_listing_paired_diff.csv",
        "worst_diagnostics": "altcoin_basis_hw_worst_diagnostics.csv",
    }

    for key, filename in csv_targets.items():
        value = artifacts.get(key)
        if isinstance(value, pd.DataFrame):
            target = out_path / filename
            value.to_csv(target, index=False)
            # 配布先依存の絶対パスは残さず、manifestには相対パスを保存する。
            file_map[key] = filename

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": _safe_value(artifacts.get("config", {})),
        "final_decision_payload": _safe_value(artifacts.get("final_decision_payload", {})),
        "files": file_map,
    }
    manifest_path = out_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    file_map["manifest"] = "manifest.json"

    return file_map


def print_final_homework_answer(artifacts: dict[str, Any]) -> None:
    payload = artifacts["final_decision_payload"]
    worst = artifacts["worst_diagnostics"].copy()
    summary_all = artifacts["input_summary"].copy()
    daily_all = artifacts["input_daily"].copy()
    curve_month = artifacts["curve_symbol_month"]
    curve_event = artifacts["curve_eventday_summary"]

    m1_med = float(curve_month["ret_m1_14_29"].median()) if "ret_m1_14_29" in curve_month else float("nan")
    m2_med = float(curve_month["ret_m2_30_59"].median()) if "ret_m2_30_59" in curve_month else float("nan")
    m2_cum_med = float(curve_month["cum_m2_end_14_59"].median()) if "cum_m2_end_14_59" in curve_month else float("nan")
    spread_10_90_last = float("nan")
    if {"event_day", "q10", "q90"}.issubset(curve_event.columns) and len(curve_event) > 0:
        last_row = curve_event.sort_values("event_day").iloc[-1]
        spread_10_90_last = float(last_row["q90"] - last_row["q10"])

    direction_mismatch = float("nan")
    direction_mismatch_all = float("nan")
    cost_killed_rate = float("nan")
    cost_killed_all = float("nan")
    tail_driven_rate = float("nan")
    if "corr_zscore_next" in worst.columns and len(worst) > 0:
        direction_mismatch = float((worst["corr_zscore_next"] > 0).mean() * 100.0)
    if "corr_zscore_next" in summary_all.columns and len(summary_all) > 0:
        direction_mismatch_all = float((summary_all["corr_zscore_next"] > 0).mean() * 100.0)
    if {"sharpe_gross", "sharpe_net"}.issubset(worst.columns) and len(worst) > 0:
        cost_killed_rate = float(((worst["sharpe_gross"] > 0) & (worst["sharpe_net"] <= 0)).mean() * 100.0)
    if {"sharpe_gross", "sharpe_net"}.issubset(summary_all.columns) and len(summary_all) > 0:
        cost_killed_all = float(
            ((summary_all["sharpe_gross"] > 0) & (summary_all["sharpe_net"] <= 0)).mean() * 100.0
        )
    if "tail_loss_ratio" in worst.columns and len(worst) > 0:
        tail_driven_rate = float((worst["tail_loss_ratio"] >= 0.5).mean() * 100.0)

    timing_df = pd.DataFrame()
    if len(worst) > 0 and {"symbol", "date", "event_day", "pnl_net"}.issubset(daily_all.columns):
        daily = daily_all.copy()
        daily["symbol"] = daily["symbol"].astype(str)
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce", utc=True).dt.tz_convert(None)
        daily["event_day"] = pd.to_numeric(daily["event_day"], errors="coerce")
        daily["pnl_net"] = pd.to_numeric(daily["pnl_net"], errors="coerce")
        daily = daily.dropna(subset=["date", "event_day", "pnl_net"]).copy()

        rows: list[dict[str, Any]] = []
        for sym in worst["symbol"].astype(str):
            d = daily[daily["symbol"] == sym].sort_values("date").copy()
            if d.empty:
                continue
            d["cum_net"] = d["pnl_net"].cumsum()
            peak_pos = int(d["cum_net"].values.argmax())
            trough_pos = int(d["cum_net"].values.argmin())
            worst1d_pos = int(d["pnl_net"].values.argmin())

            peak = d.iloc[peak_pos]
            trough = d.iloc[trough_pos]
            worst1d = d.iloc[worst1d_pos]

            wrow = worst[worst["symbol"].astype(str) == sym].iloc[0]
            rows.append(
                {
                    "symbol": sym,
                    "peak_event_day": float(peak["event_day"]),
                    "trough_event_day": float(trough["event_day"]),
                    "peak_to_trough_days": float((trough["date"] - peak["date"]).days),
                    "worst1d_event_day": float(worst1d["event_day"]),
                    "peak_cum_net": float(peak["cum_net"]),
                    "tail_loss_ratio": float(wrow.get("tail_loss_ratio", np.nan)),
                    "corr_zscore_next": float(wrow.get("corr_zscore_next", np.nan)),
                }
            )
        timing_df = pd.DataFrame(rows)

    main_type_counts = pd.Series(dtype="int64")
    if "primary_loss_type" in worst.columns and len(worst) > 0:
        main_type_counts = worst["primary_loss_type"].value_counts(dropna=False)
    gradual_decline_rate = float("nan")
    initial_struggle_rate = float("nan")
    if len(timing_df) > 0:
        gradual_mask = (timing_df["peak_to_trough_days"] >= 90) & (timing_df["tail_loss_ratio"] < 0.5)
        initial_mask = timing_df["peak_cum_net"] <= 0
        gradual_decline_rate = float(gradual_mask.mean() * 100.0)
        initial_struggle_rate = float(initial_mask.mean() * 100.0)

    top_main_type = "NA"
    top_main_count = 0
    top_main_ratio = float("nan")
    if len(main_type_counts) > 0 and len(worst) > 0:
        top_main_type = str(main_type_counts.index[0])
        top_main_count = int(main_type_counts.iloc[0])
        top_main_ratio = float(top_main_count / len(worst) * 100.0)

    peak_day_med = float(timing_df["peak_event_day"].median()) if "peak_event_day" in timing_df else float("nan")
    trough_day_med = float(timing_df["trough_event_day"].median()) if "trough_event_day" in timing_df else float("nan")
    decline_days_med = float(timing_df["peak_to_trough_days"].median()) if "peak_to_trough_days" in timing_df else float("nan")
    worst1d_day_med = float(timing_df["worst1d_event_day"].median()) if "worst1d_event_day" in timing_df else float("nan")

    def _fmt_signed(x: float) -> str:
        return "NA" if pd.isna(x) else f"{x:+.4f}"

    def _fmt_pct(x: float, nd: int = 2) -> str:
        return "NA" if pd.isna(x) else f"{x:.{nd}f}%"

    def _fmt_day(x: float) -> str:
        return "NA" if pd.isna(x) else f"{x:.1f}日"

    print("=" * 70)
    print("最終まとめ（初心者向け）")
    print("=" * 70)
    print("1) 先に結論（30秒版）")
    print("- 宿題1: day14-59の累積中央値はプラスだが、銘柄差が大きく全銘柄で安定とは言えない。")
    print(
        f"- 宿題2: 主因タイプは『{top_main_type}』が最多（{top_main_count}/{len(worst)}銘柄, {_fmt_pct(top_main_ratio)}）。"
    )
    print("- 補助タグでは『方向不一致』が高頻度で、単発ショックより中長期のじわじわ悪化が多い。")

    threshold = float(payload.get("win_rate_threshold_pct", 55.0))
    margin = threshold - 50.0
    rule1_ok = bool(payload.get("rule1_median_diff_positive", False))
    rule2_ok = bool(payload.get("rule2_win_rate_reached", False))
    robust_ok = bool(payload.get("robust_positive", False))
    verdict = payload.get("verdict", "")
    if verdict == "採択":
        verdict_line = "採用（今回条件では上場直後優位を確認）"
    elif verdict == "グレー":
        verdict_line = "保留（主分析は良いが、追加確認が必要）"
    else:
        verdict_line = "今回は採用しない（上場直後優位の恒常性は確認できず）"
    print(f"- 判定: {verdict_line}")

    print("\n2) 宿題1の結果（上場後2ヶ月: event_day 14-59）")
    print("- day14開始の理由: z-scoreを14日で作るため、day0-13は比較が安定しない。")
    print(f"- 1ヶ月目（day14-29）中央値: {_fmt_signed(m1_med)}")
    print(f"- 2ヶ月目（day30-59）中央値: {_fmt_signed(m2_med)}")
    print(f"- day59累積中央値: {_fmt_signed(m2_cum_med)}")
    print(f"- 最終日の銘柄差（10-90%幅）: {_fmt_signed(spread_10_90_last)}")
    print("- 読み方: 中央値が勝っても、銘柄差が広いと一部で大きく負ける。")

    print("\n3) 宿題2の結果（追加検証を反映）")
    print(f"- 対象: {payload['worst_definition']}（Sharpeは成績の安定度）")
    print(
        "- 負け方の分解: "
        f"方向不一致 {_fmt_pct(direction_mismatch)}（全体 {_fmt_pct(direction_mismatch_all)}） / "
        f"コスト負け {_fmt_pct(cost_killed_rate)}（全体 {_fmt_pct(cost_killed_all)}） / "
        f"テール主導 {_fmt_pct(tail_driven_rate)}"
    )
    print(
        "- いつ下がるか（中央値）: "
        f"peak {_fmt_day(peak_day_med)} -> trough {_fmt_day(trough_day_med)} "
        f"（peak->trough {_fmt_day(decline_days_med)}、worst1d {_fmt_day(worst1d_day_med)}）"
    )
    print(f"- じわじわ下落率（peak_to_trough_days>=90 かつ tail<0.5）: {_fmt_pct(gradual_decline_rate)}")
    print(f"- 初期から苦戦率（peak_cum_net<=0）: {_fmt_pct(initial_struggle_rate)}")
    if len(main_type_counts) > 0:
        print("- 主因タイプ内訳（1銘柄1ラベル）:")
        for k, v in main_type_counts.items():
            print(f"  - {k}: {int(v)}銘柄")
    print("- 補助タグ率（重複可）:")
    print(f"  - 方向不一致: {_fmt_pct(direction_mismatch)}")
    print(f"  - テール主導: {_fmt_pct(tail_driven_rate)}")
    print(f"  - じわじわ下落: {_fmt_pct(gradual_decline_rate)}")
    print(f"  - 初期から苦戦: {_fmt_pct(initial_struggle_rate)}")

    print("\n4) 判定の根拠（反論チェック込み）")
    print(f"- 判定した仮説: {payload['target_hypothesis']}")
    print("- ルールA: 上場初期−後半の中央値差がプラス")
    print(f"  実測 {_fmt_signed(payload['median_diff_s_nc'])} -> {'達成' if rule1_ok else '未達'}")
    print(f"- ルールB: 上場初期が勝つ銘柄割合が{threshold:.0f}%以上")
    print(f"  実測 {_fmt_pct(payload['win_rate_diff_s_nc'])}（基準 {threshold:.0f}%）-> {'達成' if rule2_ok else '未達'}")
    if margin > 0:
        print(f"  {threshold:.0f}%を使う理由: 50%（偶然）より{margin:.0f}pt高く、偶然勝ちを減らすため。")
    print(f"- 頑健性チェック（母集団を変えても同傾向か）: {'達成' if robust_ok else '未達'}")
    print("- 反論チェック: 「中央値がプラスなら十分」とは言い切れない。")
    print("  今回は勝率が基準未達で、偶然の可能性を捨てきれない。")
    if verdict == "棄却":
        print("- 補足: これは戦略全体の否定ではありません。")
        print("  今回の仮説（上場直後が恒常的に有利）が、この条件では確認できなかったという意味です。")
    if pd.notna(payload.get("sign_test_pvalue_s_nc", float("nan"))):
        print(f"- 参考（符号検定p値）: {payload['sign_test_pvalue_s_nc']:.3f}")
    if payload.get("reasons"):
        print("- 判定理由（未達項目）:")
        for r in payload["reasons"]:
            print(f"  - {r}")

    print("\n5) 次の改善アクション（宿題2の意図）")
    print("- 方針: 勝ちルールの追加より先に、負け回避ルールを作って検証する。")
    print("- 候補1: 方向不一致が続く銘柄・期間は逆張り停止またはロット半減。")
    print("- 候補2: テール損失が閾値超えなら一定期間の取引停止（ストッパー）。")
    print("- 候補3: 上場後日数（event_day）帯で許可/不許可を分ける。")
    print("- 候補4: じわじわ下落型に時間損切り（保有日数上限）を入れる。")

    print("\n[未確認（今後の宿題）]")
    for line in payload.get("not_confirmed", []):
        print(f"- {line}")


def plot_curve_eventday(artifacts: dict[str, Any]) -> plt.Figure:
    _ensure_japanese_font()
    df = artifacts["curve_eventday_summary"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["event_day"], df["median"], label="中央値", color="#1f77b4", linewidth=2)
    ax.plot(df["event_day"], df["mean"], label="平均", color="#ff7f0e", linewidth=1.5, alpha=0.9)
    ax.fill_between(df["event_day"], df["q25"], df["q75"], color="#1f77b4", alpha=0.20, label="25-75%")
    ax.fill_between(df["event_day"], df["q10"], df["q90"], color="#1f77b4", alpha=0.10, label="10-90%")
    ax.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax.set_title("上場後2ヶ月以内（分析区間: event_day 14-59）の累積PnL")
    ax.set_xlabel("event_day")
    ax.set_ylabel("累積PnL")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_curve_month_snapshot(artifacts: dict[str, Any]) -> plt.Figure:
    _ensure_japanese_font()
    df = artifacts["curve_symbol_month"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, col, title in [
        (axes[0], "ret_m1_14_29", "1ヶ月目 区間損益（14-29）"),
        (axes[1], "ret_m2_30_59", "2ヶ月目 区間損益（30-59）"),
    ]:
        vals = df[col].dropna()
        ax.hist(vals, bins=30, color="#4C72B0", alpha=0.75)
        if len(vals) > 0:
            ax.axvline(vals.median(), color="#DD8452", linestyle="--", linewidth=2, label=f"median {vals.median():+.4f}")
        ax.axvline(0, color="black", linewidth=1, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("return")
        ax.set_ylabel("count")
        ax.grid(alpha=0.2)
        ax.legend()

    fig.tight_layout()
    return fig


def plot_window_median_sharpe(artifacts: dict[str, Any]) -> plt.Figure:
    _ensure_japanese_font()
    ws = artifacts["window_summary"]
    labels = list(artifacts["config"]["window_labels"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, sample in zip(axes, ["S_nc", "S_all", "S_180"]):
        sub = ws[ws["sample_set"] == sample].copy()
        sub["window"] = pd.Categorical(sub["window"], categories=labels, ordered=True)
        sub = sub.sort_values("window")
        ax.bar(sub["window"].astype(str), sub["median_sharpe"], color="#55A868", alpha=0.85)
        ax.axhline(0, color="black", linewidth=1, alpha=0.5)
        ax.set_title(sample)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.2)
        for i, row in sub.reset_index(drop=True).iterrows():
            ax.text(i, row["median_sharpe"], f"n={int(row['n_symbols'])}", ha="center", va="bottom", fontsize=8)

    axes[0].set_ylabel("median sharpe")
    fig.suptitle("event_day区間別 Sharpe中央値（n併記）")
    fig.tight_layout()
    return fig


def plot_paired_diff_hist(artifacts: dict[str, Any]) -> plt.Figure:
    _ensure_japanese_font()
    paired = artifacts["paired_diff"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, sample in zip(axes, ["S_nc", "S_all", "S_180"]):
        sub = paired[paired["sample_set"] == sample]
        ax.hist(sub["diff"].dropna(), bins=30, color="#8172B2", alpha=0.8)
        if len(sub) > 0:
            ax.axvline(sub["diff"].median(), color="#C44E52", linestyle="--", linewidth=2)
        ax.axvline(0, color="black", linewidth=1, alpha=0.5)
        ax.set_title(f"{sample} (n={len(sub)})")
        ax.set_xlabel("early - late")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("count")
    fig.suptitle("銘柄別 early-late 差分分布")
    fig.tight_layout()
    return fig


def plot_worst_curves(artifacts: dict[str, Any], max_cols: int = 4, top_n: int | None = None) -> plt.Figure:
    _ensure_japanese_font()
    worst = artifacts["worst_diagnostics"].copy()
    daily = artifacts["input_daily"].copy()
    if top_n is not None:
        worst = worst.head(int(top_n)).copy()

    symbols = worst["symbol"].astype(str).tolist()
    n = len(symbols)
    ncols = max_cols
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), sharex=False, sharey=False)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)

    for i, sym in enumerate(symbols):
        r, c = divmod(i, ncols)
        ax = axes_arr[r, c]
        sub = daily[daily["symbol"] == sym].copy()
        sub["date"] = pd.to_datetime(sub["date"], errors="coerce", utc=True).dt.tz_convert(None)
        sub = sub.dropna(subset=["date"]).sort_values("date").copy()
        sub["cum_net"] = sub["pnl_net"].cumsum()
        ax.plot(sub["date"], sub["cum_net"], color="#4C72B0", linewidth=1.5)
        row = worst[worst["symbol"] == sym].iloc[0]
        ax.set_title(f"{sym} | {row['failure_type']}", fontsize=9)
        ax.axhline(0, color="black", linewidth=1, alpha=0.4)
        ax.grid(alpha=0.2)
        if len(sub):
            # 日付は最大4点に間引いて重なりを抑える（いつ下落したかを読み取りやすくする）。
            num_ticks = min(4, len(sub))
            tick_idx = np.linspace(0, len(sub) - 1, num=num_ticks, dtype=int)
            tick_dates = sub.iloc[np.unique(tick_idx)]["date"]
            ax.set_xticks(tick_dates)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
        if nrows > 2 and r < nrows - 1:
            # 行が多いときだけ上段ラベルを隠す。
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.tick_params(axis="x", labelsize=8, rotation=25, pad=1)
            for lbl in ax.get_xticklabels():
                lbl.set_ha("right")

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes_arr[r, c].axis("off")

    fig.suptitle("ワースト銘柄の累積PnL曲線（Net）", y=0.995, fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.93], h_pad=1.6, w_pad=0.9)
    return fig


def build_worst_summary_table(artifacts: dict[str, Any], top_n_plot: int = 6) -> pd.DataFrame:
    """ワースト銘柄の要約表（上位Nを図表示対象としてマーク）を返す。"""
    worst = artifacts["worst_diagnostics"].copy().reset_index(drop=True)
    daily = artifacts["input_daily"].copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce", utc=True).dt.tz_convert(None)

    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(worst.itertuples(index=False), start=1):
        sym = str(row.symbol)
        sub = daily[daily["symbol"] == sym].dropna(subset=["date"]).sort_values("date").copy()
        if sub.empty:
            rows.append(
                {
                    "rank": rank,
                    "graph_target": "図あり" if rank <= top_n_plot else "表のみ",
                    "symbol": sym,
                    "failure_type": row.failure_type,
                    "sharpe_net": row.sharpe_net,
                    "return_net": row.return_net,
                    "dd_net": row.dd_net,
                    "peak_date": pd.NaT,
                    "trough_date": pd.NaT,
                    "peak_to_trough_days": np.nan,
                    "peak_to_trough_drop": np.nan,
                }
            )
            continue

        sub["cum_net"] = sub["pnl_net"].cumsum()
        peak_pos = int(sub["cum_net"].values.argmax())
        trough_pos = int(sub["cum_net"].values.argmin())
        peak_date = sub.iloc[peak_pos]["date"]
        trough_date = sub.iloc[trough_pos]["date"]
        peak_cum = float(sub.iloc[peak_pos]["cum_net"])
        trough_cum = float(sub.iloc[trough_pos]["cum_net"])
        days = (trough_date - peak_date).days if pd.notna(peak_date) and pd.notna(trough_date) else np.nan

        rows.append(
            {
                "rank": rank,
                "graph_target": "図あり" if rank <= top_n_plot else "表のみ",
                "symbol": sym,
                "failure_type": row.failure_type,
                "sharpe_net": float(row.sharpe_net),
                "return_net": float(row.return_net),
                "dd_net": float(row.dd_net),
                "peak_date": peak_date,
                "trough_date": trough_date,
                "peak_to_trough_days": days,
                "peak_to_trough_drop": trough_cum - peak_cum,
            }
        )

    out = pd.DataFrame(rows)
    out = out[
        [
            "rank",
            "graph_target",
            "symbol",
            "failure_type",
            "sharpe_net",
            "return_net",
            "dd_net",
            "peak_date",
            "trough_date",
            "peak_to_trough_days",
            "peak_to_trough_drop",
        ]
    ]
    return out
