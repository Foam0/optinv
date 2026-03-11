from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

FRED_SERIES = {
    "eurusd": "DEXUSEU",
    "brent": "DCOILBRENTEU",
}


def fetch_series(series_id: str, column_name: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    frame = pd.read_csv(url)
    frame.columns = ["date", column_name]
    frame["date"] = pd.to_datetime(frame["date"])
    frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")
    return frame


def write_latex_table(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    latex_frame = frame.copy()
    latex_frame.columns = [str(col).replace("_", r"\_") for col in latex_frame.columns]
    for column in latex_frame.columns:
        latex_frame[column] = latex_frame[column].map(
            lambda value: value.replace("_", r"\_") if isinstance(value, str) else value
        )
    tex = latex_frame.to_latex(
        index=index,
        escape=False,
        float_format=lambda value: f"{value:.4f}",
        na_rep="",
    )
    path.write_text(tex, encoding="utf-8")


def fit_daily_regression(sample: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    reg = sample.copy()
    reg["r_brent_l1"] = reg["r_brent"].shift(1)
    reg["r_eurusd_l1"] = reg["r_eurusd"].shift(1)
    reg = reg.dropna()
    x = sm.add_constant(reg[["r_brent", "r_brent_l1", "r_eurusd_l1"]])
    return sm.OLS(reg["r_eurusd"], x).fit(cov_type="HAC", cov_kwds={"maxlags": 5})


def save_text_summary(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_frames: list[pd.DataFrame] = []
    for name, series_id in FRED_SERIES.items():
        frame = fetch_series(series_id, name)
        frame.to_csv(DATA_DIR / f"{name}_daily.csv", index=False)
        raw_frames.append(frame)

    merged = raw_frames[0]
    for frame in raw_frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")

    merged = merged.sort_values("date").set_index("date")
    daily = merged.dropna().copy()
    daily["l_eurusd"] = np.log(daily["eurusd"])
    daily["l_brent"] = np.log(daily["brent"])
    daily["r_eurusd"] = daily["l_eurusd"].diff()
    daily["r_brent"] = daily["l_brent"].diff()
    daily["rolling_corr_252"] = daily["r_eurusd"].rolling(252).corr(daily["r_brent"])
    daily = daily.dropna()
    daily.to_csv(DATA_DIR / "merged_daily.csv")

    last_complete_month = (daily.index.max().to_period("M") - 1).to_timestamp("M")
    monthly = merged[["eurusd", "brent"]].resample("ME").mean()
    monthly = monthly.loc[:last_complete_month].dropna()
    monthly["l_eurusd"] = np.log(monthly["eurusd"])
    monthly["l_brent"] = np.log(monthly["brent"])
    monthly["r_eurusd"] = monthly["l_eurusd"].diff()
    monthly["r_brent"] = monthly["l_brent"].diff()
    monthly = monthly.dropna()
    monthly.to_csv(DATA_DIR / "merged_monthly.csv")

    summary_stats = pd.DataFrame(
        {
            "series": ["EUR/USD daily log return", "Brent daily log return"],
            "mean": [daily["r_eurusd"].mean(), daily["r_brent"].mean()],
            "std": [daily["r_eurusd"].std(), daily["r_brent"].std()],
            "min": [daily["r_eurusd"].min(), daily["r_brent"].min()],
            "max": [daily["r_eurusd"].max(), daily["r_brent"].max()],
            "obs": [daily["r_eurusd"].count(), daily["r_brent"].count()],
        }
    )
    summary_stats.to_csv(RESULTS_DIR / "summary_stats.csv", index=False)
    write_latex_table(summary_stats, RESULTS_DIR / "summary_stats.tex")

    adf_table = pd.DataFrame(
        [
            {
                "series": "log EUR/USD level",
                "sample": "1999-01 to latest complete month",
                "adf_stat": adfuller(monthly["l_eurusd"], autolag="AIC")[0],
                "p_value": adfuller(monthly["l_eurusd"], autolag="AIC")[1],
            },
            {
                "series": "log Brent level",
                "sample": "1999-01 to latest complete month",
                "adf_stat": adfuller(monthly["l_brent"], autolag="AIC")[0],
                "p_value": adfuller(monthly["l_brent"], autolag="AIC")[1],
            },
            {
                "series": "EUR/USD log return",
                "sample": "1999-02 to latest complete month",
                "adf_stat": adfuller(monthly["r_eurusd"], autolag="AIC")[0],
                "p_value": adfuller(monthly["r_eurusd"], autolag="AIC")[1],
            },
            {
                "series": "Brent log return",
                "sample": "1999-02 to latest complete month",
                "adf_stat": adfuller(monthly["r_brent"], autolag="AIC")[0],
                "p_value": adfuller(monthly["r_brent"], autolag="AIC")[1],
            },
        ]
    )
    adf_table.to_csv(RESULTS_DIR / "adf_tests.csv", index=False)
    write_latex_table(adf_table, RESULTS_DIR / "adf_tests.tex")

    # Daily regressions: full sample and two regimes.
    samples = {
        "Full sample (2010-01-04 to latest)": daily.loc["2010-01-04":],
        "Pre-2022 sample": daily.loc["2010-01-04":"2021-12-31"],
        "2022+ sample": daily.loc["2022-01-01":],
    }

    regime_rows: list[dict[str, float | str]] = []
    full_daily_model = None
    for label, sample in samples.items():
        model = fit_daily_regression(sample)
        if full_daily_model is None:
            full_daily_model = model
        reg = sample.copy().dropna()
        regime_rows.append(
            {
                "sample": label,
                "obs": model.nobs,
                "beta_current_oil": model.params["r_brent"],
                "p_current_oil": model.pvalues["r_brent"],
                "beta_lagged_oil": model.params["r_brent_l1"],
                "p_lagged_oil": model.pvalues["r_brent_l1"],
                "beta_fx_lag": model.params["r_eurusd_l1"],
                "p_fx_lag": model.pvalues["r_eurusd_l1"],
                "r_squared": model.rsquared,
                "corr_oil_fx": reg["r_brent"].corr(reg["r_eurusd"]),
            }
        )

    daily_regime_table = pd.DataFrame(regime_rows)
    daily_regime_table.to_csv(RESULTS_DIR / "daily_regime_regressions.csv", index=False)
    write_latex_table(daily_regime_table, RESULTS_DIR / "daily_regime_regressions.tex")

    full_coeffs = pd.DataFrame(
        {
            "variable": full_daily_model.params.index,
            "coef": full_daily_model.params.values,
            "std_err": full_daily_model.bse.values,
            "p_value": full_daily_model.pvalues.values,
        }
    )
    full_coeffs.to_csv(RESULTS_DIR / "daily_full_coefficients.csv", index=False)
    write_latex_table(full_coeffs, RESULTS_DIR / "daily_full_coefficients.tex")

    # Cointegration and ECM on monthly data.
    cointegration_specs = [
        ("1999-01-31", "2021-12-31", "1999-2021"),
        ("2022-01-31", last_complete_month.strftime("%Y-%m-%d"), "2022-latest"),
        ("1999-01-31", last_complete_month.strftime("%Y-%m-%d"), "Full monthly sample"),
    ]
    cointegration_rows: list[dict[str, float | str]] = []
    ecm_model = None
    ecm_long_run = None
    for start, end, label in cointegration_specs:
        sub = monthly.loc[start:end, ["l_eurusd", "l_brent"]].dropna()
        stat, p_value, _ = coint(sub["l_eurusd"], sub["l_brent"])
        cointegration_rows.append(
            {
                "sample": label,
                "obs": len(sub),
                "coint_stat": stat,
                "p_value": p_value,
            }
        )
        if label == "1999-2021":
            step1 = sm.OLS(sub["l_eurusd"], sm.add_constant(sub["l_brent"])).fit()
            ecm_long_run = step1
            sub = sub.copy()
            sub["ecm"] = step1.resid
            sub["d_eurusd"] = sub["l_eurusd"].diff()
            sub["d_brent"] = sub["l_brent"].diff()
            sub["d_eurusd_l1"] = sub["d_eurusd"].shift(1)
            sub["d_brent_l1"] = sub["d_brent"].shift(1)
            sub["ecm_l1"] = sub["ecm"].shift(1)
            reg = sub.dropna()
            x = sm.add_constant(reg[["d_brent", "d_brent_l1", "d_eurusd_l1", "ecm_l1"]])
            ecm_model = sm.OLS(reg["d_eurusd"], x).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

    cointegration_table = pd.DataFrame(cointegration_rows)
    cointegration_table.to_csv(RESULTS_DIR / "cointegration_tests.csv", index=False)
    write_latex_table(cointegration_table, RESULTS_DIR / "cointegration_tests.tex")

    ecm_coeffs = pd.DataFrame(
        {
            "variable": ecm_model.params.index,
            "coef": ecm_model.params.values,
            "std_err": ecm_model.bse.values,
            "p_value": ecm_model.pvalues.values,
        }
    )
    ecm_coeffs.to_csv(RESULTS_DIR / "ecm_coefficients.csv", index=False)
    write_latex_table(ecm_coeffs, RESULTS_DIR / "ecm_coefficients.tex")

    long_run_table = pd.DataFrame(
        [
            {
                "variable": "const",
                "coef": ecm_long_run.params["const"],
                "std_err": ecm_long_run.bse["const"],
                "p_value": ecm_long_run.pvalues["const"],
            },
            {
                "variable": "log Brent",
                "coef": ecm_long_run.params["l_brent"],
                "std_err": ecm_long_run.bse["l_brent"],
                "p_value": ecm_long_run.pvalues["l_brent"],
            },
        ]
    )
    long_run_table.to_csv(RESULTS_DIR / "long_run_coefficients.csv", index=False)
    write_latex_table(long_run_table, RESULTS_DIR / "long_run_coefficients.tex")

    granger_fx = grangercausalitytests(monthly[["r_eurusd", "r_brent"]], maxlag=3, verbose=False)
    granger_oil = grangercausalitytests(monthly[["r_brent", "r_eurusd"]], maxlag=3, verbose=False)
    granger_table = pd.DataFrame(
        {
            "lag": [1, 2, 3],
            "oil_to_fx_p": [granger_fx[lag][0]["ssr_ftest"][1] for lag in [1, 2, 3]],
            "fx_to_oil_p": [granger_oil[lag][0]["ssr_ftest"][1] for lag in [1, 2, 3]],
        }
    )
    granger_table.to_csv(RESULTS_DIR / "granger_monthly.csv", index=False)
    write_latex_table(granger_table, RESULTS_DIR / "granger_monthly.tex")

    # Figures.
    base = daily.loc["2010-01-04":, ["eurusd", "brent"]].copy()
    normalized = base / base.iloc[0]
    plt.figure(figsize=(9, 4.8))
    plt.plot(normalized.index, normalized["eurusd"], label="EUR/USD, normalized")
    plt.plot(normalized.index, normalized["brent"], label="Brent, normalized")
    plt.title("Normalized EUR/USD and Brent levels")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "normalized_levels.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 4.8))
    plt.plot(daily.loc["2011-01-01":].index, daily.loc["2011-01-01":, "rolling_corr_252"])
    plt.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    plt.title("252-day rolling correlation of EUR/USD and Brent returns")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "rolling_correlation.png", dpi=180)
    plt.close()

    scatter = daily.loc["2010-01-04":, ["r_brent", "r_eurusd"]]
    plt.figure(figsize=(6.2, 5.3))
    plt.scatter(scatter["r_brent"], scatter["r_eurusd"], alpha=0.2, s=10)
    line = np.polyfit(scatter["r_brent"], scatter["r_eurusd"], 1)
    xs = np.linspace(scatter["r_brent"].min(), scatter["r_brent"].max(), 200)
    plt.plot(xs, line[0] * xs + line[1], color="darkred", linewidth=2)
    plt.xlabel("Brent daily log return")
    plt.ylabel("EUR/USD daily log return")
    plt.title("Daily co-movement between oil and EUR/USD")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "scatter_daily.png", dpi=180)
    plt.close()

    metrics = {
        "daily_start": daily.index.min().strftime("%Y-%m-%d"),
        "daily_end": daily.index.max().strftime("%Y-%m-%d"),
        "monthly_end": last_complete_month.strftime("%Y-%m-%d"),
        "full_beta_current_oil": float(full_daily_model.params["r_brent"]),
        "full_beta_current_oil_p": float(full_daily_model.pvalues["r_brent"]),
        "pre2022_beta_current_oil": float(daily_regime_table.loc[1, "beta_current_oil"]),
        "pre2022_beta_current_oil_p": float(daily_regime_table.loc[1, "p_current_oil"]),
        "post2022_beta_current_oil": float(daily_regime_table.loc[2, "beta_current_oil"]),
        "post2022_beta_current_oil_p": float(daily_regime_table.loc[2, "p_current_oil"]),
        "cointegration_pre2022_p": float(cointegration_table.loc[0, "p_value"]),
        "cointegration_full_p": float(cointegration_table.loc[2, "p_value"]),
        "ecm_lambda": float(ecm_model.params["ecm_l1"]),
        "ecm_lambda_p": float(ecm_model.pvalues["ecm_l1"]),
        "long_run_oil_coef": float(ecm_long_run.params["l_brent"]),
        "long_run_oil_p": float(ecm_long_run.pvalues["l_brent"]),
    }
    (RESULTS_DIR / "key_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_lines = [
        "Project 1 analytical summary",
        f"Daily sample: {metrics['daily_start']} to {metrics['daily_end']}",
        f"Monthly sample ends at: {metrics['monthly_end']}",
        (
            "Full-sample daily oil coefficient: "
            f"{metrics['full_beta_current_oil']:.4f} "
            f"(p={metrics['full_beta_current_oil_p']:.4f})"
        ),
        (
            "Pre-2022 daily oil coefficient: "
            f"{metrics['pre2022_beta_current_oil']:.4f} "
            f"(p={metrics['pre2022_beta_current_oil_p']:.4f})"
        ),
        (
            "2022+ daily oil coefficient: "
            f"{metrics['post2022_beta_current_oil']:.4f} "
            f"(p={metrics['post2022_beta_current_oil_p']:.4f})"
        ),
        (
            "Cointegration p-value for 1999-2021: "
            f"{metrics['cointegration_pre2022_p']:.4f}"
        ),
        (
            "ECM adjustment coefficient: "
            f"{metrics['ecm_lambda']:.4f} "
            f"(p={metrics['ecm_lambda_p']:.4f})"
        ),
    ]
    save_text_summary(RESULTS_DIR / "analysis_summary.txt", summary_lines)


if __name__ == "__main__":
    main()
