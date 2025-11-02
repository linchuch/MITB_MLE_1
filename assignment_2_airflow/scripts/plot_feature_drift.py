# /opt/airflow/scripts/plot_feature_drift.py
import argparse, os, glob, json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p


def load_all_monitoring(drift_dir: str, upto_dt: datetime) -> pd.DataFrame:
    """Load all monitoring_YYYY_MM_DD.parquet up to (and including) upto_dt."""
    files = sorted(glob.glob(os.path.join(drift_dir, "monitoring_*.parquet")))
    frames = []
    for fp in files:
        try:
            ts = fp.split("monitoring_")[-1].split(".parquet")[0]
            dt = datetime.strptime(ts, "%Y_%m_%d")
            if dt <= upto_dt:
                df = pd.read_parquet(fp)
                # normalize a 'date' column for plotting
                if "snapshot_date" in df.columns:
                    df["date"] = pd.to_datetime(df["snapshot_date"])
                elif "label_month" in df.columns:
                    df["date"] = pd.to_datetime(df["label_month"])
                else:
                    df["date"] = dt
                frames.append(df)
        except Exception as e:
            print(f"Skipping {fp}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def pretty_bin_labels(bins):
    labels = []
    for i in range(len(bins) - 1):
        left = bins[i]
        right = bins[i + 1]
        right_bracket = "]" if i == len(bins) - 2 else ")"
        labels.append(f"[{left:g},{right:g}{right_bracket}")
    return labels


def plot_two_hists(bins, ref_hist, cur_hist, title, out_png=None, ax=None):
    bins = np.asarray(bins, dtype=float)
    ref = np.asarray(ref_hist, dtype=float)
    cur = np.asarray(cur_hist, dtype=float)

    x = np.arange(len(bins) - 1)
    width = 0.4

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    ax.bar(x - width/2, ref, width, label="Baseline", align="center")
    ax.bar(x + width/2, cur, width, label="Current", align="center")

    ax.set_title(title)
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Bins")
    ax.set_xticks(x)
    ax.set_xticklabels(pretty_bin_labels(bins), rotation=45, ha="right")
    ymax = max(0.3, float(np.nanmax([ref.max() if ref.size else 0,
                                     cur.max() if cur.size else 0])) * 1.2)
    ax.set_ylim(0, ymax)
    ax.legend()

    plt.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150)
    return fig


def safe_json_list(v, fallback=None):
    if isinstance(v, (list, tuple, np.ndarray)):
        return list(map(float, v))
    try:
        return list(map(float, json.loads(v)))
    except Exception:
        return fallback if fallback is not None else []


def main(snapshotdate, modelname, datamart_dir, model_bank_dir):
    # we use modelname as-is for paths and titles
    upto_dt = datetime.strptime(snapshotdate, "%Y-%m-%d")

    # paths
    drift_dir = os.path.join(datamart_dir, "gold", "model_monitoring", modelname)

    reports_root = os.path.join(datamart_dir, "reports", "model_drift", modelname)
    ensure_dir(reports_root)
    # time-series (cumulative) saved once per invocation (up to month)
    psi_ts_png = os.path.join(reports_root, f"psi_timeseries_upto_{upto_dt.strftime('%Y_%m_%d')}.png")

    # monthly (current month only) report folder and PDF
    month_dir = ensure_dir(os.path.join(reports_root, upto_dt.strftime("%Y_%m_%d")))
    month_pdf = os.path.join(month_dir, f"drift_report_{upto_dt.strftime('%Y_%m_%d')}.pdf")

    # load all monitoring rows up to date
    mon = load_all_monitoring(drift_dir, upto_dt)
    if mon.empty:
        print("No monitoring data found. Nothing to plot.")
        return

    # ---------- 1) PSI time-series (features) up to current month ----------
    feat_ts = mon[mon["metric_scope"] == "feature"][["date", "feature", "psi"]].copy()
    if not feat_ts.empty:
        feat_ts.sort_values(["feature", "date"], inplace=True)
        plt.figure(figsize=(10, 5))
        for feat, sub in feat_ts.groupby("feature"):
            plt.plot(sub["date"], sub["psi"], marker="o", label=feat)
        for thr in (0.1, 0.25):
            plt.axhline(thr, linestyle="--", color="gray", linewidth=1)
        plt.title(f"Feature PSI over time – {modelname}")
        plt.xlabel("Month")
        plt.ylabel("PSI")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(psi_ts_png, dpi=150)
        plt.close()
        print("Saved:", psi_ts_png)

    # ---------- 2) Current-month histograms (baseline vs current) ----------
    # We only plot the given month (no recompute; read bins/ref_hist/cur_hist from the parquet)
    this_month = mon[mon["date"] == pd.to_datetime(upto_dt)].copy()
    if this_month.empty:
        print(f"No monitoring rows for {upto_dt:%Y-%m}.")
        return

    with PdfPages(month_pdf) as pdf:
        # SCORE (if present)
        score_rows = this_month[this_month["metric_scope"] == "score"]
        if not score_rows.empty:
            r = score_rows.iloc[0]
            bins = safe_json_list(r.get("bins"))
            refh = safe_json_list(r.get("ref_hist"))
            curh = safe_json_list(r.get("cur_hist"))
            if bins and refh and curh:
                title = f"{modelname} – Score histogram (Baseline vs {upto_dt:%Y-%m})"
                fig = plot_two_hists(bins, refh, curh, title=title)
                pdf.savefig(fig); plt.close(fig)
                out_png = os.path.join(month_dir, f"score_hist_{upto_dt.strftime('%Y_%m_%d')}.png")
                fig = plot_two_hists(bins, refh, curh, title=title)
                fig.savefig(out_png, dpi=150); plt.close(fig)
                print("Saved:", out_png)
            else:
                print("Score histogram data missing (bins/ref_hist/cur_hist). Skipping score plot.")

        # FEATURES (top features monitored that month)
        feat_rows = this_month[this_month["metric_scope"] == "feature"].copy()
        for _, r in feat_rows.iterrows():
            feat = r.get("feature", "unknown_feature")
            bins = safe_json_list(r.get("bins"))
            refh = safe_json_list(r.get("ref_hist"))
            curh = safe_json_list(r.get("cur_hist"))
            if not (bins and refh and curh):
                print(f"Skipping {feat}: missing bins/ref_hist/cur_hist.")
                continue

            title = f"{modelname} – {feat} histogram (Baseline vs {upto_dt:%Y-%m})"
            fig = plot_two_hists(bins, refh, curh, title=title)
            pdf.savefig(fig); plt.close(fig)

            out_png = os.path.join(month_dir, f"{feat}_hist_{upto_dt.strftime('%Y_%m_%d')}.png")
            fig = plot_two_hists(bins, refh, curh, title=title)
            fig.savefig(out_png, dpi=150); plt.close(fig)
            print("Saved:", out_png)

    print("Saved monthly drift report PDF:", month_pdf)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshotdate", required=True)                 # YYYY-MM-DD (inference month)
    ap.add_argument("--modelname",    required=True)                 # use as-is (folder names, titles)
    ap.add_argument("--datamart_dir", default="/opt/airflow/datamart")
    ap.add_argument("--model_bank_dir", default="/opt/airflow/model_bank")
    args = ap.parse_args()
    main(args.snapshotdate, args.modelname, args.datamart_dir, args.model_bank_dir)
