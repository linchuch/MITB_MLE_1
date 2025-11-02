# /opt/airflow/scripts/plot_model_performance.py
import argparse, os, glob
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p

def load_perf_df(perf_dir: str) -> pd.DataFrame:
    """Load all perf_YYYY_MM_DD.parquet files; normalize 'date' from label_month."""
    files = sorted(glob.glob(os.path.join(perf_dir, "perf_*.parquet")))
    frames = []
    for fp in files:
        try:
            ts = fp.split("perf_")[-1].split(".parquet")[0]  # YYYY_MM_DD
            dt = datetime.strptime(ts, "%Y_%m_%d")
            df = pd.read_parquet(fp)
            # model_evaluation.py writes 'label_month' as YYYY-MM-DD string
            if "label_month" in df.columns:
                df["date"] = pd.to_datetime(df["label_month"])
            else:
                df["date"] = dt
            frames.append(df)
        except Exception as e:
            print(f"Skip {fp}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def plot_metric(ax, df, metric: str, title_suffix: str):
    if metric not in df.columns:
        ax.axis("off"); ax.set_title(f"{metric} (not available)")
        return
    sub = df[["date", metric]].dropna().sort_values("date")
    if sub.empty:
        ax.axis("off"); ax.set_title(f"{metric} (no data)")
        return
    ax.plot(sub["date"], sub[metric], marker="o")
    ax.set_title(f"{metric} over time – {title_suffix}")
    ax.set_xlabel("Label month"); ax.set_ylabel(metric)
    if metric.lower() in ("auc","gini","precision","recall"):
        lo, hi = sub[metric].min(), sub[metric].max()
        pad = max(0.02, (hi - lo) * 0.15)
        ax.set_ylim(max(0, lo - pad), min(1.0, hi + pad))
    ax.grid(True, linestyle="--", linewidth=0.4)

def main(modelname, datamart_dir):
    # match evaluation script’s folder convention (strip extension)
    model_base = os.path.splitext(modelname)[0]

    perf_dir = os.path.join(datamart_dir, "gold", "model_performance", model_base)
    out_root = ensure_dir(os.path.join(datamart_dir, "reports", "model_performance", model_base))

    df = load_perf_df(perf_dir)
    if df.empty:
        print(f"No performance files found in {perf_dir}")
        return

    # keep latest per date if duplicates
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    # save a tidy CSV summary
    summary_csv = os.path.join(out_root, "performance_summary.csv")
    cols_keep = ["date","pred_month","n","auc","gini","precision","recall","model_name"]
    cols_keep = [c for c in cols_keep if c in df.columns]
    df[cols_keep].sort_values("date").to_csv(summary_csv, index=False)
    print("Saved:", summary_csv)

    # single-page figure with 4 core metrics
    metrics = [m for m in ["auc","gini","precision","recall"] if m in df.columns]
    n = len(metrics)
    cols = 2 if n >= 2 else n
    rows = int(np.ceil(n / max(cols,1)))

    fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*3.8))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    axes = axes.reshape(rows, cols) if n > 0 else np.array([[]])

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n:
                axes[r, c].axis("off"); continue
            plot_metric(axes[r, c], df, metrics[idx], model_base)
            idx += 1

    plt.tight_layout()
    # name outputs using last available label month
    last_dt = df["date"].max()
    onepage_pdf = os.path.join(out_root, f"performance_upto_{last_dt.strftime('%Y_%m_%d')}.pdf")
    onepage_png = onepage_pdf.replace(".pdf", ".png")
    with PdfPages(onepage_pdf) as pdf:
        pdf.savefig(fig, dpi=150)
    plt.savefig(onepage_png, dpi=150)
    plt.close(fig)
    print("Saved:", onepage_pdf)
    print("Saved:", onepage_png)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot model performance over time.")
    ap.add_argument("--modelname", required=True)                 # e.g. credit_model_2024_06_01 (or .pkl)
    ap.add_argument("--datamart_dir", default="/opt/airflow/datamart")
    args = ap.parse_args()
    main(args.modelname, args.datamart_dir)
