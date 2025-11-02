# /opt/airflow/model_inference.py

import argparse
import os
import glob
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
import xgboost as xgb

# to call this script:
# python model_inference.py --snapshotdate "2024-09-01" --modelname "credit_model_2024_06_01"

def hist_proportions(values, bin_edges):
    counts, _ = np.histogram(values, bins=bin_edges)
    total = counts.sum()
    return (counts / total).tolist() if total > 0 else [0.0] * (len(bin_edges) - 1)

def psi_from_bins(expected_hist, actual_hist, eps=1e-9):
    """Population Stability Index between two distributions."""
    e = np.clip(np.array(expected_hist, dtype=float), eps, 1.0)
    a = np.clip(np.array(actual_hist,   dtype=float), eps, 1.0)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))

def series_stats(s):
    """Compute basic descriptive stats for a pandas Series."""
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return {"n": 0, "missing_rate": 1.0, "mean": None, "std": None,
                "min": None, "max": None, "p50": None}
    return {
        "n": int(s.notna().sum()),
        "missing_rate": float(1 - s.notna().mean()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)),
        "min": float(s.min()),
        "max": float(s.max()),
        "p50": float(s.quantile(0.5)),
    }

def main(snapshotdate, modelname):
    print("\n\n---starting job---\n\n")

    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder.appName("dev").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # --- set up config ---
    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
    config["model_name"] = modelname  # may be with or without extension

    # paths (allow override via env)
    config["model_bank_directory"] = "/opt/airflow/model_bank/"
    config["datamart_dir"] = "/opt/airflow/datamart/"

    # normalize to base model version (strip extension if provided)
    model_base = os.path.splitext(config["model_name"])[0]

    booster_path  = os.path.join(config["model_bank_directory"], f"{model_base}_booster.json")
    scaler_path   = os.path.join(config["model_bank_directory"], f"{model_base}_scaler.json")
    metadata_path = os.path.join(config["model_bank_directory"], f"{model_base}_metadata.joblib")

    pprint.pprint({
        **config,
        "booster_path": booster_path,
        "scaler_path": scaler_path,
        "metadata_path": metadata_path,
    })

    # --- load model/scaler/metadata ---
    assert os.path.exists(booster_path),  f"Missing booster JSON: {booster_path}"
    assert os.path.exists(scaler_path),   f"Missing scaler JSON: {scaler_path}"
    assert os.path.exists(metadata_path), f"Missing metadata joblib: {metadata_path}"

    booster = xgb.Booster()
    booster.load_model(booster_path)

    with open(scaler_path, "r") as f:
        scaler_info = json.load(f)
    feature_order = scaler_info["feature_order"]
    mean = np.array(scaler_info["mean"], dtype=float)
    scale = np.array(scaler_info["scale"], dtype=float)
    scale = np.where(scale == 0.0, 1.0, scale)  # avoid divide-by-zero

    metadata = joblib.load(metadata_path)
    drift_base = metadata.get("drift_baseline", {})
    top_feats = metadata.get("important_features", {}).get("names", [])

    # --- load feature store for this snapshot ---
    # clickstream
    folder_path = os.path.join(config["datamart_dir"], "gold/feature_store_clickstream/")
    files_list = [folder_path + os.path.basename(f) for f in glob.glob(os.path.join(folder_path, "*"))]
    feature_clickstream_store_sdf = spark.read.option("header", "true").parquet(*files_list)
    feature_clickstream_sdf = feature_clickstream_store_sdf.filter(col("snapshot_date") == config["snapshot_date"])
    print("extracted feature_clickstream_sdf", feature_clickstream_sdf.count(), config["snapshot_date"])

    # financials
    folder_path = os.path.join(config["datamart_dir"], "gold/feature_store_financials/")
    files_list = [folder_path + os.path.basename(f) for f in glob.glob(os.path.join(folder_path, "*"))]
    feature_financial_store_sdf = spark.read.option("header", "true").parquet(*files_list)
    feature_financial_sdf = feature_financial_store_sdf.filter(col("snapshot_date") == config["snapshot_date"])
    print("extracted feature_financial_sdf", feature_financial_sdf.count(), config["snapshot_date"])

    # join & prepare
    features_pdf = feature_financial_sdf.join(
        feature_clickstream_sdf, on=["Customer_ID", "snapshot_date"], how="left"
    ).toPandas()

    # numeric transform
    features_pdf["Outstanding_Debt_log"] = np.log1p(pd.to_numeric(features_pdf["Outstanding_Debt"], errors="coerce"))
    if "Outstanding_Debt" in features_pdf.columns:
        features_pdf = features_pdf.drop(columns=["Outstanding_Debt"])

    # ensure all required columns exist; if missing, create as NaN (then become NaN std)
    missing_cols = [c for c in feature_order if c not in features_pdf.columns]
    for c in missing_cols:
        features_pdf[c] = np.nan

    # align order and standardize with saved stats
    X_inf = features_pdf[feature_order].to_numpy(dtype=float)
    X_inf_std = (X_inf - mean) / scale

    # --- predict with Booster ---
    dmat = xgb.DMatrix(X_inf_std)
    y_pred = booster.predict(dmat)  # probability for class=1 (binary:logistic)
    print("inference rows:", len(y_pred))

    # --- save predictions (Gold) ---
    y_inference_pdf = features_pdf[["Customer_ID", "snapshot_date"]].copy()
    y_inference_pdf["model_name"] = model_base
    y_inference_pdf["pred_score"] = y_pred

    gold_pred_dir = os.path.join(config["datamart_dir"], "gold", "model_predictions", model_base)
    os.makedirs(gold_pred_dir, exist_ok=True)
    pred_file = f"{model_base}_predictions_{config['snapshot_date_str'].replace('-', '_')}.parquet"
    spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(os.path.join(gold_pred_dir, pred_file))
    print("✅ saved predictions:", os.path.join(gold_pred_dir, pred_file))

    # --- compute & save drift monitoring (score + top features) ---
    mon_rows = []

    # score drift
    score_bins = drift_base.get("score_bins")
    score_ref = drift_base.get("score_ref_hist")
    if score_bins is not None and score_ref is not None:
        cur_hist = hist_proportions(y_inference_pdf["pred_score"].values, score_bins)
        score_psi = psi_from_bins(score_ref, cur_hist)
        mon_rows.append({
            "model_name": model_base,
            "snapshot_date": config["snapshot_date_str"],
            "metric_scope": "score",
            "feature": "pred_score",
            "psi": score_psi,
            "stats": json.dumps(series_stats(y_inference_pdf["pred_score"])),
        })

    # top feature drift
    # feature_bins, feature_ref_hists, feature_stats = {}, {}, {}
    # X_train_df = X_train if hasattr(X_train, "columns") else pd.DataFrame(X_train_processed, columns=feature_names)
    # for feat, _ in top_features:
    #     s = pd.to_numeric(X_train_df[feat], errors="coerce")
    #     bins = make_bins_from_quantiles(s)
    #     feature_bins[feat] = bins.tolist()
    #     feature_ref_hists[feat] = hist_proportions(s.values, bins)
    #     feature_stats[feat] = series_stats(s)


    f_bins = drift_base.get("feature_bins", {})
    f_refh = drift_base.get("feature_ref_hists", {})

    for feat in top_feats:
        if feat not in features_pdf.columns:
            continue
        if feat not in f_bins or feat not in f_refh:
            continue
        bins = f_bins[feat]
        ref_hist = f_refh[feat]

        s = pd.to_numeric(features_pdf[feat], errors="coerce").values
        cur_hist = hist_proportions(s, bins)
        psi_val = psi_from_bins(ref_hist, cur_hist)

        mon_rows.append({
            "model_name": model_base,
            "snapshot_date": config["snapshot_date_str"],
            "metric_scope": "feature",
            "feature": feat,
            "psi": psi_val,
            "bins": json.dumps([float(b) for b in bins]),          # NEW (optional but handy)
            "ref_hist": json.dumps([float(x) for x in ref_hist]),  # NEW (optional)
            "cur_hist": json.dumps([float(x) for x in cur_hist]),  # NEW ← store this!
            "stats": json.dumps(series_stats(features_pdf[feat])),
        })

    monitor_pdf = pd.DataFrame(mon_rows)
    gold_mon_dir = os.path.join(config["datamart_dir"], "gold", "model_monitoring", model_base)
    os.makedirs(gold_mon_dir, exist_ok=True)
    mon_file = f"monitoring_{config['snapshot_date_str'].replace('-', '_')}.parquet"
    spark.createDataFrame(monitor_pdf).write.mode("overwrite").parquet(os.path.join(gold_mon_dir, mon_file))
    print("✅ saved monitoring:", os.path.join(gold_mon_dir, mon_file))

    spark.stop()
    print("\n\n---completed job---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model base or filename")
    args = parser.parse_args()
    main(args.snapshotdate, args.modelname)
