import argparse
import os
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# to call this script: python model_inference.py --snapshotdate "2024-09-01" --modelname "credit_model_2024_09_01.pkl"

def main(snapshotdate, modelname):
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # --- helper functions ---
    def hist_proportions(values, bin_edges):
        counts, _ = np.histogram(values, bins=bin_edges)
        total = counts.sum()
    return (counts / total).tolist() if total > 0 else [0.0] * (len(bin_edges)-1)

    def psi_from_bins(expected_hist, actual_hist, eps=1e-9):
        e = np.clip(np.array(expected_hist, dtype=float), eps, 1.0)
        a = np.clip(np.array(actual_hist,   dtype=float), eps, 1.0)
        e, a = e/e.sum(), a/a.sum()
        return float(np.sum((a - e) * np.log(a / e)))
    
    def series_stats(s):
        s = pd.to_numeric(s, errors="coerce")
        return {
            "n": int(s.notna().sum()),
            "missing_rate": float(1 - s.notna().mean()),
            "mean": float(s.mean()) if s.notna().any() else None,
            "std": float(s.std())  if s.notna().any() else None,
            "min": float(s.min())  if s.notna().any() else None,
            "max": float(s.max())  if s.notna().any() else None,
            "p50": float(s.quantile(0.5)) if s.notna().any() else None
        }

    # --- set up config ---
    config = {}
    config["snapshot_date_str"] = snapshotdate
    config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
    config["model_name"] = modelname
    config["model_bank_directory"] = "model_bank/"
    config["model_artefact_filepath"] = config["model_bank_directory"] + config["model_name"]
    
    pprint.pprint(config)
    

    # --- load model artefact from model bank ---
    # Load the model from the pickle file
    with open(config["model_artefact_filepath"], 'rb') as file:
        model_artefact = pickle.load(file)
    
    print("Model loaded successfully! " + config["model_artefact_filepath"])

    # important features & baselines
    top_feats = model_artefact["important_features"]["names"]
    drift_base = model_artefact["drift_baseline"]
    model = model_artefact["model"]
    transformer_stdscaler = model_artefact["preprocessing_transformers"]["stdscaler"]

    # --- load feature store ---
    folder_path = "datamart/gold/feature_store_clickstream/"
    files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
    feature_clickstream_store_sdf = spark.read.option("header", "true").parquet(*files_list)
    feature_clickstream_sdf = feature_clickstream_store_sdf.filter((col("snapshot_date") == config["snapshot_date"]))
    
    print("extracted feature_clickstream_sdf", feature_clickstream_sdf.count(), config["snapshot_date"])
    
    feature_clickstream_pdf = feature_clickstream_sdf.toPandas()

    folder_path = "datamart/gold/feature_store_financials/"
    files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
    feature_financial_store_sdf = spark.read.option("header", "true").parquet(*files_list)
    feature_financial_sdf = feature_financial_store_sdf.filter((col("snapshot_date") == config["snapshot_date"]))
    
    print("extracted feature_financial_sdf", feature_financial_sdf.count(), config["snapshot_date"])
    
    feature_financial_pdf = feature_financial_sdf.toPandas()

    # prepare data for modeling
    features_pdf = feature_financial_sdf.join(feature_clickstream_sdf, on=["Customer_ID","snapshot_date"], how="left").toPandas()
    features_pdf["Outstanding_Debt_log"] = np.log1p(features_pdf["Outstanding_Debt"])
    features_pdf = features_pdf.drop('Outstanding_Debt', axis=1)

    # --- preprocess data for modeling ---
    # prepare X_inference
    feature_cols = ['avg_fe_1',
     'avg_fe_2',
     'avg_fe_3',
     'avg_fe_4',
     'avg_fe_5',
     'avg_fe_6',
     'avg_fe_7',
     'avg_fe_8',
     'avg_fe_9',
     'avg_fe_10',
     'avg_fe_11',
     'avg_fe_12',
     'avg_fe_13',
     'avg_fe_14',
     'avg_fe_15',
     'avg_fe_16',
     'avg_fe_17',
     'avg_fe_18',
     'avg_fe_19',
     'avg_fe_20',
     'Num_Fin_Pdts',
     'Debt_to_Salary',
     'Loans_per_Credit_Item',
     'Changed_Credit_Limit',
     'Credit_History_Age_Month',
     'Outstanding_Debt_log']
    
    X_inference = features_pdf[feature_cols]
    X_inference = transformer_stdscaler.transform(X_inference) # apply transformer - standard scaler
    print('X_inference', X_inference.shape[0])
    

    # --- model prediction inference ---    
    # predict model
    y_inference = model.predict_proba(X_inference)[:, 1]
    
    # prepare output
    y_inference_pdf = features_pdf[["Customer_ID","snapshot_date"]].copy()
    y_inference_pdf["model_name"] = config["model_name"]
    y_inference_pdf["model_predictions"] = y_inference
    

    # --- save model inference to datamart gold table ---
    # create bronze datalake
    gold_directory = f"datamart/gold/model_predictions/{config["model_name"][:-4]}/"
    print(gold_directory)
    
    if not os.path.exists(gold_directory):
        os.makedirs(gold_directory)
    
    # save gold table - IRL connect to database to write
    partition_name = config["model_name"][:-4] + "_predictions_" + config["snapshot_date_str"].replace('-','_') + '.parquet'
    filepath = gold_directory + partition_name
    spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)

    # --- compute drift metrics ---
    
    mon_rows = []

    # score drift
    score_bins  = drift_base["score_bins"]
    score_ref   = drift_base["score_ref_hist"]
    cur_hist    = hist_proportions(y_inference_pdf["pred_score"].values, score_bins)
    score_psi   = psi_from_bins(score_ref, cur_hist)
    mon_rows.append({
        "model_name": config["model_name"],
        "snapshot_date": config["snapshot_date_str"],
        "metric_scope": "score",
        "feature": "pred_score",
        "psi": score_psi,
        "stats": json.dumps(series_stats(y_inference_pdf["pred_score"]))
    })

    # top feature drift
    for feat in top_feats:
        if feat not in features_pdf.columns:
            continue
        bins     = drift_base["feature_bins"][feat]
        ref_hist = drift_base["feature_ref_hists"][feat]
        cur_hist = hist_proportions(pd.to_numeric(features_pdf[feat], errors="coerce").values, bins)
        psi_val  = psi_from_bins(ref_hist, cur_hist)
        mon_rows.append({
            "model_name": config["model_name"],
            "snapshot_date": config["snapshot_date_str"],
            "metric_scope": "feature",
            "feature": feat,
            "psi": psi_val,
            "stats": json.dumps(series_stats(features_pdf[feat]))
        })

    monitor_pdf = pd.DataFrame(mon_rows)

    # --- save monitoring rows (Gold) ---
    gold_mon_dir = f"datamart/gold/model_monitoring/{config['model_name'][:-4]}/"
    os.makedirs(gold_mon_dir, exist_ok=True)
    mon_file = "monitoring_" + config["snapshot_date_str"].replace('-', '_') + ".parquet"
    spark.createDataFrame(monitor_pdf).write.mode("overwrite").parquet(os.path.join(gold_mon_dir, mon_file))
    print("✅ saved monitoring:", os.path.join(gold_mon_dir, mon_file))
    
    # --- end spark session --- 
    spark.stop()
    
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model_name")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate, args.modelname)
