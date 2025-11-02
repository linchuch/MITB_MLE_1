# /opt/airflow/scripts/model_evaluation.py
import argparse, os, json, joblib, numpy as np, pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
)

def main(label_month, modelname, model_bank_dir, datamart_dir):
    spark = pyspark.sql.SparkSession.builder.appName("model_evaluations").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    label_dt = datetime.strptime(label_month, "%Y-%m-%d")
    pred_dt  = label_dt - relativedelta(months=6)

    model_base = os.path.splitext(modelname)[0]
    meta_path  = os.path.join(model_bank_dir, f"{model_base}_metadata.joblib")
    metadata   = joblib.load(meta_path)

    # 1) labels (gold) at label_month
    labels_dir = os.path.join(datamart_dir, "gold", "label_store")
    lab = spark.read.parquet(os.path.join(labels_dir, "*"))
    lab = lab.filter(col("snapshot_date") == label_dt).select("Customer_ID","label")

    # 2) predictions (gold) from pred_month
    pred_dir = os.path.join(datamart_dir, "gold", "model_predictions", model_base)
    pred = spark.read.parquet(os.path.join(pred_dir, "*")) \
           .filter(col("snapshot_date") == pred_dt) \
           .select("Customer_ID","pred_score")

    # 3) join and evaluate
    eval_df = lab.join(pred, on="Customer_ID", how="inner").toPandas()
    eval_df["pred_label"] = (eval_df["pred_score"] > 0.5).astype(int)

    y_true = eval_df["label"]
    y_score = eval_df["pred_score"]
    y_pred = eval_df["pred_label"]

    auc = roc_auc_score(y_true, y_score)
    gini = 2 * auc - 1
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    perf_pdf = pd.DataFrame([{
        "model_name": model_base,
        "label_month": label_month,
        "pred_month": pred_dt.strftime("%Y-%m-%d"),
        "n": int(len(eval_df)),
        "auc": float(auc),
        "gini": float(gini),
        "precision": float(precision),
        "recall": float(recall),        
    }])


    # 6) write gold outputs
    perf_out = os.path.join(datamart_dir, "gold", "model_performance", model_base)
    os.makedirs(perf_out, exist_ok=True)

    spark.createDataFrame(perf_pdf).write.mode("overwrite").parquet(
        os.path.join(perf_out, f"perf_{label_dt.strftime('%Y_%m_%d')}.parquet"))

    spark.stop()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_month", required=True)
    ap.add_argument("--modelname", required=True)
    ap.add_argument("--model_bank_dir", default="/opt/airflow/model_bank")
    ap.add_argument("--datamart_dir", default="/opt/airflow/datamart")
    a = ap.parse_args()
    main(a.label_month, a.modelname, a.model_bank_dir, a.datamart_dir)
