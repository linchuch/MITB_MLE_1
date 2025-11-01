import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

########################################
def process_labels_gold_table(snapshot_date_str, silver_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_lms_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "loan_start_date", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


########################################
def process_features_clickstream_gold_table(snapshot_date_str, silver_directory, gold_feature_store_clickstream_directory, spark):
    """
    Take past 6 month average of each feature (1-20) for each customer.
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    dfs = []
    for k in range(0, 6): 
        m = snapshot_date- relativedelta(months=k)
        partition_name = "silver_features_clickstream_" + m.strftime('%Y_%m_%d') + '.parquet'
        filepath = silver_directory + partition_name
        try:
            dfk = (spark.read.parquet(filepath)
                   .select("Customer_ID",
                           *[f'fe_{i}' for i in range(1, 21)]))
            print('loaded from:', filepath, 'row count:', dfk.count())
            dfs.append(dfk)
        except Exception:
            pass

    if not dfs:
        return None

    
    union6 = dfs[0]
    for d in dfs[1:]:
        union6 = union6.unionByName(d, allowMissingColumns=True)

    agg_exprs = [F.avg(F.col(f'fe_{i}')).alias(f'avg_fe_{i}') for i in range(1, 21)]
    df_grouped = union6.groupBy("Customer_ID").agg(*agg_exprs).withColumn("snapshot_date", F.lit(snapshot_date_str))
    print('grouped data for', snapshot_date)
    
    # --- 5) Write one file per snapshot (recommend partitioning by snapshot_date) ---
    partition_name = "gold_feature_store_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_feature_store_clickstream_directory + partition_name
    df_grouped.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return df_grouped

########################################
def process_features_financials_gold_table(snapshot_date_str, silver_directory, gold_feature_store_financials_directory, spark):

    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_features_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # extract columns for gold table (includes features engineered)
    df = df.select("Customer_ID", "snapshot_date", 'Num_Fin_Pdts', 'Debt_to_Salary', 
                   'Loans_per_Credit_Item', 'Outstanding_Debt', 'Changed_Credit_Limit','Credit_History_Age_Month')

    # save gold table - IRL connect to database to write
    partition_name = "gold_feature_store_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_feature_store_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df
    
