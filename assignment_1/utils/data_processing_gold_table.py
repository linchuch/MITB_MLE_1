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
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


########################################
def process_features_gold_table(snapshot_date_str, gold_label_store_directory, gold_feature_store_directory, feature_date_str, silver_attribute_directory, silver_financials_directory, spark):

    ########## Get label store
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to label store table
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # select columns from label store
    df = df.select("loan_id", "Customer_ID", "snapshot_date")

    ########## Get features attributes
    # get features from features attributes
    attribute_partition_name = "silver_features_attributes_" + feature_date_str.replace('-','_') + '.parquet'
    attribute_filepath = silver_attribute_directory + attribute_partition_name
    df_feature_attributes = spark.read.parquet(attribute_filepath)
    print('loaded feature attributes from:', attribute_filepath, 'row count:', df_feature_attributes.count())

    df_w_feature_attributes = df.alias("l").join(df_feature_attributes.alias("r"), on="Customer_ID", how="left")
    years_elapsed = F.floor(F.months_between(F.col("l.snapshot_date"), F.col("r.snapshot_date")) / 12)

    df_w_feature_attributes = (df_w_feature_attributes
        .withColumn("years_elapsed", F.greatest(F.lit(0), years_elapsed))  # don’t subtract if attr snapshot is in the future
        .withColumn("Age_as_of",
            (F.col("r.Age").cast("int") + F.col("years_elapsed")).cast("int"))
        )

    # select columns to save
    df_w_feature_attributes = df_w_feature_attributes.select("loan_id", "Customer_ID", "Age_as_of", "Occupation", F.col("l.snapshot_date").alias("snapshot_date"))

    ########## Get features financials
    # get features from features financials
    financials_partition_name = "silver_features_financials_" + feature_date_str.replace('-','_') + '.parquet'
    financials_filepath = silver_financials_directory + financials_partition_name
    df_feature_financials = spark.read.parquet(financials_filepath)
    print('loaded feature financials from:', financials_filepath, 'row count:', df_feature_financials.count())

    df_w_feature_financials = df_w_feature_attributes.alias("l").join(df_feature_financials.alias("r"), on="Customer_ID", how="left")
    df_w_feature_financials = df_w_feature_financials.select("loan_id", "Customer_ID", "Age_as_of", "Occupation", "Annual_Income", "Credit_Mix", F.col("l.snapshot_date").alias("snapshot_date"))
    
    # save gold table - IRL connect to database to write
    partition_name = "gold_feature_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_feature_store_directory + partition_name
    df_w_feature_financials.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    print('next snapshot date')
    
    return df_w_feature_financials