#!/usr/bin/env python
# coding: utf-8

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import importlib, utils

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table


# ## Set up Spark session

# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")


# ## Set up Config

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    # Convert the date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    # List to store the first of month dates
    first_of_month_dates = []

    # Start from the first of the month of the start_date
    current_date = datetime(start_date.year, start_date.month, 1)

    while current_date <= end_date:
        # Append the date in yyyy-mm-dd format
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))

        # Move to the first of the next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)

    return first_of_month_dates


# ## Build Bronze Table

BRONZE_DATASETS = [
    {
        "name": "feature_clickstream",
        "csv_file_path": "data/feature_clickstream.csv",
        "bronze_partition": "bronze_features_clickstream_",
        "bronze_path": "datamart/bronze/features_clickstream/",
        "update_type": "append"
    },
    {
        "name": "features_attributes",
        "csv_file_path": "data/features_attributes.csv",
        "bronze_partition": "bronze_features_attribute_",
        "bronze_path": "datamart/bronze/features_attributes/",
        "update_type": "overwrite"
    },
    {
        # Overwrite-at-source; we stamp each ingest with "today"
        "name": "features_financials",
        "csv_file_path": "data/features_financials.csv",
        "bronze_partition": "bronze_features_financials_",
        "bronze_path": "datamart/bronze/features_financials/",
        "update_type": "overwrite"
    },
    {
        "name": "lms_loan_daily",
        "csv_file_path": "data/lms_loan_daily.csv",
        "bronze_partition": "bronze_lms_loan_daily_",
        "bronze_path": "datamart/bronze/lms_loan_daily/",
        "update_type": "append"
    },
]


dates_str_lst = {}

for file in BRONZE_DATASETS:

    table_name = file["name"]
    csv_file_path = file["csv_file_path"]
    update_type = file["update_type"]

    if update_type=="append":
        df = pd.read_csv(file["csv_file_path"])
        min_date = df["snapshot_date"].min()
        max_date = df["snapshot_date"].max()
        dates = generate_first_of_month_dates(min_date, max_date)

    else:
        min_date = pd.to_datetime(date.today()).strftime("%Y-%m-%d")
        max_date = min_date
        dates = generate_first_of_month_dates(min_date, max_date)

    dates_str_lst[table_name] = dates


for file in BRONZE_DATASETS:

    table_name = file["name"]
    csv_file_path = file["csv_file_path"]
    bronze_directory = file["bronze_path"]
    bronze_partition = file["bronze_partition"]
    update_type = file["update_type"]
    df = pd.read_csv(csv_file_path)

    if not os.path.exists(bronze_directory):
        os.makedirs(bronze_directory)

    if update_type=="append":
        # run bronze backfill
        for date_str in dates_str_lst[table_name]:
            utils.data_processing_bronze_table.process_bronze_table_append(date_str, csv_file_path, bronze_partition, bronze_directory, spark)

    else:
        # run bronze backfill
        for date_str in dates_str_lst[table_name]:
            utils.data_processing_bronze_table.process_bronze_table_overwrite(date_str, csv_file_path, bronze_partition, bronze_directory, spark)


# ## Build Silver Table

SILVER_DATASETS = [
    {
        "name": "feature_clickstream",
        "bronze_path": "datamart/bronze/features_clickstream/",
        "silver_path": "datamart/silver/features_clickstream/"
    },
    {
        "name": "features_attributes",
        "bronze_path": "datamart/bronze/features_attributes/",
        "silver_path": "datamart/silver/features_attributes/"
    },
    {
        "name": "features_financials",
        "bronze_path": "datamart/bronze/features_financials/",
        "silver_path": "datamart/silver/features_financials/"
    },
    {
        "name": "lms_loan_daily",
        "bronze_path": "datamart/bronze/lms_loan_daily/",
        "silver_path": "datamart/silver/lms_loan_daily/"
    },
]


# update silver processing file:
silver_processing = importlib.import_module("utils.data_processing_silver_table")

for file in SILVER_DATASETS:

    table_name = file["name"]
    bronze_directory = file["bronze_path"]
    silver_directory = file["silver_path"]
    func = silver_processing.REGISTRY.get(table_name)

    if not os.path.exists(silver_directory):
        os.makedirs(silver_directory)

    # run silver backfill
    for date_str in dates_str_lst[table_name]:
        func(date_str, bronze_directory, silver_directory, spark)


# ## Build Gold Table - Label Store

# create gold table datalake
silver_directory = "datamart/silver/lms_loan_daily/"
gold_label_store_directory = "datamart/gold/label_store/"

if not os.path.exists(gold_label_store_directory):
    os.makedirs(gold_label_store_directory)


for date_str in dates_str_lst["lms_loan_daily"]:
    utils.data_processing_gold_table.process_labels_gold_table(date_str, silver_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)


# ## Build Gold Table - Feature Store

# create gold table datalake
gold_label_store_directory = "datamart/gold/label_store/"
gold_feature_store_directory = "datamart/gold/feature_store/"
silver_attribute_directory = "datamart/silver/features_attributes/"
silver_financials_directory = "datamart/silver/features_financials/"
feature_date_str = dates_str_lst["features_attributes"][0]

if not os.path.exists(gold_feature_store_directory):
    os.makedirs(gold_feature_store_directory)


for date_str in dates_str_lst["lms_loan_daily"]:
    utils.data_processing_gold_table.process_features_gold_table(date_str, gold_label_store_directory, gold_feature_store_directory, feature_date_str, silver_attribute_directory, silver_financials_directory, spark)

