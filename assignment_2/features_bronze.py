import argparse
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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table

# to call this script: python bronze_label_store.py --snapshotdate "2023-01-01"

def main(snapshotdate):
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder.appName("features_bronze").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # load arguments
    date_str = snapshotdate
    
    # create bronze datalake for features
    click_csv, fin_csv: "data/feature_clickstream.csv", "data/features_financials.csv"
    click_bronze_partition, fin_bronze_partition: "bronze_features_clickstream_", "bronze_features_financials_"
    click_bronze_dir, fin_bronze_dir: "datamart/bronze/features_clickstream/", "datamart/bronze/features_financials/"

    # 1/ clickstream
    print('\n\n---starting clickstream job---\n\n')
    if not os.path.exists(click_bronze_dir):
        os.makedirs(click_bronze_dir)

    utils.data_processing_bronze_table.process_bronze_table(date_str, click_csv, click_bronze_partition, click_bronze_dir, spark)

    # 2/ financials
    print('\n\n---starting financials job---\n\n')
    if not os.path.exists(fin_bronze_dir):
        os.makedirs(fin_bronze_dir)

    utils.data_processing_bronze_table.process_bronze_table(date_str, fin_csv, fin_bronze_partition, fin_bronze_dir, spark)
    
    # end spark session
    spark.stop()
    
    print('\n\n---completed job---\n\n')

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate)
