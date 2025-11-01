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
from utils import data_processing_silver_table as S

# to call this script: python bronze_label_store.py --snapshotdate "2023-01-01"

def main(snapshotdate):
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder.appName("features_silver").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # load arguments
    date_str = snapshotdate
    
    # create bronze datalake for features
    click_bronze, click_silver: "datamart/bronze/features_clickstream/", "datamart/silver/features_clickstream/"
    fin_bronze, fin_silver: "datamart/bronze/features_financials/", "datamart/silver/features_financials/"

    # 1/ clickstream
    print('\n\n---starting clickstream job---\n\n')
    if not os.path.exists(click_silver):
        os.makedirs(click_silver)

    S.process_silver_table_feature_clickstream(date_str, click_bronze, click_silver, spark)

    # 2/ financials
    print('\n\n---starting financials job---\n\n')
    if not os.path.exists(fin_silver):
        os.makedirs(fin_silver)

    S.process_silver_table_feature_clickstream(date_str, fin_bronze, fin_silver, spark)
    
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
