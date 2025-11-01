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
from utils import data_processing_gold_table as G

# to call this script: python bronze_label_store.py --snapshotdate "2023-01-01"

def main(snapshotdate):
    print('\n\n---starting job---\n\n')
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder.appName("features_gold").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # load arguments
    date_str = snapshotdate
    
    # create bronze datalake for features
    click_silver, click_gold: "datamart/silver/features_clickstream/", "datamart/gold/feature_store_clickstream/"
    fin_silver, fin_gold: "datamart/silver/features_financials/", "datamart/gold/feature_store_financials/"

    # 1/ clickstream
    print('\n\n---starting clickstream job---\n\n')
    if not os.path.exists(click_gold):
        os.makedirs(click_gold)

    G.process_features_clickstream_gold_table(date_str, click_silver, click_gold, spark)

    # 2/ financials
    print('\n\n---starting financials job---\n\n')
    if not os.path.exists(fin_gold):
        os.makedirs(fin_gold)

    S.process_silver_table_feature_clickstream(date_str, fin_silver, fin_gold, spark)
    
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
