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


REGISTRY = {}

def register(table_name: str):
    def deco(f):
        REGISTRY[table_name] = f
        return f
    return deco

######################################## 
@register("feature_clickstream")
def process_silver_table_feature_clickstream(snapshot_date_str, bronze_directory, silver_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_features_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "fe_1": IntegerType(),
        "fe_2": IntegerType(),
        "fe_3": IntegerType(),
        "fe_4": IntegerType(),
        "fe_5": IntegerType(),
        "fe_6": IntegerType(),
        "fe_7": IntegerType(),
        "fe_8": IntegerType(),
        "fe_9": IntegerType(),
        "fe_10": IntegerType(),
        "fe_11": IntegerType(),
        "fe_12": IntegerType(),
        "fe_13": IntegerType(),
        "fe_14": IntegerType(),
        "fe_15": IntegerType(),
        "fe_16": IntegerType(),
        "fe_17": IntegerType(),
        "fe_18": IntegerType(),
        "fe_19": IntegerType(),
        "fe_20": IntegerType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    feature_cols = [f"fe_{i}" for i in range(1, 21)]
    for feature in feature_cols:
        df = df.withColumn(feature, F.when(F.col(feature) < 0, 0).otherwise(F.col(feature)))
        
    # save silver table - IRL connect to database to write
    partition_name = "silver_features_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df
    

########################################
@register("features_attributes")
def process_silver_table_features_attributes(snapshot_date_str, bronze_directory, silver_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_features_attribute_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: fix data errors
    df = df.withColumn("Name", F.regexp_replace(F.col("Name"), r'"[^"]*$', '')).withColumn("Name", F.regexp_replace(F.col("Name"), '"', '')).withColumn("Name", F.trim(F.col("Name")))
    df = df.withColumn("Age", F.regexp_replace(F.col("Age"), "_", "")).withColumn("Age", F.col("Age").cast("int")).withColumn("Age", F.least(F.lit(100), F.greatest(F.lit(0), F.col("Age"))))
    df = df.replace("_______", "Unknown", subset=["Occupation"])
    df = df.replace("#F%$D@*&8","Unknown", subset=["SSN"])
    
    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # save silver table - IRL connect to database to write
    partition_name = "silver_features_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


########################################
@register("features_financials")
def process_silver_table_features_financials(snapshot_date_str, bronze_directory, silver_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_features_financials_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: fix data errors
    cols_int = ['Num_Bank_Accounts','Num_Credit_Card','Interest_Rate','Num_of_Loan','Delay_from_due_date','Num_of_Delayed_Payment','Num_Credit_Inquiries']

    for x in cols_int:
        df = df.withColumn(x, F.regexp_replace(F.col(x), "_", "")).withColumn(x, F.col(x).cast("int"))
        df = df.withColumn(x, F.when(F.col(x) < 0, None).otherwise(F.col(x)))

    cols_float = ['Annual_Income','Monthly_Inhand_Salary','Changed_Credit_Limit','Outstanding_Debt','Credit_Utilization_Ratio','Total_EMI_per_month','Amount_invested_monthly','Monthly_Balance']

    for x in cols_float:
        df = df.withColumn(x, F.regexp_replace(F.col(x), "_", "")).withColumn(x, F.col(x).cast("double"))
        df = df.withColumn(x, F.when(F.col(x) < 0, None).otherwise(F.col(x)))

    # cap outliers
    outlier_caps = {
        'Num_Bank_Accounts': 7,
        'Num_Credit_Card': 7,
        'Interest_Rate': 20,
        'Num_of_Loan': 9,
        'Num_of_Delayed_Payment': 47,
        'Num_Credit_Inquiries': 10
    }
    for x, cap in outlier_caps.items():
        df = df.withColumn(x, F.when(F.col(x) > cap, cap).otherwise(F.col(x)))

    # fix gibberish
    df = df.withColumn("Payment_Behaviour", F.regexp_replace(F.col("Payment_Behaviour"), "!@9#%8", "Unknown"))
    df = df.withColumn("Credit_Mix", F.regexp_replace(F.col("Credit_Mix"), "_", "Unknown"))
    df = df.withColumn("Payment_of_Min_Amount", F.regexp_replace(F.col("Payment_of_Min_Amount"), "NM", "Unknown"))
    df = df.withColumn("Type_of_Loan",
            F.array_join(                                              # turn array back to comma-separated string
                F.array_distinct(                                     # remove duplicates (case-insensitive after normalization below)
                    F.filter(                                         # drop empty/nulls
                        F.transform(
                            F.split(F.regexp_replace("Type_of_Loan", r"\s*,\s*", ","), ","),  # split on commas (ignore spaces)
                            lambda x: F.initcap(F.trim(x))                          # trim + canonicalize casing (e.g., 'mortgage loan' → 'Mortgage Loan')
                        ),
                        lambda x: (x.isNotNull()) & (x != "") & (x != F.lit("Not Specified"))
                    )
                ),
                ", "
            )
        ).withColumn("Type_of_Loan", F.regexp_replace("Type_of_Loan", r'(?i)\s*,?\s*and\s*,?\s*', ', '))

    # feature engineering
    df = df.withColumn("Num_Fin_Pdts", F.col("Num_Bank_Accounts") + F.col("Num_Credit_Card") + F.col("Num_of_Loan"))
    df = df.withColumn("Debt_to_Salary", F.col("Outstanding_Debt") / (F.col("Monthly_Inhand_Salary") + F.lit(1)))
    df = df.withColumn("Loans_per_Credit_Item", F.col("Num_of_Loan") / (F.col("Num_Bank_Accounts") + F.col("Num_Credit_Card") + F.lit(1)))
    df = df.withColumn("credit_history_age_year",
                        F.regexp_extract(col('credit_history_age'), r'(\d+)\s+Year', 1))
    df = df.withColumn("credit_history_age_year", F.col("credit_history_age_year").cast("int"))
    df = df.withColumn("credit_history_age_month",
                        F.regexp_extract(F.col('credit_history_age'), r'(\d+)\s+Month', 1))
    df = df.withColumn("credit_history_age_month", F.col("credit_history_age_month").cast("int"))
    df = df.withColumn("credit_history_age_month", F.col("credit_history_age_year") * 12 + F.col("credit_history_age_month"))
    df = df.drop("credit_history_age_year")
    df = df.withColumnRenamed("credit_history_age_month", "Credit_History_Age_Month")

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": FloatType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Num_of_Loan": IntegerType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age": StringType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": FloatType(),
        "Num_Fin_Pdts": IntegerType(),
        "Debt_to_Salary": FloatType(),
        "Loans_per_Credit_Item": FloatType(),
        "Credit_History_Age_Month": IntegerType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # save silver table - IRL connect to database to write
    partition_name = "silver_features_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df


########################################
@register("lms_loan_daily")
def process_silver_table_lms_loan_daily(snapshot_date_str, bronze_directory, silver_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_lms_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_lms_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df