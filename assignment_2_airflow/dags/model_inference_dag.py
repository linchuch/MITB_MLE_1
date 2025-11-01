from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'model_inference_dag',
    default_args=default_args,
    description='data pipeline run once a month',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2024, 7, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
) as dag:

    # data pipeline

    # --- label store ---

    start = DummyOperator(task_id="start")

    bronze = BashOperator(
        task_id="features_bronze",
        bash_command="cd /opt/airflow/scripts && python3 features_bronze.py --snapshotdate '{{ ds }}'",
    )
    silver = BashOperator(
        task_id="features_silver",
        bash_command="cd /opt/airflow/scripts && python3 features_silver.py --snapshotdate '{{ ds }}'",
    )
    gold = BashOperator(
        task_id="features_gold",
        bash_command="cd /opt/airflow/scripts && python3 features_gold.py --snapshotdate '{{ ds }}'",
    )
    infer = BashOperator(
        task_id="model_inference",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 model_inference.py --snapshotdate '{{ ds }}' --modelname 'credit_model_2024_06.pkl'"
        ),
    )
    end = DummyOperator(task_id="end")

    start >> bronze >> silver >> gold >> infer >> end