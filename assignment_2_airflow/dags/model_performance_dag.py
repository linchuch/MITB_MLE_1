# dags/model_monitoring_dag.py
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="model_monitoring_dag",
    description="Monthly: build label store and compute performance + drift",
    schedule_interval="0 0 1 * *",   # run on the 1st of every month
    start_date=datetime(2025, 1, 1), # first label month that matures
    end_date=datetime(2025, 6, 1),   # last label month
    catchup=True,
    default_args=default_args,
) as dag:
    start = DummyOperator(task_id="start")

    # 1) LABEL STORE (bronze -> silver -> gold)
    labels_bronze = BashOperator(
        task_id="labels_bronze",
        bash_command="cd /opt/airflow/scripts && python3 labels_bronze.py --snapshotdate '{{ ds }}'",
    )

    labels_silver = BashOperator(
        task_id="labels_silver",
        bash_command="cd /opt/airflow/scripts && python3 labels_silver.py --snapshotdate '{{ ds }}'",
    )

    labels_gold = BashOperator(
        task_id="labels_gold",
        bash_command="cd /opt/airflow/scripts && python3 labels_gold.py --snapshotdate '{{ ds }}'",
    )
    # Only pass label month; the script will compute pred_month = label_month - 6 months
    model_evaluation = BashOperator(
        task_id="model_evaluation",
        bash_command=(
            "cd /opt/airflow/scripts && "
            "python3 model_evaluation.py "
            "--label_month '{{ ds }}' "
            "--modelname 'credit_model_2024_06_01' "
            "--model_bank_dir '/opt/airflow/model_bank' "
            "--datamart_dir '/opt/airflow/datamart'"
        ),
    )
    plot_model_performance = BashOperator(
        task_id="plot_model_performance",
        bash_command=(
        "cd /opt/airflow/scripts && "
        "python3 plot_model_performance.py "
        "--modelname 'credit_model_2024_06_01' "
        "--datamart_dir '/opt/airflow/datamart'"
        ),
    )

    end = DummyOperator(task_id="end")

    start >> labels_bronze >> labels_silver >> labels_gold >> model_evaluation >> plot_model_performance >> end
