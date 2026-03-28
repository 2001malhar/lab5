# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.pipeline import load_data, data_preprocessing, build_save_model, load_model_predict

# NOTE:
# In Airflow 3.x, enabling XCom pickling should be done via environment variable:
# export AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True

# Define default arguments for the DAG
default_args = {
    'owner': 'Malhar Parikshak',
    'start_date': datetime(2026, 3, 1),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

# Create a DAG instance
with DAG(
    'Wholesale_Clustering_Pipeline',
    default_args=default_args,
    description='DBSCAN clustering on wholesale customer spending data',
    catchup=False,
) as dag:

    # Task 1: Load the wholesale customer dataset
    load_data_task = PythonOperator(
        task_id='load_data_task',
        python_callable=load_data,
    )

    # Task 2: Preprocess — select features, scale with StandardScaler
    data_preprocessing_task = PythonOperator(
        task_id='data_preprocessing_task',
        python_callable=data_preprocessing,
        op_args=[load_data_task.output],
    )

    # Task 3: Run DBSCAN across eps values, save best model
    build_save_model_task = PythonOperator(
        task_id='build_save_model_task',
        python_callable=build_save_model,
        op_args=[data_preprocessing_task.output, "dbscan_model.pkl"],
    )

    # Task 4: Load saved model, report best eps, predict on test data
    load_model_task = PythonOperator(
        task_id='load_model_task',
        python_callable=load_model_predict,
        op_args=["dbscan_model.pkl", build_save_model_task.output],
    )

    # Set task dependencies
    load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task

if __name__ == "__main__":
    dag.test()