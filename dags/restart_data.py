from airflow import DAG
from airflow.decorators import dag, task
from datetime import datetime
from scripts.utils import restart_data_generation

API_URL = "http://10.43.101.149:80"
GROUP_NUMBER = 6

@dag(start_date=datetime(2024, 5, 5), catchup=False, schedule_interval='@once')
def restar_data_generation_dag():
    
    @task
    def restart_data_generation_task(api_url: str, group_number: int):
        restart_data_generation(api_url, group_number)

    restart_data_generation_task(API_URL, GROUP_NUMBER)

dag_instance = restar_data_generation_dag()