from airflow import DAG
from airflow.decorators import dag, task
from datetime import datetime
from sqlalchemy import create_engine, text
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from airflow.operators.python import BranchPythonOperator
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, chi2_contingency
import numpy as np

API_URL = "http://10.43.101.149:80"
GROUP_NUMBER = 6
RAW_DATABASE_URI = 'postgresql+psycopg2://rawusr:rawpass@rawdb:5432/rawdb'
CLEAN_DATABASE_URI = 'postgresql+psycopg2://cleanusr:cleanpass@cleandb:5432/cleandb'
REPORTS_DIR = '/opt/airflow/logs/reports'

def create_table_if_not_exists():
    engine = create_engine(RAW_DATABASE_URI)
    create_table_query = """
    CREATE TABLE IF NOT EXISTS raw_data (
        brokered_by VARCHAR(255),
        status VARCHAR(255),
        price NUMERIC,
        bed INTEGER,
        bath INTEGER,
        acre_lot NUMERIC,
        street VARCHAR(255),
        city VARCHAR(255),
        state VARCHAR(255),
        zip_code VARCHAR(20),
        house_size NUMERIC,
        prev_sold_date DATE,
        batch_number INTEGER
    );
    """
    with engine.connect() as connection:
        connection.execute(text(create_table_query))

@dag(
    default_args={
        'owner': 'airflow',
        'start_date': datetime(2024, 1, 1),
    },
    description='Fetch data from API and store in RAW DB',
    catchup=False,
    tags=['ml_pipeline'],
    schedule_interval='*/20 * * * *'  # This sets the DAG to run every 10 minutes
)
def ml_pipeline():
    
    @task
    def fetch_and_store_data():
        from scripts.utils import get_data_from_api
        create_table_if_not_exists()  # Ensure table exists
        data, batch_number = get_data_from_api(API_URL, GROUP_NUMBER)
        
        columns = [
            'brokered_by', 'status', 'price', 'bed', 'bath', 'acre_lot', 
            'street', 'city', 'state', 'zip_code', 'house_size', 'prev_sold_date'
        ]
        
        df = pd.DataFrame(data, columns=columns)
        df['batch_number'] = batch_number
        
        engine = create_engine(RAW_DATABASE_URI)
        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT EXISTS(SELECT 1 FROM raw_data WHERE batch_number = :batch_number)"),
                {'batch_number': batch_number}
            )
            if result.scalar():
                return {"new_data": 0, "batch_number": batch_number}
            else:
                df.to_sql('raw_data', con=engine, if_exists='append', index=False)
                return {"new_data": 1, "batch_number": batch_number}
    
    @task
    def process_data(fetch_result):
        new_data = fetch_result['new_data']
        batch_number = fetch_result['batch_number']
        
        if new_data == 0:
            print("No new data added to db")
            return batch_number
        
        engine = create_engine(RAW_DATABASE_URI)
        df = pd.read_sql('raw_data', engine)
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        engine = create_engine(CLEAN_DATABASE_URI)
        train_df.to_sql('training_data', engine, if_exists='append', index=False)
        val_df.to_sql('validation_data', engine, if_exists='append', index=False)
        test_df.to_sql('test_data', engine, if_exists='append', index=False)
        
        return batch_number

    @task
    def check_data_drift(last_batch_number, threshold=0.1, sample_size=1000):
        engine = create_engine(RAW_DATABASE_URI)
        df_new = pd.read_sql(f"SELECT * FROM raw_data WHERE batch_number = {last_batch_number}", engine)
        df_all = pd.read_sql("SELECT * FROM raw_data", engine)
        df_all.drop_duplicates(inplace=True)
        df_all.dropna(inplace=True)
        
        if df_all['batch_number'].nunique() == 1:
            return False
        else:
            ref_data = df_all[df_all['batch_number'] < last_batch_number]
            curr_data = df_new
            
            if len(ref_data) > sample_size:
                ref_data = ref_data.sample(sample_size, random_state=42)
            if len(curr_data) > sample_size:
                curr_data = curr_data.sample(sample_size, random_state=42)
            
            drift_detected = False
            test_skipped = False
            report_data = []

            for column in ref_data.columns:
                if column == 'batch_number':
                    continue
                if np.issubdtype(ref_data[column].dtype, np.number):
                    if ref_data[column].empty or curr_data[column].empty:
                        print(f"Skipping ks_2samp test for column {column} due to empty data.")
                        test_skipped = True
                        break
                    stat, p_value = ks_2samp(ref_data[column], curr_data[column])
                else:
                    contingency_table = pd.crosstab(ref_data[column], curr_data[column])
                    if contingency_table.size == 0:
                        print(f"Skipping chi2 test for column {column} due to empty contingency table.")
                        test_skipped = True
                        break
                    stat, p_value, _, _ = chi2_contingency(contingency_table)
                
                report_data.append({
                    'column': column,
                    'statistic': stat,
                    'p_value': p_value,
                    'drift_detected': p_value < threshold
                })

                if p_value < threshold:
                    drift_detected = True
            
            if test_skipped:
                return False

            if drift_detected:
                if not os.path.exists(REPORTS_DIR):
                    os.makedirs(REPORTS_DIR)
                report_file = os.path.join(REPORTS_DIR, f'data_drift_report_{last_batch_number}.txt')
                with open(report_file, 'w') as f:
                    for entry in report_data:
                        f.write(f"Column: {entry['column']}, Statistic: {entry['statistic']}, p-value: {entry['p_value']}, Drift Detected: {entry['drift_detected']}\n")

                fig, ax1 = plt.subplots(figsize=(10, 6))

                columns = [entry['column'] for entry in report_data]
                statistics = [entry['statistic'] for entry in report_data]
                p_values = [entry['p_value'] for entry in report_data]
                drift_detected = [entry['drift_detected'] for entry in report_data]

                bar_colors = ['red' if detected else 'green' for detected in drift_detected]
                
                ax1.bar(columns, statistics, color=bar_colors)
                ax1.set_xlabel('Columns')
                ax1.set_ylabel('Statistic Value')
                ax1.set_title('Data Drift Detection')
                plt.xticks(rotation=90)
                
                for i, p_value in enumerate(p_values):
                    ax1.text(i, statistics[i], f'{p_value:.3f}', ha='center', va='bottom')

                plt.tight_layout()
                plt.savefig(os.path.join(REPORTS_DIR, f'data_drift_visual_{last_batch_number}.png'))

        return drift_detected

    @task
    def train_decision(drift_detected):
        return 'train_model' if drift_detected else 'skip_training'

    @task
    def train_model():
        from scripts.train import train_models
        train_models()

    @task
    def skip_training():
        print("Model training skipped.")

    fetch_and_store_data_task = fetch_and_store_data()
    process_data_task = process_data(fetch_and_store_data_task)
    check_data_drift_task = check_data_drift(process_data_task)
    train_decision_task = train_decision(check_data_drift_task)

    train_model_task = train_model()
    skip_training_task = skip_training()

    def decide_branch(**kwargs):
        ti = kwargs['ti']
        return ti.xcom_pull(task_ids='train_decision')

    branch_op = BranchPythonOperator(
        task_id='branching',
        python_callable=decide_branch,
        provide_context=True
    )

    fetch_and_store_data_task >> process_data_task >> check_data_drift_task >> train_decision_task >> branch_op
    branch_op >> train_model_task
    branch_op >> skip_training_task

ml_pipeline_dag = ml_pipeline()
