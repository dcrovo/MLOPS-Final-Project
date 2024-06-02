from airflow import DAG
from airflow.decorators import dag, task
from datetime import datetime
from sqlalchemy import create_engine
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from airflow.operators.python import BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from scipy.stats import ks_2samp, chi2_contingency
import numpy as np

API_URL = "http://10.43.101.149:80"
GROUP_NUMBER = 6
RAW_DATABASE_URI = f'postgresql+psycopg2://rawusr:rawpass@rawdb:5432/rawdb'
CLEAN_DATABASE_URI = f'postgresql+psycopg2://cleanusr:cleanpass@cleandb:5432/cleandb'

REPORTS_DIR = '/opt/airflow/logs/reports'

@dag(
    default_args={
        'owner': 'airflow',
        'start_date': datetime(2024, 1, 1),
    },
    description='Fetch data from API and store in RAW DB',
    catchup=False,
    tags=['ml_pipeline'],
)
def ml_pipeline():
    
    @task
    def fetch_and_store_data():
        from scripts.utils import get_data_from_api
        from sqlalchemy import create_engine, text
        import pandas as pd
        
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
        from sqlalchemy import create_engine
        import pandas as pd
        
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
    def check_data_drift(last_batch_number, threshold=0.05):
        engine = create_engine(RAW_DATABASE_URI)
        
        # Load new data
        df_new = pd.read_sql(f"SELECT * FROM raw_data WHERE batch_number = {last_batch_number}", engine)
        
        # Load reference data
        df_all = pd.read_sql("SELECT * FROM raw_data", engine)
        df_all.drop_duplicates(inplace=True)
        df_all.dropna(inplace=True)
        
        if df_all['batch_number'].nunique() == 1:
            drift_detected = True
        else:
            ref_data = df_all[df_all['batch_number'] < last_batch_number]
            curr_data = df_new
            
            # Initialize the drift status
            drift_detected = False

            # Iterate through columns and apply the appropriate test
            for column in ref_data.columns:
                if column == 'batch_number':
                    continue
                if np.issubdtype(ref_data[column].dtype, np.number):
                    # For numerical features, use the KS test
                    stat, p_value = ks_2samp(ref_data[column], curr_data[column])
                else:
                    # For categorical features, use the Chi-squared test
                    contingency_table = pd.crosstab(ref_data[column], curr_data[column])
                    stat, p_value, _, _ = chi2_contingency(contingency_table)
                
                # Check if p-value is below the threshold
                if p_value < threshold:
                    drift_detected = True
                    break
            
            # Save drift report (optional)
            if not os.path.exists(REPORTS_DIR):
                os.makedirs(REPORTS_DIR)
            report_file = os.path.join(REPORTS_DIR, f'data_drift_report_{last_batch_number}.txt')
            with open(report_file, 'w') as f:
                f.write(f'Drift Detected: {drift_detected}\n')
                f.write(f'Column: {column}\n')
                f.write(f'p-value: {p_value}\n')

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
