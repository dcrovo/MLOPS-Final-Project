from airflow import DAG
from airflow.decorators import task, dag
from datetime import datetime, timedelta
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from sqlalchemy import create_engine
import pandas as pd
import tensorflow_data_validation as tfdv
import os
import boto3

API_URL = "http://10.43.101.149:80"
GROUP_NUMBER = 6
RAW_DATABASE_URI = f'postgresql+psycopg2://rawusr:rapass@rawdb:5432/rawdb'
CLEAN_DATABASE_URI = f'postgresql+psycopg2://cleanusr:cleanpass@cleandb:5432/cleandb'
MINIO_URL = "http://minio:9000"
MINIO_BUCKET = "ml-artifacts"
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")

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
    def process_data(new_data, batch_number):
        from sqlalchemy import create_engine
        import pandas as pd
        
        if new_data == 0:
            return
        
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
    def check_data_drift(last_batch_number):
        from evidently import ColumnMapping
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        from sqlalchemy import create_engine
        import pandas as pd
        import boto3

        engine = create_engine(RAW_DATABASE_URI)
        df_new = pd.read_sql(f"SELECT * FROM raw_data WHERE batch_number = {last_batch_number}", engine)
        
        df_all = pd.read_sql("SELECT * FROM raw_data", engine)
        df_all.drop_duplicates(inplace=True)
        df_all.dropna(inplace=True)
        
        ref_data = df_all[df_all['batch_number'] < last_batch_number]
        curr_data = df_new
        
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=ref_data, current_data=curr_data)
        
        report_file = f'/tmp/data_drift_report_{last_batch_number}.html'
        report.save_html(report_file)

        s3 = boto3.client('s3', endpoint_url=MINIO_URL, aws_access_key_id=MINIO_ACCESS_KEY, aws_secret_access_key=MINIO_SECRET_KEY)
        s3.upload_file(report_file, MINIO_BUCKET, f'data_drift_report_{last_batch_number}.html')

        drift_detected = report.as_dict()['metrics'][0]['result']['drift_share'] > 0.1
        
        return drift_detected

    @task
    def check_tfdv_statistics(last_batch_number):
        from sqlalchemy import create_engine
        import tensorflow_data_validation as tfdv
        import pandas as pd
        import os
        import json
        import boto3

        engine = create_engine(RAW_DATABASE_URI)
        df = pd.read_sql(f"SELECT * FROM raw_data WHERE batch_number = {last_batch_number}", engine)
        
        stats = tfdv.generate_statistics_from_dataframe(df)
        anomalies = tfdv.validate_statistics(stats, previous_stats)

        report_file = f'/tmp/tfdv_anomalies_{last_batch_number}.json'
        with open(report_file, 'w') as f:
            json.dump(tfdv.utils.display_util.anomalies_to_json(anomalies), f)

        s3 = boto3.client('s3', endpoint_url=MINIO_URL, aws_access_key_id=MINIO_ACCESS_KEY, aws_secret_access_key=MINIO_SECRET_KEY)
        s3.upload_file(report_file, MINIO_BUCKET, f'tfdv_anomalies_{last_batch_number}.json')

        anomalies_detected = len(anomalies.anomaly_info) > 0
        
        return anomalies_detected

    @task
    def train_decision(drift_detected, anomalies_detected):
        return drift_detected or anomalies_detected

    @task
    def train_model():
        from scripts.train import train_model
        train_model()
    
    fetch_and_store_data_task = fetch_and_store_data()
    process_data_task = process_data(fetch_and_store_data_task["new_data"], fetch_and_store_data_task["batch_number"])
    check_data_drift_task = check_data_drift(process_data_task)
    check_tfdv_statistics_task = check_tfdv_statistics(process_data_task)
    train_decision_task = train_decision(check_data_drift_task, check_tfdv_statistics_task)
    train_model_task = train_model()
    
    fetch_and_store_data_task >> process_data_task >> [check_data_drift_task, check_tfdv_statistics_task] >> train_decision_task >> train_model_task

ml_pipeline_dag = ml_pipeline()
