from airflow import DAG
from airflow.models import Variable
from airflow.operators.python_operator import PythonOperator, ShortCircuitOperator,BranchOperator
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime, timedelta
import requests
import os
from scripts.train import train_random_forest, train_gbm


# Definiciones globales
GROUP_NUMBER = 6
API_URL = f"http://10.43.101.149:80/data?group_number={GROUP_NUMBER}"
DATABASE_URI = 'mysql+pymysql://modeldbuser:modeldbpass@modeldb:3306/modeldb'

    
def fetch_and_store_data(**kwargs):
    '''
    Fetch data from the API and store it directly into the database.
    '''
    
    # De acuerdo con la documentación de Arflow, top-level code e imports pesados
    # deben ser importados en la función llamable
    max_calls = 10 
    import pandas as pd
    from sqlalchemy import create_engine

    success_calls = int(Variable.get("success_calls", default_var=0))

    response = requests.get(API_URL)

    if response.status_code == 200:
        data = response.json()['data']
        success_calls  += 1
        Variable.set("success_calls", str(success_calls))
        # Define las columnas del DataFrame según el esquema proporcionado
        columns = [
            "Elevation",
            "Aspect",
            "Slope",
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points",
            "Wilderness_Area",
            "Soil_Type",
            "Cover_Type"
        ]
        # Crea un DataFrame con los datos
        df = pd.DataFrame(data, columns=columns)

                
        # Almacena los datos en la base de datos
        engine = create_engine(DATABASE_URI)
        df.to_sql('dataset_covertype_table', con=engine, if_exists='append', index=False)
        
        return success_calls >= max_calls

    else:
        if response.status_code == 400: 
            return True
        else:
            raise Exception(f"Error al obtener datos de la API: Estado {response.status_code}")
    

def end_of_dag():
    pass 


def process_data(**kwargs):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sqlalchemy import create_engine



    engine = create_engine(DATABASE_URI)
    df = pd.read_sql('dataset_covertype_table', engine)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_df.to_sql('training_data', engine, if_exists='replace', index=False)
    test_df.to_sql('test_data', engine, if_exists='replace', index=False)



# Configuración del DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_pipeline',
    default_args=default_args,
    description='Fetch data from API and store in ModelDB ',
    schedule_interval=timedelta(minutes=5),
    catchup=False,
)

fetch_and_store_data_task = ShortCircuitOperator(
    task_id='fetch_and_store_data',
    python_callable=fetch_and_store_data,
    provide_context=True,
    dag=dag,
)


process_data_task = PythonOperator(
    task_id = 'process_data',
    python_callable = process_data,
    provide_context = True,
    dag=dag
)

train_random_forest_task = PythonOperator(
    task_id = 'train_random_forest',
    python_callable = train_random_forest,
    dag = dag,
)

train_gbm_task = PythonOperator(
    task_id = 'train_gbm',
    python_callable = train_gbm,
    dag = dag,
)


end = DummyOperator(
    task_id='end',
    trigger_rule='none_failed_or_skipped',
    dag=dag,
)

# Task dependencies
fetch_and_store_data_task >> process_data_task >> [train_random_forest_task, train_gbm_task] >> end
