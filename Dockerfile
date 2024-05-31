FROM apache/airflow:2.6.0-python3.8

USER airflow

RUN pip install --upgrade pip && \
    pip install mlflow==2.10.2 pandas sqlalchemy pymysql scikit-learn matplotlib xgboost