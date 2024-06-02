FROM apache/airflow:2.6.0-python3.8

USER airflow

RUN pip install --upgrade pip &&
RUN pip install mlflow==2.10.2 \
    pandas \
    sqlalchemy \
    pymysql \
    scikit-learn \ 
    matplotlib \
    evidently \
    pycaret \
    tensorflow-data-validation \
    shap\
    urllib3==1.26.6 \
    botocore==1.21.0 \
    boto3

    