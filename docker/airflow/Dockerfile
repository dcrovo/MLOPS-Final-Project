FROM apache/airflow:2.6.0-python3.8

# Cambia al usuario root para instalar herramientas del sistema
USER root

# Instala las herramientas de desarrollo necesarias
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Cambia de nuevo al usuario airflow
USER airflow

# Actualiza pip e instala las dependencias de Python
RUN pip install --upgrade pip
RUN pip install mlflow==2.10.2 \
    pandas \
    sqlalchemy \
    pymysql \
    scikit-learn \ 
    matplotlib \
    evidently \
    pycaret \
    tensorflow-data-validation \
    shap \
    urllib3==1.26.6 \
    botocore==1.21.0 \
    boto3

    