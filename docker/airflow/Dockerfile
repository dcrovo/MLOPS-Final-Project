FROM apache/airflow:2.6.0-python3.8

# Switch to root user to install system tools
USER root

# Install the necessary development tools
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Switch back to the airflow user
USER airflow

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir mlflow==2.10.2 \
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