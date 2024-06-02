import os
import pandas as pd
from sqlalchemy import create_engine
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import mlflow

# MLFlow configs
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio:9000"
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
mlflow.set_tracking_uri("http://mlflow_server:5000")
mlflow.set_experiment("mlflow_training_tracking")

CLEAN_DATABASE_URI = 'postgresql+psycopg2://cleanusr:cleanpass@cleandb:5432/cleandb'
RANDOM_STATE = 42

def get_training_data(sample_size=None):
    engine = create_engine(CLEAN_DATABASE_URI)
    train_df = pd.read_sql('training_data', engine)
    
    # Convert all column names to strings
    train_df.columns = train_df.columns.map(str)
    
    # Drop specific columns
    columns_to_drop = [
        'batch',
        'brokered_by',
        'prev_sold_date',
        'zip_code',
        'street',
        'city'
    ]
    train_df = train_df.drop(columns=columns_to_drop, errors='ignore')
    
    if sample_size:
        train_df = train_df.sample(n=sample_size, random_state=RANDOM_STATE)
    
    return train_df

def get_test_data():
    engine = create_engine(CLEAN_DATABASE_URI)
    test_df = pd.read_sql('test_data', engine)
    
    # Convert all column names to strings
    test_df.columns = test_df.columns.map(str)
    
    # Drop specific columns
    columns_to_drop = [
        'batch',
        'brokered_by',
        'prev_sold_date',
        'zip_code',
        'street',
        'city'
    ]
    test_df = test_df.drop(columns=columns_to_drop, errors='ignore')
    
    return test_df

def train_models(sample_size=100): 
    df = get_training_data(sample_size)
    y_train = df['price']
    X_train = df.drop(columns=['price'])
    
    # One hot encoding
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    column_transform = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), categorical_features),
        remainder='passthrough'
    )

    # Pipeline
    pipeline = Pipeline(steps=[
        ("column_transformer", column_transform),
        ("scaler", StandardScaler(with_mean=False)),
        ("random_forest", RandomForestRegressor())
    ]) 

    param_grid = { 
        "random_forest__max_depth": [5, 10, 15],
        "random_forest__n_estimators": [100, 150, 200]
    }

    model_name = "random_forest"
    search_rf = GridSearchCV(pipeline, param_grid, n_jobs=2)

    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True, registered_model_name=model_name)
    with mlflow.start_run(run_name="random_forest_run") as run:
        search_rf.fit(X_train, y_train)

def train_gbm(sample_size=100):
    df = get_training_data(sample_size)
    y_train = df['price']
    X_train = df.drop(columns=['price'])

    model_name = "gbm"
    non_numerical_features = ["state"]

    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True, registered_model_name=model_name)

    column_transform = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), non_numerical_features),
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ("column_transform", column_transform),
        ("scaler", StandardScaler(with_mean=False)),
        ("gbm", GradientBoostingRegressor())
    ])

    with mlflow.start_run(run_name="gbm_run") as run:
        pipeline.fit(X_train, y_train)

if __name__ == "__main__":
    train_models(sample_size=100)
