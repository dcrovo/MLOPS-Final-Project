import os
import pandas as pd
from sqlalchemy import create_engine
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from mlflow import MlflowClient
import mlflow
import mlflow.sklearn
import logging
from mlflow.models.signature import infer_signature
import shap
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLFlow configs
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio:9000"
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
mlflow.set_tracking_uri("http://mlflow_server:5000")
mlflow.set_experiment("realtor_price_prediction")

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
        'prev_sold_date',
        'zip_code',
        'batch_number'
    ]
    train_df = train_df.drop(columns=columns_to_drop, errors='ignore')
    
    if sample_size:
        train_df = train_df.sample(n=sample_size, random_state=RANDOM_STATE)
    
    return train_df

def get_test_data(sample_size=None):
    engine = create_engine(CLEAN_DATABASE_URI)
    test_df = pd.read_sql('test_data', engine)
    
    # Convert all column names to strings
    test_df.columns = test_df.columns.map(str)
    
    # Drop specific columns
    columns_to_drop = [
        'batch',
        'prev_sold_date',
        'zip_code',
        'batch_number'
    ]
    test_df = test_df.drop(columns=columns_to_drop, errors='ignore')
    
    if sample_size:
        test_df = test_df.sample(n=sample_size, random_state=RANDOM_STATE)
    
    return test_df

def train_models(sample_size=100):
    df_train = get_training_data(sample_size)
    df_test = get_test_data(sample_size)
    
    y_train = df_train['price']
    X_train = df_train.drop(columns=['price'])
    y_test = df_test['price']
    X_test = df_test.drop(columns=['price'])
    
    # One hot encoding
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    column_transform = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), categorical_features),
        remainder='passthrough'
    )

    # RandomForest Pipeline
    pipeline_rf = Pipeline(steps=[
        ("column_transformer", column_transform),
        ("scaler", StandardScaler(with_mean=False)),
        ("random_forest", RandomForestRegressor(random_state=RANDOM_STATE))
    ])

    param_grid_rf = { 
        "random_forest__max_depth": [5, 10],
        "random_forest__n_estimators": [50, 100]
    }

    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

    search = GridSearchCV(pipeline_rf, param_grid_rf, n_jobs=4)
    with mlflow.start_run(run_name="random_forest_run") as run:
        logger.info("Starting Random Forest training")
        search.fit(X_train, y_train)
        logger.info("Random Forest training completed")

        best_model = search.best_estimator_
        best_params = search.best_params_
        y_pred = best_model.predict(X_test)
        signature = infer_signature(X_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        
        mlflow.log_param("best_params", best_params)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="random_forest_best",
            signature=signature,
            registered_model_name="random_forest_best_model",
        )

        logger.info(f"Best parameters: {best_params}")
        logger.info(f"RMSE: {rmse}")
        
        # Register the model
        client = MlflowClient()
        model_name = "random_forest_best_model"
        latest_version_info = client.get_latest_versions(name=model_name, stages=["None"])
        version = latest_version_info[0].version if latest_version_info else 1
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )


if __name__ == "__main__":
    train_models(sample_size=None)
