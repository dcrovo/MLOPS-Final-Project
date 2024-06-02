import os
import pandas as pd
from sqlalchemy import create_engine
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import mlflow


# MLFlow configs

os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio:9000"
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'
mlflow.set_tracking_uri("http://mlflow_server:5000")
mlflow.set_experiment("mlflow_training_tracking")

DATABASE_URI = 'mysql+pymysql://modeldbuser:modeldbpass@modeldb:3306/modeldb'
TABLE_NAME = 'dataset_covertype_table'
RANDOM_STATE = 42



def get_training_data():
    engine = create_engine(DATABASE_URI)
    train_df = pd.read_sql('training_data', engine)
    # Ajusta los tipos de datos antes de guardar en SQL
    train_df["Wilderness_Area"] = train_df["Wilderness_Area"].astype(str)
    train_df["Soil_Type"] = train_df["Soil_Type"].astype(str)

    # Para las otras columnas, convierte a enteros si es apropiado
    columns_to_exclude = ["Wilderness_Area", "Soil_Type"]

    for columns in train_df.columns:
        if columns not in columns_to_exclude:
            train_df[columns] = pd.to_numeric(train_df[columns])
    return train_df

def get_test_data():
    engine = create_engine(DATABASE_URI)
    test_df = pd.read_sql('test_data', engine)
    # Ajusta los tipos de datos antes de guardar en SQL
    test_df["Wilderness_Area"] = test_df["Wilderness_Area"].astype(str)
    test_df["Soil_Type"] = test_df["Soil_Type"].astype(str)

    # Para las otras columnas, convierte a enteros si es apropiado
    columns_to_exclude = ["Wilderness_Area", "Soil_Type"]

    for columns in test_df.columns:
        if columns not in columns_to_exclude:
            test_df[columns] = pd.to_numeric(test_df[columns])  
    return test_df


def train_random_forest(): 

    df = get_training_data()
    y_train = df['Cover_Type']
    X_train = df.drop(columns=['Cover_Type'])
    
    # One hot encoding
    non_numerical_features = ["Wilderness_Area", "Soil_Type"]
    column_transform = make_column_transformer((OneHotEncoder(handle_unknown='ignore'),
                                                non_numerical_features), remainder='passthrough')

    #Pipeline

    pipeline = Pipeline(steps=[
        ("column_transformer", column_transform),
        ("scaler", StandardScaler(with_mean=False)),
        ("random_forest", RandomForestClassifier())
    ]) 

    param_grid = { 
        "random_forest__max_depth":[5,10,15],
        "random_forest__n_estimators":[100,150,200]
    }

    model_name = "random_forest"
    search_rf = GridSearchCV(pipeline, param_grid, n_jobs=2)

    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True,registered_model_name=model_name)
    with mlflow.start_run(run_name="random_forest_run") as run:

        search_rf.fit(X_train, y_train)


def train_gbm():

    df = get_training_data()
    
    y_train = df['Cover_Type']
    X_train = df.drop(columns=['Cover_Type'])

    model_name = "gbm"

    non_numerical_features = ["Wilderness_Area", "Soil_Type"]

    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True, registered_model_name=model_name)

    column_transform = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), non_numerical_features),
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ("column_transform", column_transform),
        ("scaler", StandardScaler(with_mean=False)),
        ("gbm", GradientBoostingClassifier())
    ])


    #search_gb = GridSearchCV(pipeline, param_grid, n_jobs=2)

    with mlflow.start_run(run_name="gbm_run") as run:


        pipeline.fit(X_train, y_train)
