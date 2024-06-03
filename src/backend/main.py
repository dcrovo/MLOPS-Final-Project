from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import mlflow
import pandas as pd
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from mlflow import MlflowClient
from sqlalchemy import create_engine, text
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables for MinIO and MLflow
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://minio:9000"
os.environ['AWS_ACCESS_KEY_ID'] = 'minioadmin'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minioadmin'

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://mlflow:5000")

# Load the model from MLflow
model_name = "random_forest_best_model"
model_uri = f"models:/{model_name}/production"
model = mlflow.pyfunc.load_model(model_uri=model_uri)
client = MlflowClient()

# Get the model version
latest_versions = client.get_latest_versions(model_name, stages=["Production"])
model_version = latest_versions[0].version if latest_versions else "Unknown"

# Initialize FastAPI app
app = FastAPI()

# Database URIs
RAW_DATABASE_URI = 'postgresql+psycopg2://rawusr:rawpass@rawdb:5432/rawdb'
CLEAN_DATABASE_URI = 'postgresql+psycopg2://cleanusr:cleanpass@cleandb:5432/cleandb'

# Define input schema with default values
class RealtorModel(BaseModel):
    brokered_by: str = "30287.0"
    status: str = "sold"
    bed: int = 2
    bath: int = 1
    acre_lot: float = 0.1
    street: str = "1089229.0"
    city: str = "Salamanca"
    state: str = "New York"
    house_size: float = 960.0

# Define root endpoint

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
        train_df = train_df.sample(n=sample_size, random_state=42)
    
    return train_df

@app.get("/")
async def root():
    return {"message": "Welcome to the Realtor Price Prediction API! Serving model: {}".format(model_version)}

# Define predict endpoint
@app.post("/predict/")
async def predict_price(data: RealtorModel):
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([data.dict()])
        
        # Perform prediction
        prediction = model.predict(input_data)
        
        # Append the prediction result to the database
        input_data['prediction'] = prediction[0]
        input_data['model_version'] = model_version
        engine = create_engine(RAW_DATABASE_URI)
        input_data.to_sql('predictions', engine, if_exists='append', index=False)
        
        # Return prediction
        return {"prediction": f"${prediction[0]:,.2f}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Define endpoint to get unique values for dropdowns
@app.get("/unique_values/")
async def get_unique_values():
    try:
        engine = create_engine(CLEAN_DATABASE_URI)
        with engine.connect() as connection:
            brokered_by = pd.read_sql("SELECT DISTINCT brokered_by FROM training_data", connection)['brokered_by'].tolist()
            street = pd.read_sql("SELECT DISTINCT street FROM training_data", connection)['street'].tolist()
            city = pd.read_sql("SELECT DISTINCT city FROM training_data", connection)['city'].tolist()
            state = pd.read_sql("SELECT DISTINCT state FROM training_data", connection)['state'].tolist()
        
        return {
            "brokered_by": brokered_by,
            "street": street,
            "city": city,
            "state": state
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Define SHAP values endpoint
@app.post("/shap/")
async def shap_values(data: RealtorModel):
    try:

        # Retrieve training data for preprocessing
        df_train = get_training_data(100)
        X_train = df_train.drop(columns=['price'])
        
        # Extract the categorical features and create the ColumnTransformer
        categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
        logger.info("before columns.")
        column_transformer = make_column_transformer(
            (OneHotEncoder(handle_unknown='ignore'), categorical_features),
            remainder='passthrough'
        )
        
        # Fit the ColumnTransformer and apply to input data
        column_transformer.fit(X_train)
        input_data = pd.DataFrame([data.dict()])
        transformed_input_data = column_transformer.transform(input_data)
        
        # Convert the sparse matrix to a dense format
        transformed_input_data_dense = transformed_input_data.toarray()
        
        # Extract the raw model from the pipeline
        pipeline = model._model_impl
        rf_model = pipeline.named_steps["random_forest"]
        
        # Initialize SHAP explainer using the raw model
        explainer = shap.TreeExplainer(rf_model)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(transformed_input_data_dense)
        
        # Plot SHAP values
        plt.figure()
        shap.summary_plot(shap_values, transformed_input_data_dense, show=False)
        
        # Save the plot to a BytesIO object
        # buf = BytesIO()
        # plt.savefig(buf, format="png")
        plt.savefig(f'/opt/code/img/shap_{model_name}_v{model_version}.png', format="png")
        # buf.seek(0)
        plt.close()
        
        # Encode the plot as base64
        # plot_base64 = base64.b64encode(buf.read()).decode("utf-8")
        
        # Return the plot
        return {"shap_plot": "plot_base64"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
