from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import mlflow
import pandas as pd
from mlflow import MlflowClient

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
        
        # Return prediction
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Example to test the predict function
test_data = {
    "brokered_by": 30287.0,
    "status": "sold",
    "bed": 2,
    "bath": 1,
    "acre_lot": 0.1,
    "street": 1089229.0,
    "city": "Salamanca",
    "state": "New York",
    "house_size": 960
}