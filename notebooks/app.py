import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Load the trained model (ensure the correct path)
model = joblib.load("trained_model.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define the input schema using Pydantic BaseModel
class ModelInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

# Define the output schema for prediction results
class ModelOutput(BaseModel):
    prediction: int

# Define the POST route for predictions
@app.post("/predict", response_model=ModelOutput)
async def predict(input_data: ModelInput):
    # Extract input features and reshape for the model
    input_values = [[
        input_data.feature1,
        input_data.feature2,
        input_data.feature3,
        input_data.feature4
    ]]
    
    # Perform prediction using the loaded model
    try:
        prediction = model.predict(input_values)[0]  # Get the first prediction
    except Exception as e:
        return {"error": str(e)}  # Handle any errors in prediction
    
    # Return the prediction as an output
    return ModelOutput(prediction=int(prediction))

# Root endpoint for testing if the API is running
@app.get("/")
def read_root():
    return {"message": "Credit Scoring Model API is up and running"}
