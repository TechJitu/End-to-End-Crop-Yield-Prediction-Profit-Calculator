import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import sys

# --- 1. Define Data Input Model ---
class CropInput(BaseModel):
    Crop: str = Field(..., example="Rice")
    Season: str = Field(..., example="Kharif")
    State: str = Field(..., example="Assam")
    Crop_Year: int = Field(..., example=2024)
    Area: float = Field(..., example=50000.0)
    Annual_Rainfall: float = Field(..., example=1500.0)
    Fertilizer: float = Field(..., example=700000.0)
    Pesticide: float = Field(..., example=20000.0)

# --- 2. Initialize FastAPI App ---
app = FastAPI(
    title="Agri-Climate Yield Forecaster API",
    description="API to predict crop yield based on climate and farming inputs.",
    version="1.0.0"
)

# --- 3. Add CORS Middleware ---
# This allows your front-end (running on a different domain)
# to make requests to this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- 4. Load the ML Model ---
# This will be run once on startup.
pipeline = None

@app.on_event("startup")
async def load_model():
    """
    Load the trained machine learning pipeline from disk when the API starts.
    """
    global pipeline
    model_path = "crop_yield_model.joblib"
    try:
        pipeline = joblib.load(model_path)
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found.")
        print("Please run 'python ml_model.py' first to train and save the model.")
        # We exit here because the API is useless without the model.
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

# --- 5. Define API Endpoints ---

@app.get("/")
async def root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "Welcome to the Crop Yield Prediction API! Model is loaded and ready."}

@app.post("/predict")
async def predict_yield(data: CropInput):
    """
    Endpoint to make a new prediction.
    
    Accepts a JSON object matching the `CropInput` schema and returns
    a JSON object with the predicted yield.
    """
    if pipeline is None:
        # This check is technically redundant because of the startup event,
        # but it's good practice.
        return {"error": "Model is not loaded. Please check API server logs."}, 503

    try:
        # 1. Convert the input data into a dictionary
        data_dict = data.model_dump()
        
        # 2. Convert the dictionary into a single-row pandas DataFrame
        # The pipeline expects a DataFrame in the same format it was trained on.
        input_df = pd.DataFrame([data_dict])
        
        # 3. Make the prediction
        # The .predict() method returns a numpy array, so we get the first item [0]
        prediction = pipeline.predict(input_df)
        
        # 4. Return the prediction
        return {"predicted_yield": prediction[0]}

    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"error": f"Prediction failed: {str(e)}"}, 400

# --- 6. Run the API Server ---
if __name__ == "__main__":
    # This runs the Uvicorn server when you execute `python backend_api.py`
    # 'app' is the name of your FastAPI instance
    # host="0.0.0.0" makes it accessible on your network
    # port=8000 is the standard port
    # reload=True automatically restarts the server when you save changes.
    uvicorn.run("backend_api:app", host="0.0.0.0", port=8000, reload=True)

