    import uvicorn
    import joblib
    import pandas as pd
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import os
    import sys

    # --- 1. Define Data Input Model ---
    # We keep the validation strict to ensure data quality in production
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
        description="Production API for predicting crop yield.",
        version="1.0.0"
    )

    # --- 3. CORS Configuration (CRITICAL FOR PRODUCTION) ---
    # In production, replace ["*"] with your actual frontend domain (e.g., ["https://my-agri-app.vercel.app"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- 4. Load Model Globally ---
    # We load this once at startup so we don't read the disk on every request
    model_pipeline = None

    @app.on_event("startup")
    def load_model():
        global model_pipeline
        try:
            # Ensure crop_yield_model.joblib is in the same directory
            model_pipeline = joblib.load('crop_yield_model.joblib')
            print("Model loaded successfully.")
        except Exception as e:
            print(f"CRITICAL ERROR: Could not load model. {e}")
            # In production, we might want to halt here, but we'll print for logs
            model_pipeline = None

    @app.get("/")
    def health_check():
        """Simple health check endpoint for monitoring tools."""
        return {"status": "active", "model_loaded": model_pipeline is not None}

    # --- 5. Prediction Endpoint ---
    @app.post("/predict")
    def predict_yield(data: CropInput):
        if model_pipeline is None:
            raise HTTPException(status_code=503, detail="Model not loaded on server.")

        try:
            # Convert input to DataFrame
            data_dict = data.dict()
            input_df = pd.DataFrame([data_dict])
            
            # Predict
            prediction = model_pipeline.predict(input_df)
            
            # Return JSON
            return {"predicted_yield": float(prediction[0])}

        except Exception as e:
            print(f"Error during prediction: {e}")
            raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

    # --- 6. Execution Entry Point ---
    if __name__ == "__main__":
        # Get port from environment variable (Required for Render/Heroku)
        port = int(os.environ.get("PORT", 8000))
        uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
