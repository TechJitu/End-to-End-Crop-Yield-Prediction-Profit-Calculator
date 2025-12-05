# End-to-End Crop Yield Prediction & Profit Calculator

This project is a complete end-to-end Machine Learning application that predicts agricultural crop yield based on farming and environmental factors. It features a trained ML model, a Python backend API, and an interactive web-based frontend.

## Features

ML Prediction: Uses a RandomForestRegressor model trained on historical Indian crop data (crop_yield.csv) to predict yield in tons per hectare.

Backend API: A FastAPI backend loads the trained model and serves predictions via a /predict API endpoint.

Interactive Frontend: A single-page web app (HTML/CSS/JS) provides a user-friendly form to input 8 key factors (Crop, Season, State, Year, Area, Rainfall, Fertilizer, Pesticide).

Profit Calculator: An additional feature that allows the user to input their estimated selling price (per ton) and farming costs (per hectare) to instantly calculate the total estimated profit based on the predicted yield.

## Tech Stack

Machine Learning: Python, scikit-learn, pandas, joblib

Backend: Python, FastAPI, uvicorn

Frontend: HTML, Tailwind CSS, Vanilla JavaScript

# How to Run

Install Dependencies:
```pip install -r requirements.txt ```
(You'll need pandas, scikit-learn, fastapi, uvicorn, joblib)

# Train the Model:

Run the ML script once to create the crop_yield_model.joblib file.

## Run the Frontend:
index.html

## Start the FastAPI backend server:

```uvicorn main:app```

## Contributor

Frontend & Backend: [Zuber Shaikh](https://github.com/zubershk)
