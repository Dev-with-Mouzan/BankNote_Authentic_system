from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, ".."))

app.mount("/static", StaticFiles(directory=os.path.join(current_dir, "static")), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import pandas as pd

model_path = os.path.join(project_root, "models", "model.pkl")
logger.info(f"Loading model from: {model_path}")

model_error = None
try:
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at {model_path}")
    model = joblib.load(model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model_error = str(e)
    model = None


class PredictRequest(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    message: str


@app.get("/")
async def root():
    html_path = os.path.join(current_dir, "templates", "index.html")
    if not os.path.exists(html_path):
        logger.error(f"Template not found at: {html_path}")
        return HTMLResponse(content=f"<h1>Error</h1><p>Template not found. Check server logs.</p>", status_code=404)
    return FileResponse(html_path)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model could not be loaded. Please check deployment logs.")
    try:
        features = np.array(
            [[request.variance, request.skewness, request.curtosis, request.entropy]]
        )
        
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        
        print(f"Input: {features}, Prediction: {prediction}")

        return PredictResponse(
            prediction=int(prediction),
            probability=float(max(proba)),
            message="Fake Banknote" if prediction == 1 else "Legitimate Banknote",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
