from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

current_dir = os.path.dirname(__file__)

app.mount("/static", StaticFiles(directory=os.path.join(current_dir, "static")), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import pandas as pd

model_path = os.path.abspath(os.path.join(current_dir, "..", "models", "model.pkl"))
model = joblib.load(model_path)


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
    with open(html_path, "r") as f:
        return HTMLResponse(content=f.read())


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
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
