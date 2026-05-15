from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "results" / "cleveland" / "models" / "best_pipeline.joblib"

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PatientFeatures(BaseModel):
    age: int = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=0, le=3)
    trestbps: float
    chol: float
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float
    exang: int = Field(ge=0, le=1)
    oldpeak: float
    slope: int = Field(ge=0, le=2)
    ca: int = Field(ge=0, le=3)
    thal: int = Field(description="Expected values: 3, 6, or 7")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run training first: python src/train.py"
        )
    return joblib.load(MODEL_PATH)


@app.get("/health")
def healthcheck():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PatientFeatures):
    try:
        model = load_model()
        X = pd.DataFrame([payload.model_dump()])

        pred = int(model.predict(X)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(X)[0, 1])

        return {
            "prediction": pred,
            "probability": probability,
            "label": "disease" if pred == 1 else "no_disease",
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
