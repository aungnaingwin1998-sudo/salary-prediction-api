"""
app/main.py
-----------
FastAPI application for salary prediction.

Endpoints
---------
GET  /         → serves the frontend HTML
GET  /api      → API status (JSON)
GET  /health   → health check (model loaded?)
POST /predict  → salary prediction
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    PredictRequest,
    PredictResponse,
    HealthResponse,
    RootResponse,
)

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global model registry ──
MODEL_PATH = os.getenv("MODEL_PATH", "app/model/model.pkl")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

model_registry: dict[str, Any] = {
    "pipeline": None,
    "loaded": False,
}


def load_model() -> None:
    if not os.path.exists(MODEL_PATH):
        logger.error("Model file not found at '%s'. Run train.py first.", MODEL_PATH)
        return
    try:
        model_registry["pipeline"] = joblib.load(MODEL_PATH)
        model_registry["loaded"] = True
        logger.info("Model loaded successfully from '%s'.", MODEL_PATH)
    except Exception as exc:
        logger.exception("Failed to load model: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading ML model …")
    load_model()
    yield
    logger.info("Shutting down.")


# ── FastAPI app ──
app = FastAPI(
    title="Salary Prediction API",
    description="Predicts annual salary based on years of professional experience.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — allows the frontend to call the API from the browser ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve static frontend files ──
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ── Routes ──

@app.get("/", tags=["Frontend"])
def serve_frontend():
    """Serve the web UI."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "Frontend not found. API is running at /api"})


@app.get("/api", response_model=RootResponse, tags=["Status"])
def root() -> RootResponse:
    """Return basic API status."""
    return RootResponse(
        message="Salary Prediction API is running.",
        version="1.0.0",
        docs="/docs",
    )


@app.get("/health", response_model=HealthResponse, tags=["Status"])
def health() -> HealthResponse:
    """Health check — confirms model is loaded and ready."""
    if not model_registry["loaded"]:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unavailable", "model_loaded": False},
        )
    return HealthResponse(status="ok", model_loaded=True)


@app.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
)
def predict(payload: PredictRequest) -> PredictResponse:
    """Predict annual salary given years of professional experience."""
    if not model_registry["loaded"] or model_registry["pipeline"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. The service is not ready.",
        )
    try:
        X = np.array([[payload.years_experience]])
        prediction: float = float(model_registry["pipeline"].predict(X)[0])
        prediction = max(prediction, 0.0)
        logger.info(
            "Prediction: years_experience=%.2f → salary=%.2f",
            payload.years_experience,
            prediction,
        )
        return PredictResponse(
            predicted_salary=round(prediction, 2),
            years_experience=payload.years_experience,
        )
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during prediction.",
        ) from exc
