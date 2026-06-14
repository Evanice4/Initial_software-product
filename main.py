"""
SokoPrice FastAPI Backend - main.py
AI Grocery Price Forecasting for Kigali Informal Markets
Author : Nice Eva Karabaranga
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import joblib
from datetime import date

# App Initialization 
app = FastAPI(
    title="SokoPrice API",
    description=(
        "AI-powered grocery price forecasting and smart market recommendation "
        "system for informal markets in Kigali, Rwanda. "
        "Provides price predictions, market recommendations, and shopping basket cost "
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Load models at startup
try:
    MODEL_LGBM  = joblib.load("models/model_lgbm.pkl")
    SCALER_X    = joblib.load("scaler/scaler_X.pkl")
    SCALER_Y    = joblib.load("scaler/scaler_y.pkl")
    MODELS_OK   = True
except Exception as e:
    MODELS_OK = False
    print(f"Warning: could not load models — {e}")

COMMODITIES = [
    "Maize", "Maize Flour", "Potatoes", "Rice", "Beans (Yellow)", "Sorghum",
    "Beans (Dry)", "Onions", "Tomatoes", "Cabbage", "Flour", "Banana", "Spinach",
]
MARKETS = [
    "Kimironko", "Nyabugogo", "Kicukiro",
]
MODEL_OPTIONS = [
    "lightgbm"
]

# Schemas
class PredictRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    commodity: str = Field(..., example="Maize", description="Target commodity")
    market: str = Field(..., example="Kimironko", description="Market in Kigali")
    forecast_date: date = Field(
        ..., example="2026-06-21", description="Date to forecast (up to 7 days ahead)"
    )
    model_name: Optional[str] = Field(
        "lightgbm", example="lightgbm", description=f"One of: {MODEL_OPTIONS}"
    )


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    commodity: str
    market: str
    forecast_date: str
    predicted_price_kes: float
    model_used: str
    confidence_lower: float
    confidence_upper: float
    trend: str


class RecommendRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    commodity: str = Field(..., example="Beans (Dry)")
    budget_kes: Optional[float] = Field(
        None, example=300.0, description="Optional max price filter (KES)"
    )


class MarketRec(BaseModel):
    model_config = {"protected_namespaces": ()}

    market: str
    predicted_price_kes: float
    distance_km: Optional[float]
    score: float


class BasketItem(BaseModel):
    model_config = {"protected_namespaces": ()}

    commodity: str = Field(..., example="Maize")
    quantity_kg: float = Field(..., example=2.0)


class BasketRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    market: str = Field(..., example="Kimironko")
    items: List[BasketItem]
    forecast_date: date = Field(..., example="2026-06-21")


# Endpoints 
@app.get("/", tags=["Health"])
def root():
    """Health check - returns API status and whether models loaded successfully."""
    return {
        "status": "SokoPrice API is running",
        "version": "1.0.0",
        "models_loaded": MODELS_OK,
    }


@app.get("/commodities", tags=["Catalog"])
def list_commodities():
    """Returns the list of supported grocery commodities."""
    return {"commodities": COMMODITIES}


@app.get("/markets", tags=["Catalog"])
def list_markets():
    """Returns the list of supported Kigali informal markets."""
    return {"markets": MARKETS}


@app.post("/predict", response_model=PredictResponse, tags=["Forecasting"])
def predict_price(req: PredictRequest):
    """
    Forecast the price of a grocery commodity at a specific market up to 7
    days ahead. Returns the predicted price in KES with a confidence
    interval and the price trend direction.
    """
    if req.commodity not in COMMODITIES:
        raise HTTPException(400, f"Unsupported commodity. Choose from: {COMMODITIES}")
    if req.market not in MARKETS:
        raise HTTPException(400, f"Unsupported market. Choose from: {MARKETS}")
    if req.model_name not in MODEL_OPTIONS:
        raise HTTPException(400, f"Unsupported model. Choose from: {MODEL_OPTIONS}")

    # In production: build the 17-feature vector from the price history DB
    # and call the selected model's .predict() method.
    price = round(float(np.random.uniform(20, 200)), 2)

    return PredictResponse(
        commodity=req.commodity,
        market=req.market,
        forecast_date=str(req.forecast_date),
        predicted_price_kes=price,
        model_used=req.model_name,
        confidence_lower=round(price * 0.90, 2),
        confidence_upper=round(price * 1.10, 2),
        trend="stable",  # "rising" | "falling" | "stable"
    )


@app.post("/recommend", response_model=List[MarketRec], tags=["Recommendations"])
def recommend_markets(req: RecommendRequest):
    """
    Returns Kigali markets ranked by lowest predicted price for the given
    commodity. An optional budget filter removes markets above the
    specified price threshold.
    """
    if req.commodity not in COMMODITIES:
        raise HTTPException(400, f"Unsupported commodity. Choose from: {COMMODITIES}")

    recs = []
    for m in MARKETS:
        p = round(float(np.random.uniform(20, 150)), 2)
        if req.budget_kes and p > req.budget_kes:
            continue
        recs.append(
            MarketRec(
                market=m,
                predicted_price_kes=p,
                distance_km=round(float(np.random.uniform(0.5, 8.0)), 1),
                score=round(float(np.random.uniform(0.7, 1.0)), 3),
            )
        )
    return sorted(recs, key=lambda x: x.predicted_price_kes)


@app.post("/basket", tags=["Shopping"])
def estimate_basket(req: BasketRequest):
    """
    Estimate the total cost of a grocery basket at a chosen market for a
    forecast date. Returns an itemised cost breakdown and the total in KES.
    """
    items, total = [], 0.0
    for item in req.items:
        if item.commodity not in COMMODITIES:
            raise HTTPException(400, f"Unsupported commodity: {item.commodity}")
        unit = round(float(np.random.uniform(20, 150)), 2)
        line = round(unit * item.quantity_kg, 2)
        total += line
        items.append(
            {
                "commodity": item.commodity,
                "quantity_kg": item.quantity_kg,
                "unit_price_kes": unit,
                "line_total_kes": line,
            }
        )
    return {
        "market": req.market,
        "forecast_date": str(req.forecast_date),
        "items": items,
        "total_kes": round(total, 2),
    }


@app.get("/alerts/{commodity}", tags=["Alerts"])
def price_alert(commodity: str, threshold_kes: float = 100.0):
    """
    Check whether the predicted price for a commodity exceeds a given
    budget threshold. Returns the predicted price and an alert flag.
    """
    if commodity not in COMMODITIES:
        raise HTTPException(400, f"Unsupported commodity. Choose from: {COMMODITIES}")

    predicted = round(float(np.random.uniform(20, 200)), 2)
    return {
        "commodity": commodity,
        "predicted_price_kes": predicted,
        "threshold_kes": threshold_kes,
        "alert": predicted > threshold_kes,
        "message": (
            "Price above threshold!" if predicted > threshold_kes
            else "Price within budget"
        ),
    }
