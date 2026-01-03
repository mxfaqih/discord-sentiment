# src/api/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.inference.predictor import predict_sentiment

app = FastAPI(
    title="Sentiment Analysis API",
    description="Binary sentiment classification for Indonesian text",
    version="1.0.0"
)

# === CORS CONFIGURATION ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # for local/demo purposes
    allow_credentials=True,
    allow_methods=["*"],          # allow POST, OPTIONS, etc.
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    confidence: float

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.text or request.text.strip() == "":
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    result = predict_sentiment(request.text)
    return result
