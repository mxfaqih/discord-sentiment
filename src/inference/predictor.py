# src/inference/predictor.py

import torch
from src.inference.model_loader import ModelLoader
from src.preprocessing.text_cleaner import clean_text
from src.config.settings import MAX_LENGTH, DEVICE
from src.utils.logger import logger

LABEL_MAP = {
    0: "negative",
    1: "positive"
}

def predict_sentiment(text: str) -> dict:
    model, tokenizer = ModelLoader.load()

    cleaned_text = clean_text(text)

    logger.info(f"Input text (cleaned): {cleaned_text[:100]}")

    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()

    result = {
        "label": LABEL_MAP[pred_id],
        "confidence": round(probs[0][pred_id].item(), 4)
    }

    logger.info(
        f"Prediction: label={result['label']} | confidence={result['confidence']}"
    )

    return result
