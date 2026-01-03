# src/inference/model_loader.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config.settings import MODEL_PATH, DEVICE

class ModelLoader:
    _model = None
    _tokenizer = None

    @classmethod
    def load(cls):
        if cls._model is None or cls._tokenizer is None:
            cls._tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            cls._model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
            cls._model.to(DEVICE)
            cls._model.eval()

        return cls._model, cls._tokenizer
