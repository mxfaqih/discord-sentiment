# src/config/settings.py

from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "indobert_sentiment"

MAX_LENGTH = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
