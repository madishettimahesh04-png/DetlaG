import torch
import joblib

from model import Model

cfg = joblib.load("model_config.pkl")

model = Model(
    desc_dim=cfg["input_dim"],
    hidden_dim=cfg["hidden_dim"],
    heads=cfg["heads"],
    dropout=cfg["dropout"],
    desc_hidden=cfg["desc_hidden"]
)

state_dict = torch.load(
    "best_model.pt",
    map_location="cpu"
)

model.load_state_dict(state_dict)

model.eval()

print("✅ Model loaded successfully")