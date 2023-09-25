import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_MODEL = True
SAVE_MODEL = True
MODEL_PATH = "Models/feature_extractor/autoencoder.pth"