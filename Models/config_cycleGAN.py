import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/artworks/train"
VAL_DIR = "data/artworks/val"
LEARNING_RATE = 3e-4
BATCH_SIZE = 1
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = False

CHECKPOINT_GEN_SKETCH = "Models/outputs/cycleGAN/trained_models/gens_" + str(LEARNING_RATE) + "_" + str(NUM_EPOCHS) + ".pth.tar"
CHECKPOINT_GEN_REAL = "Models/outputs/cycleGAN/trained_models/genr_" + str(LEARNING_RATE) + "_" + str(NUM_EPOCHS) + ".pth.tar"
CHECKPOINT_CRITIC_SKETCH = "Models/outputs/cycleGAN/trained_models/critics_" + str(LEARNING_RATE) + "_" + str(NUM_EPOCHS) + ".pth.tar"
CHECKPOINT_CRITIC_REAL = "Models/outputs/cycleGAN/trained_models/criticr" + str(LEARNING_RATE) + "_" + str(NUM_EPOCHS) + ".pth.tar"
OUTPUT_DIR = "Models/outputs/cycleGAN"
OUTPUT_DIR_EVAL = "Models/outputs/cycleGAN/evaluation"

G_LOSS_LIST = []
D_LOSS_LIST = []
RANDOM_INDEX = 0
VAL_INDEX = 0

