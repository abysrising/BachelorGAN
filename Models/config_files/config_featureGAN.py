import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/artworks/train"
VAL_DIR = "data/artworks/val"
LEARNING_RATE = 2e-6
BATCH_SIZE = 1
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA = 1
NUM_EPOCHS = 10

CHECKPOINT_DISC = "Models/outputs/featureGAN/trained_models/disc_" + str(LEARNING_RATE) + "_" + str(NUM_EPOCHS) + ".pth.tar"
CHECKPOINT_GEN = "Models/outputs/featureGAN/trained_models/gen_" + str(LEARNING_RATE) + "_" + str(NUM_EPOCHS) + ".pth.tar"
OUTPUT_DIR = "Models/outputs/featureGAN"
G_LOSS_LIST = []
D_LOSS_LIST = []
LR_LIST_GEN = []
LR_LIST_DISC = []
RANDOM_INDEX = 0
VAL_INDEX = 0


LOAD_MODEL = False
SAVE_MODEL = False
LR_REDUCTION = False
