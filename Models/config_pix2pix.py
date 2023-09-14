import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/artworks/train"
VAL_DIR = "data/artworks/val"
LEARNING_RATE = 3e-4
BATCH_SIZE = 8
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA = 1
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "Models/outputs/pix2pix/trained_models/disc_" + str(LEARNING_RATE) + "_" + str(NUM_EPOCHS) + ".pth.tar"
CHECKPOINT_GEN = "Models/outputs/pix2pix/trained_models/gen_" + str(LEARNING_RATE) + "_" + str(NUM_EPOCHS) + ".pth.tar"
OUTPUT_DIR = "Models/outputs/pix2pix"

G_LOSS_LIST = []
D_LOSS_LIST = []
RANDOM_INDEX = 0
VAL_INDEX = 0


