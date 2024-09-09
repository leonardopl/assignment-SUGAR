import time

# Data and experiment configuration
DATA_DIR = "dataset/Pinhole"
EXP_NAME = "test_focal_length"
EXP_ID = EXP_NAME + "-" + time.strftime("%Y%m%d_%H_%M_%S", time.localtime(time.time()))
ROOT_DIR = "output/experiments/" + EXP_ID

# Training parameters
BATCH_SIZE_TRAIN = 8
BATCH_SIZE_VAL = 4
MAX_EPOCHS = 50
VAL_INTERVAL = 1
LR = 1e-4
PATIENCE = 15

# Image parameters and normalization for EfficientNetV2
IMG_SIZE = (480, 480)
IMG_MEAN = [0.5, 0.5, 0.5]
IMG_STD = [0.5, 0.5, 0.5]

# Device configuration
DEVICE = "cuda:0"
VAL_AMP = True
