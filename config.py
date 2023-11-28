TRAIN_SAMPLE_PATH = './data/train.csv'
DEV_SAMPLE_PATH = './data/dev.csv'

BGE_PAD_ID = 0
TEXT_LEN = 200

BGE_MODEL = 'BAAI/bge-large-zh-v1.5'
# BGE_MODEL = './bge_model'
MODEL_DIR = './model/'
TRAINED_MODEL = './trained_model/15.pth'

SOURCE_DIR = './report_sample'
OUTPUT_DIR = './output'

EMBEDDING_DIM = 1024
NUM_CLASSES = 11

EPOCH = 100
LR = 1e-5

import torch

if torch.cuda.is_available():
    torch.cuda.set_device(0)  # This sets the current CUDA device to GPU0
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
