
BATCH_SIZE = 16
OPTIMIZER = 'sgd'
LEARNING_RATE = 0.02
MOMENTUM = 0
L2 = 1e-8
LR_DECAY = 0
EPOCHS = 100

USE_CRF = True
#USE_CRF = False
DROPOUT = 0.25
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
LSTM_LAYER = 1
BI_DIRECT = True

PICKLE_DIR = 'data/data.pickle'
# TRAIN_DIR = '../../lrner_data/NER.train'
# DEV_DIR = '../../lrner_data/NER.dev'
# TRAIN_DIR = 'data/trainFin.txt'
TRAIN_DIR = 'data/trainFin.txt'
DEV_DIR = 'data/Findev.txt'
TEST_DIR = 'data/Fintest.txt'
VOCAB_SIZE = 1
TAG_SIZE = 1
TAG_SCHEME = 'BIO'
SEED_NUM = None
USE_GPU = True
