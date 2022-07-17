import os
from enum import Enum

class DISTRIBUTIONS(Enum):
    REAL = 1
    REAL_AND_APPARENT = 2

class APPROACH(Enum):
    REGRESSION = 1
    CLASSIFIER = 2
    INVERSE = 3
    STATIC = 4
    DYNAMIC = 5
    COURA = 6
    DCOURA =7

class TRAINING(Enum):
    SMART_DECAY = 1
    DECAY_PERIOD = 2

class KDE_KERNEL(Enum):
    GAUSSIAN = 'gaussian'
    TOPHAT = 'tophat'
    EPANECHNIKOV = 'epanechnikov'
    EXPONENTIAL = 'exponential'
    LINEAR = 'linear'
    COSINE = 'cosine'

class SET(Enum):
    TRAINING = 0
    VALIDATION = 1
    TEST = 2

USING_GPU = True  # Train model on GPU if True, on CPU if False
USING_PARALLEL = True  # Train model on PARALLEL GPU if True, on CPU if False

PIXEL_MEAN = [0.485, 0.456, 0.406]  # Mean values for RGB channels. Default [0.485, 0.456, 0.406] is for IMAGENET
PIXEL_STD = [0.229, 0.224, 0.225]  # Std dev values for RGB channels. Default [0.229, 0.224, 0.225] is for IMAGENET
NUM_WORKERS = 1  # Number of subprocesses simultaneously loading data from the system

IMDB_WIKI_DIR = os.path.join("/home/research/dcoura/IMDB-WIKI/")
APPA_REAL_DIR = os.path.join("/mnt/fastdata/datasets/appa-real/appa-real-release")
#APPA_REAL_DIR = os.path.join("D:\\database faces\\APPA-REAL\\appa-real-release")

NUM_EPOCHS = 120
PRINT_EVERY = 1

BATCH_SIZE = 128 # Number of images in each batch
INITIAL_LR = 0.00005
LR_REDUCING_FACTOR = 0.1
WEIGHT_DECAY = 10**-5  # Factor by which to regularize the weights during optimization
TRAINING_TYPE = TRAINING.SMART_DECAY
SMART_DECAY_PATIENCE = 5
LR_DECAY_PERIOD = 20  # How often/when to drop the learning rate, depending on whether an int or a list
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 20

SINGLE_ALPHA = 0.8
SINGLE_STD_DEV = 3
REAL_STD_DEV_IMDB_WIKI = 3
REAL_STD_DEV = [2.7, 3.3, 2.4, 3.6]
#REAL_STD_DEV = [1.8, 2.1, 3.9, 4.2]
#ALPHA = [1, 0.9, 1.1, 0.8, 1.2]
ALPHA = [1, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4]
#LAMBDA = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
LAMBDA = [0.8]

DIR_NEW_DB = "D:\\database faces\\NEW_DB"

CSV_IMDB_WIKI = os.path.join("csv_labels", "gt_imdb_wiki.csv")