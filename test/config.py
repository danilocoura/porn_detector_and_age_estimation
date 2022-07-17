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

PIXEL_MEAN = [0.485, 0.456, 0.406]  # Mean values for RGB channels. Default [0.485, 0.456, 0.406] is for IMAGENET
PIXEL_STD = [0.229, 0.224, 0.225]  # Std dev values for RGB channels. Default [0.229, 0.224, 0.225] is for IMAGENET
NUM_WORKERS = 1  # Number of subprocesses simultaneously loading data from the system

DATA_DIR = os.path.join("database")
DIR_AGEDB = "D:\\database faces\\AgeDB\\AgeDB"
DIR_UTKFACE = "D:\\database faces\\UTKFace_full\\part1"
DIR_FGNET = "D:\\database faces\\FGNET\\FGNET\\images"
DIR_APPA_REAL = "D:\\database faces\\APPA-REAL\\appa-real-release"
DIR_MEDS2 = "D:\\database faces\\MEDS-II"
DIR_NEW_DB = "D:\\database faces\\NEW_DB_MTCNN_BIGFACE_CONFIDENCE_0.89"
CSV_A1 = "A1.csv"
CSV_A2 = "A2.csv"
CSV_A3 = "A3.csv"
CSV_FINAL = "FINAL.csv"
BATCH_SIZE = 4
LIM_INF = 18
LIM_SUP = 18