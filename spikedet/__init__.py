import multiprocessing
import os
from pathlib import Path


NUM_CORES = multiprocessing.cpu_count()

ROOT_DIR = Path(__file__).parents[0].parents[0]

DATA_DIR = ROOT_DIR / "data"

RANGED_DATASET_DIR = DATA_DIR / "original_dataset"
#TRAIN_DATA_PATH = DATA_DIR / "covid19_refined.csv"
#TRAIN_DATA_PATH = DATA_DIR / "covid19_filtered.csv"
#TRAIN_DATA_PATH = DATA_DIR / "covid19_merged.csv"
FOLDS_TRAIN_DATA_PATH = DATA_DIR / "covid19_merged.csv"

TRAIN_DATA_PATH = DATA_DIR / "train.csv"
VAL_DATA_PATH = DATA_DIR / "val.csv"

CHECKPOINTS_DIR: str = str(DATA_DIR / "checkpoints")  # type: ignore
LOGS_DIR: str = str(DATA_DIR / "logs")  # type: ignore

CARDIO_RR_MEAN = 680.
CARDIO_RR_SCALE = 380.
