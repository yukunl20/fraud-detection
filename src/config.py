"""Project-wide constants for the fraud detection pipeline."""

RANDOM_STATE = 42
TARGET = "Class"
EXCLUDE_FEATURES = ("Hour",)

DEFAULT_DATA_DIR = "./data"
DEFAULT_DATA_FILE = "creditcard.csv"

# Stratified split ratios: 60% train / 20% val / 20% test
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# Time-based split uses the same ratio by index position
TIME_TRAIN_RATIO = 0.6
TIME_VAL_RATIO = 0.2

DEFAULT_RECALL_TARGET = 0.80
