from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_ROOT = DATA_DIR / "train"
EVAL_ROOT = DATA_DIR / "eval"
LABEL_VOCAB_PATH = DATA_DIR / "label_vocab.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Training
BATCH_SIZE = 32
NUM_WORKERS = 0
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LOG_EVERY_STEPS = 50

# Model: base, multiloss, parallel_heads, sequential_heads, or moe
MODEL_ARCHITECTURE = "base"
BASE_CHANNELS = 16
DROPOUT = 0.1
NUM_EXPERTS = 8
GATE_TEMPERATURE = 1.0

# Multi-task labels and loss weights
CATEGORICAL_LABELS = ["country", "climate", "land_cover"]
COORD_LOSS_WEIGHT = 1.0
DIST_SEA_LOSS_WEIGHT = 0.25
COUNTRY_LOSS_WEIGHT = 0.25
CLIMATE_LOSS_WEIGHT = 0.25
LAND_COVER_LOSS_WEIGHT = 0.25
EXPERT_DIVERSITY_LOSS_WEIGHT = 0.02
EXPERT_LOAD_BALANCE_LOSS_WEIGHT = 0.02

# Checkpoints
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"
LAST_MODEL_PATH = CHECKPOINT_DIR / "last_model.pt"
