import json

CONFIG_PATH = "artifacts/config.json"

with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

VOCAB_SIZE = CONFIG["vocab_size"]
MAX_LENGTH = CONFIG["max_length"]
EMBEDDING_DIM = CONFIG["embedding_dim"]
HIDDEN_DIM = CONFIG["hidden_dim"]
RANDOM_STATE = CONFIG["random_state"]