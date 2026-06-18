import pandas as pd
import numpy as np
import torch
import os

from datasets import Dataset

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

# ==========================================
# CONFIG
# ==========================================

MODEL_NAME = "bert-base-uncased"

DATA_PATH = "datasets/cleaned_sentiment140.csv"

MAX_LENGTH = 128

BATCH_SIZE = 8

EPOCHS = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# LOAD DATA
# ==========================================

# ==========================================
# LOAD ARTIFACTS
# ==========================================

print("Loading train/test artifacts...")

train_df = pd.read_csv(
    "artifacts/train.csv"
)

test_df = pd.read_csv(
    "artifacts/test.csv"
)

# Optional faster training

train_df = train_df.sample(
    n=50000,
    random_state=42
)

test_df = test_df.sample(
    n=10000,
    random_state=42
)

train_df = train_df[
    ["clean_text", "sentiment"]
]

test_df = test_df[
    ["clean_text", "sentiment"]
]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================



# ==========================================
# HUGGINGFACE DATASETS
# ==========================================

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# ==========================================
# TOKENIZER
# ==========================================

tokenizer = BertTokenizer.from_pretrained(
    MODEL_NAME
)

def tokenize(batch):

    return tokenizer(
        batch["clean_text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

train_dataset = train_dataset.map(
    tokenize,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize,
    batched=True
)

# ==========================================
# FORMAT DATASET
# ==========================================

train_dataset = train_dataset.rename_column(
    "sentiment",
    "labels"
)

test_dataset = test_dataset.rename_column(
    "sentiment",
    "labels"
)

train_dataset.set_format(
    "torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels"
    ]
)

test_dataset.set_format(
    "torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels"
    ]
)

# ==========================================
# MODEL
# ==========================================

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# ==========================================
# METRICS
# ==========================================

def compute_metrics(pred):

    labels = pred.label_ids

    preds = pred.predictions.argmax(-1)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            preds,
            average="binary"
        )
    )

    acc = accuracy_score(
        labels,
        preds
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ==========================================
# TRAINING ARGUMENTS
# ==========================================

training_args = TrainingArguments(
    output_dir="saved_models/bert",

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    num_train_epochs=EPOCHS,

    weight_decay=0.01,

    logging_dir="logs",

    load_best_model_at_end=True
)

# ==========================================
# TRAINER
# ==========================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# ==========================================
# TRAIN
# ==========================================

print("Training BERT...")

trainer.train()

# ==========================================
# EVALUATE
# ==========================================

results = trainer.evaluate()

print("\nEvaluation Results")

for key, value in results.items():
    print(f"{key}: {value}")

# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs(
    "saved_models",
    exist_ok=True
)

trainer.save_model(
    "saved_models/bert"
)

tokenizer.save_pretrained(
    "saved_models/bert"
)

print("\nBERT model saved.")