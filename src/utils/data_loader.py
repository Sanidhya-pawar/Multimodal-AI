import pandas as pd

def load_train():

    return pd.read_csv(
        "artifacts/train.csv"
    )

def load_test():

    return pd.read_csv(
        "artifacts/test.csv"
    )