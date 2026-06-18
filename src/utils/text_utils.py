import pickle

with open(
    "artifacts/vocab.pkl",
    "rb"
) as f:

    VOCAB = pickle.load(f)

MAX_LEN = 100

def encode_text(text):

    tokens = text.lower().split()

    encoded = [
        VOCAB.get(token, 1)
        for token in tokens
    ]

    encoded = encoded[:MAX_LEN]

    encoded += [0] * (
        MAX_LEN - len(encoded)
    )

    return encoded