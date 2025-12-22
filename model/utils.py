### ENCODING VOCAB INTO INTEGERS ###

vocab = ["1", "2"]

stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}


def encode(text):
    """simple encoder"""
    return [stoi[x] for x in text]


def decode(code):
    """decoder for simple encoder"""
    return [itos[x] for x in code]
