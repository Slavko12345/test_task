import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split
from shared_utils import tokenize, encode_sentence


def preprocess_csv(path):
    table = pd.read_csv(path)
    table = table[['excerpt', 'target']]
    table['excerpt_length'] = table['excerpt'].apply(lambda x: len(x.split()))
    return table


def prepare_vocab(table):
    counts = Counter()
    for index, row in table.iterrows():
        counts.update(tokenize(row['excerpt']))
    for word in list(counts):
        if counts[word] < 2:
            del counts[word]

    vocab2index = {"": 0, "UNK": 1}
    words = ["", "UNK"]
    for word in counts:
        vocab2index[word] = len(words)
        words.append(word)
    return vocab2index, words


class ReviewsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx], self.X[idx][1]


def get_train_val_dataloaders():
    train_table = preprocess_csv("data/train.csv")
    vocab2index, words = prepare_vocab(train_table)

    train_table['encoded'] = train_table['excerpt'].apply(lambda x: np.array(encode_sentence(x, vocab2index)))

    X, y = list(train_table['encoded']), list(train_table['target'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    train_ds = ReviewsDataset(X_train, y_train)
    test_ds = ReviewsDataset(X_test, y_test)

    batch_size = 5000
    vocab_size = len(words)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    return train_dl, test_dl, vocab2index, vocab_size


def validation_metrics(model, valid_dl):
    model.eval()
    total = 0
    sum_loss = 0.0
    for x, y, l in valid_dl:
        x = x.long()
        y = y.float()
        y_hat = model(x, l)[:, 0]
        loss = F.mse_loss(y_hat, y)
        total += y.shape[0]
        sum_loss += loss.item() * y.shape[0]
    return sum_loss / total


def train_model(model, train_dl, test_dl, epochs=10, lr=0.001):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for x, y, l in train_dl:
            x = x.long()
            y = y.float()
            y_pred = model(x, l)[:, 0]
            optimizer.zero_grad()
            loss = F.mse_loss(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item() * y.shape[0]
            total += y.shape[0]
        val_loss = validation_metrics(model, test_dl)
        if i % 5 == 4:
            print(f"epoch: {i+1}; train rmse: {sum_loss / total}; val rmse {val_loss}")
