import torch
import torch.nn as nn


class LSTM_fixed_len(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x, l):
        x = self.embeddings(x)
        lstm_out, (ht, ct) = self.lstm(x)
        return self.linear(ht[-1])
