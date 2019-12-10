import torch
import math
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from src.models.positionalencoding import PositionalEncoding

class Predictor(nn.Module):
    def __init__(self, vocab_size=50, embedding_dim=64, minimum=-5, maximum=5):
        super(Predictor, self).__init__()
        self.embedding = nn.Linear(vocab_size, embedding_dim, bias=False)
        self.position = PositionalEncoding(embedding_dim)
        self.encoderLayer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dim_feedforward=4 * embedding_dim)
        self.encoder = nn.TransformerEncoder(self.encoderLayer, num_layers=4)
        self.l1 = nn.Linear(in_features=embedding_dim, out_features=1)
        self.l2 = nn.Linear(in_features=75, out_features=1)

        self.min = minimum
        self.max = maximum

    def forward(self, x):
        embedded = self.embedding(x)
        positioned = self.position(embedded)
        encoded = self.encoder(positioned).transpose(0, 1)
        final_values = self.l1(encoded).squeeze(2)
        final = self.l2(final_values).squeeze(1)
        return ((self.max - self.min) * torch.sigmoid(final) + self.min)
