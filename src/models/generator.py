import torch
import math
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from rdkit import Chem
from src.models.positionalencoding import PositionalEncoding
from src.utils import *

class Generator(nn.Module):
    def __init__(self, vocabulary, vocab_size=50, embedding_dim=64, max_len=75):
        super(Generator, self).__init__()
        self.embedding = nn.Linear(in_features=vocab_size, out_features=embedding_dim, bias=False)
        self.position = PositionalEncoding(embedding_dim)
        self.encoderLayer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dim_feedforward=4 * embedding_dim)
        self.encoder = nn.TransformerEncoder(self.encoderLayer, num_layers=4)
        self.l1 = nn.Linear(in_features=embedding_dim, out_features=vocab_size)

        self.max_len = max_len
        self.vocabulary = vocabulary

    def forward(self, x, m):
        embedded = self.embedding(x)
        positioned = self.position(embedded)
        encoded = self.encoder(positioned, mask=m)
        final = self.l1(encoded)
        return final

    def generateValidSmiles(self, minibatch_size=1, device='cpu'):
        while True:
            #Create string to return
            current_smile = ""

            #Create mask to prevent attending to future positions
            mask = create_mask(self.max_len - 1, device)

            #Create initial token
            curr = torch.zeros((self.max_len - 1, minibatch_size, len(self.vocabulary) + 2)).to(device)
            curr[0][:][len(self.vocabulary)] = 1

            for i in range(1, max_len - 1):
                logits = self.forward(curr, m=mask)
                output = torch.softmax(logits, dim=2)[i, 0, :].squeeze().to(device)

                #Create and sample from distribution of probabilities
                prob = torch.distributions.Categorical(output)
                sample = prob.sample().item()
                while sample == len(self.vocabulary):
                    sample = prob.sample().item()

                #Update curr, end generation if token is ending token, o.w. append character to end of current_smile
                curr[i][:][sample] = 1
                if sample == len(self.vocabulary) + 1:
                    break
                else:
                    current_smile += self.vocabulary[sample + 1]

            test = Chem.MolFromSmiles(current_smile)
            if test != None:
                #Return tuple of (tensor, string) so other methods can use whichever representation they want
                return (curr, current_smile)
