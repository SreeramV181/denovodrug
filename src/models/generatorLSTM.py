import torch
import math
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from rdkit import Chem
from src.utils import *

class LSTMGenerator(nn.Module):
    def __init__(self, vocabulary, vocab_size, embedding_dim):
        super(LSTMGenerator, self).__init__()
        self.embedding = nn.Linear(in_features=vocab_size, out_features=embedding_dim, bias=False)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=512, num_layers=2)
        self.l1 = nn.Linear(in_features=512, out_features=vocab_size)
        self.vocabulary = vocabulary

    def forward(self, x):
        embedded = self.embedding(x)
        encoded, (H,C) = self.lstm(embedded)
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
