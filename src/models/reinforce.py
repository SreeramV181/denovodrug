import time
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from utils import *

class REINFORCE():
    def __init__(self, generator, predictor, vocabulary, tokenizer, reward_func):
        super(REINFORCE, self).__init__()
        self.G = generator
        self.P = predictor
        self.vocab = vocabulary
        self.tokenizer = tokenizer
        self.reward = reward_func

    def train(self, minibatch_size, discount, device, tqdm,
              max_len, iter_max, iter_save, loss_save, save_name):
        losses = []

        self.G.train()
        self.P.eval()
        optimizer = optim.AdamW(self.G.parameters(), lr=2.5e-6)

        #Keep track of time
        start_time = time.time()
        with tqdm(total=iter_max) as pbar:
            while True:
                for i in range(1, iter_max+1):
                    loss = 0
                    optimizer.zero_grad()

                    for _ in range(minibatch_size):
                        smile_tensor, smile = self.generateValidSmiles(max_len=max_len, device='cuda')
                        modified = torch.cat((torch.zeros(1, 1, 38).to(device), smile_tensor), 0)
                        modified = modified[:, :, :36]
                        R = self.reward(self.P, modified)

                        discounted_R = R

                        for j in range(1, len(smile) + 1):
                            log_probs = F.log_softmax(self.G(smile_tensor[:j][:][:]), dim=2)
                            loss -= torch.sum(log_probs * smile_tensor[j][:][:]) * discounted_R
                            discounted_R *= discount

                    loss /= minibatch_size
                    loss.backward()
                    optimizer.step()

                    #Update progress bar
                    pbar.set_postfix(loss='{:.2e}'.format(loss.item()))
                    pbar.update(1)

                    # Save model
                    if i % iter_save == 0:
                        #print("Saving model at iteration {}".format(i))
                        save_losses_by_name(losses, save_name, i)
                        save_joint_by_name(self.G, self.P, optimizer, self.tokenizer, save_name, i)
                    if i % loss_save == 0:
                        losses.append(loss.item())
                    if i == iter_max:
                        return

    def generateValidSmiles(self, max_len, device='cpu'):
        while True:
            #Create string to return
            current_smile = ""

            #Create initial token
            curr = torch.zeros((max_len - 1, 1, len(self.G.vocabulary) + 2)).to(device)
            curr[0][0][len(self.G.vocabulary)] = 1

            for i in range(1, max_len - 1):
                logits = self.G(curr)
                output = torch.softmax(logits, dim=2)[i, 0, :].squeeze().to(device)

                #Create and sample from distribution of probabilities
                prob = torch.distributions.Categorical(output)
                sample = prob.sample().item()
                while sample == len(self.G.vocabulary):
                    sample = prob.sample().item()

                #Update curr, end generation if token is ending token, o.w. append character to end of current_smile
                curr[i][0][sample] = 1
                if sample == len(self.G.vocabulary) + 1:
                    break
                else:
                    current_smile += self.G.vocabulary[sample + 1]

            return (curr, current_smile)


    def eval(self, num_samples):
        print('evaluating')
