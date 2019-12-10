import torch
import time
import tqdm
import pandas as pd
import math
import pickle
from data.make_dataset import load_predictor_data
from models.predictor import Predictor
from utils import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader


def train(model, train_loader, tokenizer, device, tqdm, iter_max, iter_save, loss_save, save_name):
    #Model setup
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=5e-6)
    criterion = nn.MSELoss()
    total_loss = 0
    losses = []
    i = 0

    #Keep track of time
    start_time = time.time()
    with tqdm(total=iter_max) as pbar:
        while True:
            for x, y in train_loader:
                i += 1 # i is num of gradient steps taken by end of loop iteration

                #Extract data from dataloader
                x, y = x.transpose(0, 1).type(torch.FloatTensor).to(device), y.to(device)


                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output.to(device), y)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

                #Update progress bar
                pbar.set_postfix(loss='{:.2e}'.format(loss.item()))
                pbar.update(1)

                # Save model
                if i % iter_save == 0:
                    print("Saving model")
                    save_losses_by_name(losses, save_name, i)
                    save_model_by_name(model, optimizer, tokenizer, save_name, i)
                if i % loss_save == 0:
                    losses.append(loss.item())
                if i == iter_max:
                    return

def main():
    d = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATA_FILEPATH ="../data/processed/logp_data.csv"
    INPUT = 'smiles'
    TARGET = 'logp'
    MINIBATCH = 128

    training_data, testing_data, NUM_CHARS, MAX_VAL, MIN_VAL, T = load_predictor_data(csv_file=DATA_FILEPATH,
                                                       input=INPUT,
                                                       target=TARGET,
                                                       minibatch_size=MINIBATCH,
                                                       device=d,
                                                       train_percentage=.8)
    P = Predictor(vocab_size=NUM_CHARS, embedding_dim=32, minimum=MIN_VAL, maximum=MAX_VAL).to(d)
    train(model=P,
          train_loader=training_data,
          tokenizer=T,
          device=d,
          tqdm=tqdm.tqdm,
          iter_max=1000000,
          iter_save=100000,
          loss_save=1000,
          save_name="LogP Predictor")

if __name__ == '__main__':
    main()
