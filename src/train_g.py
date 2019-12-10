import torch
import time
import tqdm
import math
import pickle
from data.make_dataset import load_generator_data
from torch import nn, optim
from utils import *
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from models.generator import Generator
from models.generatorLSTM import LSTMGenerator


def train(model, train_loader, device, tqdm, max_len, class_weights, tokenizer, iter_max, iter_save, loss_save, save_name):
    #Model setup
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=-1) #weight=class_weights
    total_loss = 0
    losses = []
    i = 0

    #Create mask to prevent attending to future positions
    mask = create_mask(max_len - 1, device)

    #Keep track of time
    start_time = time.time()
    with tqdm(total=iter_max) as pbar:
        while True:
            for x, y in train_loader:
                x, y = x.type(torch.FloatTensor).transpose(0, 1).to(device), y.to(device)

                i += 1 # i is num of gradient steps taken by end of loop iteration
                optimizer.zero_grad()
                if type(model) is LSTMGenerator:
                    output = model(x)
                else:
                    output = model(x, m=mask)
                loss = criterion(output.permute(1, 2, 0).to(device), y - 1)
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
    G_TYPE = 'LSTM'
    DATA_FILEPATH ="../data/processed/logp_data.csv"
    INPUT = 'smiles'
    MINIBATCH = 128

    training_data, testing_data, NUM_CHARS, MAX_LEN, W, T = load_generator_data(csv_file=DATA_FILEPATH,
                                                       input=INPUT,
                                                       minibatch_size=MINIBATCH,
                                                       device=d,
                                                       train_percentage=.95)
    vocab = {v:k for k,v in T.word_index.items()}
    if G_TYPE == 'LSTM':
        G = LSTMGenerator(vocabulary=vocab, vocab_size=NUM_CHARS, embedding_dim=32).to(d)
    else:
        G = Generator(vocabulary=vocab, vocab_size=NUM_CHARS, embedding_dim=32, max_len=MAX_LEN).to(d)
    train(model=G,
          train_loader=training_data,
          device=d,
          tqdm=tqdm.tqdm,
          max_len=MAX_LEN,
          class_weights=W,
          tokenizer=T,
          iter_max=1000000,
          iter_save=100000,
          loss_save=1000,
          save_name="LSTM_E32_H512")

if __name__ == '__main__':
    main()
