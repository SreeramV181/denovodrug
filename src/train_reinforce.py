import torch
import time
import tqdm
import math
import pickle
from torch import nn, optim
from utils import *
from torch.nn import functional as F
from models.predictor import Predictor
from models.generator import Generator
from models.generatorLSTM import LSTMGenerator
from models.reinforce import REINFORCE


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    generatorFile = "../models/checkpoints/LSTM_E32_H512/model-200000.pt"
    predictorFile = "../models/checkpoints/LogP Predictor/model-300000.pt"
    G_EMBEDDING = 32

    g_dict = torch.load(generatorFile, map_location=device)
    p_dict = torch.load(predictorFile, map_location=device)

    #Tokenizer
    G_tokenizer = g_dict['tokenizer']
    vocabulary={v:k for k,v in G_tokenizer.word_index.items()}
    V = len(vocabulary)

    #Load generator
    G = LSTMGenerator(vocabulary, vocab_size=V + 2, embedding_dim=32)
    G.load_state_dict(g_dict['model_state_dict'])
    G.to(device)

    #Load predictor
    P_tokenizer = p_dict['tokenizer']
    P = Predictor(vocab_size=V, embedding_dim=32)
    P.load_state_dict(p_dict['model_state_dict'])
    P.to(device)

    def reward(pred, smile_tensor):
        val = pred(smile_tensor)
        if val >= 1 and val <= 4:
            return 11
        return 1

    joint_model = REINFORCE(G, P, vocabulary, G_tokenizer, reward_func=reward)
    joint_model.train(minibatch_size=8, discount=.95, device=device,
                      tqdm=tqdm.tqdm, max_len=75,
                      iter_max=20000, iter_save=2000,
                      loss_save=100, save_name="joint_PARAMS")

if __name__ == '__main__':
    main()
