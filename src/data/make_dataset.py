import torch
import pandas as pd
import math
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

def load_predictor_data(csv_file, input, target, minibatch_size, device, train_percentage):
    #Extract list of smile strings from data
    data = pd.read_csv(csv_file)
    smiles = data[input].values

    #Pad text
    tokenizer = Tokenizer(char_level=True, lower=False)
    tokenizer.fit_on_texts(smiles)
    seq_of_ints = tokenizer.texts_to_sequences(smiles)
    seq_of_ints = torch.LongTensor(pad_sequences(seq_of_ints, padding='post', maxlen=75)).to(device)
    print(tokenizer.word_counts)

    #One hot encode
    X = F.one_hot(seq_of_ints)[:, :, 1:]
    NUM_CHARS = X.shape[2]

    #Extract logP targets, max and minimum values for scaling output
    y = torch.FloatTensor(data[target].values)
    max_val = math.ceil(torch.max(y))
    min_val = math.floor(torch.min(y))

    #Calculate lengths
    len_of_data = len(smiles)
    train_size = int(train_percentage * len_of_data)
    test_size = len_of_data - train_size

    #Load dataset into DataLoader
    dataset = TensorDataset(X, y)
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    final_train = DataLoader(train_data, batch_size=minibatch_size, shuffle=True)
    final_test = DataLoader(test_data, batch_size=minibatch_size, shuffle=True)

    return final_train, final_test, NUM_CHARS, max_val, min_val, tokenizer

def load_generator_data(csv_file, input, minibatch_size, device, train_percentage):
    #Extract list of smile strings from data
    data = pd.read_csv(csv_file)
    smiles = data[input].values
    print("Data loaded")

    #Tokenize text
    tokenizer = Tokenizer(char_level=True, lower=False)
    tokenizer.fit_on_texts(smiles)
    seq_of_ints = tokenizer.texts_to_sequences(smiles)
    print("Text tokenized")

    #Add beginning and end of sequence tokens
    MAX_WORD = len(tokenizer.word_counts)
    MAX_LEN = 0
    MAX_INDEX = 0
    for i in range(len(seq_of_ints)):
        seq_of_ints[i] = [MAX_WORD + 1] + seq_of_ints[i] + [MAX_WORD + 2]
        if len(seq_of_ints[i]) > MAX_LEN:
            MAX_INDEX = i
            MAX_LEN = len(seq_of_ints[i])

    #Create weighting for output classes (1/total count); Therefore, outputs with lower frequency have higher weighted losses
    reverse_indices = {v:k for k,v in tokenizer.word_index.items()}
    weights = torch.zeros(MAX_WORD + 2).to(device)
    for i in range(1, MAX_WORD + 1):
        total_count = float(tokenizer.word_counts[reverse_indices[i]])
        weights[i - 1] = 1.0 / total_count
    weights[MAX_WORD:] = 1.0 / (len(smiles))

    #Add padding to ensure all sequences are of same length; set y to next character as targets
    seq_of_ints = torch.LongTensor(pad_sequences(seq_of_ints, padding='post', maxlen=MAX_LEN)).to(device)
    X = seq_of_ints[:, :-1]
    y = seq_of_ints[:, 1:]

    #One hot encode sequences; set X to appropriate inputs
    final_data = F.one_hot(seq_of_ints)
    X = final_data[:, :MAX_LEN - 1, 1:]
    NUM_CHARS = X.shape[2]

    #Calculate lengths
    len_of_data = len(smiles)
    train_size = int(train_percentage * len_of_data)
    test_size = len_of_data - train_size

    #Load dataset into DataLoader
    dataset = TensorDataset(X, y)
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    final_train = DataLoader(train_data, batch_size=minibatch_size, shuffle=True)
    final_test = DataLoader(test_data, batch_size=minibatch_size, shuffle=True)

    return final_train, final_test, NUM_CHARS, MAX_LEN, weights, tokenizer
