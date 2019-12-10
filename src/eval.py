import torch
import pickle
from molvs import validate_smiles
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.nn import functional as F
from rdkit import Chem
from utils import *
from models import Predictor, Generator

def generateSmiles(model, num, vocabulary, max_len, device):
    smiles = []

    #Create mask to prevent attending to future positions
    mask = create_mask(max_len - 1, device)

    for _ in range(num):
        #Create initial token
        curr = torch.zeros((max_len - 1, 1, len(vocabulary) + 2)).to(device)
        curr[0][0][len(vocabulary)] = 1
        current_smile = ""

        for i in range(1, max_len - 1):
            output = torch.softmax(model(curr, m=mask), dim=2)[i:i+1, 0, :].squeeze().to(device)
            prob = torch.distributions.Categorical(output)
            sample = prob.sample().item()
            while sample == len(vocabulary):
                sample = prob.sample().item()
            if sample == len(vocabulary) + 1:
                break
            #Update current smile string and curr tensor
            current_smile += vocabulary[sample + 1]
            curr[i][0][sample] = 1
        smiles.append(current_smile)

    return smiles

def evaluateLogP(model, smiles, vocab, device):
    logP = []

    base = ""
    for v in vocab.values():
        base += v
    smiles.insert(0, base)

    #Unpack pickle
    processing = pickle.load(open('processing.pkl', 'rb'))
    tokenizer = processing['tokenizer']
    max = processing['max']
    min = processing['min']

    seq_of_ints = tokenizer.texts_to_sequences(smiles)
    seq_of_ints = torch.LongTensor(pad_sequences(seq_of_ints, padding='post', maxlen=75)).to(device)

    #One hot encode
    X = F.one_hot(seq_of_ints)[1:, :, 1:].transpose(0, 1).type(torch.FloatTensor).to(device)
    logP = (max - min) * model(X) + min

    return logP.tolist()

def checkValidity(smiles):
    validSmiles = []
    numWrong = 0
    for s in smiles:
        test = Chem.MolFromSmiles(s)
        if test != None:
            validSmiles.append(s)
        else:
            numWrong += 1
    percentValid = (float(len(smiles) - numWrong)/float(len(smiles))) * 100
    print("Percentage of Valid Smiles: {}%".format(percentValid))
    return validSmiles

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #MODIFY THESE PARAMETERS
    predictorFile = "checkpoints/LogP Predictor/model-200000.pt"
    generatorFile = "checkpoints/SMILE Generator/model-400000.pt"
    vocabulary_file = "../vocab_mapping.pkl"
    NUM_SMILES_TO_GENERATE = 1000

    #Load vocabulary
    vocabulary = pickle.load(open(vocabulary_file, 'rb'))
    vocabulary = {v: k for k, v in vocabulary.items()}
    V = len(vocabulary) #Vocab size
    print("Vocab loaded")

    #Load predictor
    p = Predictor(vocab_size=V, embedding_dim=32).load_state_dict(torch.load(predictorFile)).to(device)
    p.eval()
    print("Predictor loaded")

    #Load generator
    g = Generator(vocab_size=V + 2, embedding_dim=32).load_state_dict(torch.load(generatorFile)).to(device)
    g.eval()
    print("Generator loaded")

    #Evaluate results
    smiles = generateSmiles(g, NUM_SMILES_TO_GENERATE, vocabulary, 77, device)
    validSmiles = checkValidity(smiles)
    logP = evaluateLogP(p, validSmiles, vocabulary, device=device)

    #Save results
    results = {}
    results['smiles'] = smiles
    results['validSmiles'] = validSmiles
    results['logP'] = logP
    f = open("evaluation_results.pkl","wb")
    pickle.dump(results,f)
    f.close()



if __name__ == '__main__':
    main()
