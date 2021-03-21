# CS 224W Final Project: FLAG + DropEdge

This CS 224W project seeked to improve upon the FLAG baseline by augmenting the training data with structural features. Specifically,
we decided to add on DropEdge and a variety of modifications to see if dropping a certain percentage of training edges during training time
improve model performance.

## Running the Code

All of the necessary code is contained in the jupyter notebook. If running in Google Colab, ensure that runtime type is switched to "GPU".

## Hyperparamter Explanations and Defaults

Below is an explanation of each of the hyperparameters, along with their default values.

```bash
device: device on which to run the model (type: int, default: 0)

use_sage: whether model used should be GraphSAGE; falses uses GCN (type: bool, default: True)

num_layers: total number of layers in graph neural network model (type: int, default: 3)

hidden_channels: hidden dimensionality of graph model (type: int, default: 256)

dropout: dropout percentage for dropout layer (type: float, default: .5)

lr: learning rate (type: float, default: .01)

epochs: number of times to go through the training set on each run (type: int, default: 500)

runs: number of times to test model (type: int, default: 5)

start_seed: starting randomization seed (type: int, default: 0)

step_size: alpha parameter corresponding to step size update used in FLAG paper (type: float, default: 1e-3)

m: number of ascent steps in FLAG (type: int, default: 3)

test_freq: how frequently to run model on validation and test sets (after half of the epochs have passed (type: int, default: 1)

attack: type of training method used (MUST BE EITHER 'flag' or 'vanilla') (type: str, default: 'flag')

percent_keep: what percentage of edges to keep in the adjacency matrix in drop edge (type: float, default: .7)

epsilon: approximate percentage of time we want to use the entire adjacency matrix in training instead of dropping edges (type: float, default: 0)
```

The hyperparameters can be modified within the args dictionary under the Main heading in the notebook.


## Citations

[FLAG](https://arxiv.org/pdf/2010.09891.pdf)

[DropEdge](https://openreview.net/forum?id=Hkx1qkrKPr)

[Baseline Skeleton Code](https://github.com/devnkong/FLAG/blob/main/ogb/nodeproppred/arxiv/gnn.py)


## Contributions
This project was completed by Sreeram Venkatarao (sreeramv) and Anirudh Prabhu (aprabhu3)
