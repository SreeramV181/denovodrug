# denovodrug

This library was created for the generation of novel molecules using Transformer-based architectures and reinforcement learning.

## src
src/models is the primary folder containing the code detailing the deep generative model architectures. src also contains files for training a generator and predictor as well as a file for the biasing training of a generator. src/data contains functions for reading datasets into memory for training. An additional utils file defines functions for saving models.


## notebooks
notebooks contains Jupyter Notebook files used for quickplotting of the losses of various models as well as a notebook to conduct evaluations of the different models. These evaluations include how many generated SMILE strings are valid, what their log P values are, and more.

## models
models contains checkpoints for the training of different models. These checkpoints contain the models themselves, the optmizer used, and the corresponding tokenizer.

## data
data contains the raw and processed data used in this project.
