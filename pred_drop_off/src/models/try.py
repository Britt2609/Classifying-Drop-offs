# -*- coding: utf-8 -*-
"""
    File name: train.py
    Author: Maike Nützel
    Date created: 14/05/2023
    Date last modified: 
    Python Version: 3.6
"""
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from tensorflow import keras 
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LSTM, Dense, Conv1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import TruePositives, TrueNegatives, FalsePositives, FalseNegatives, Recall, AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.math import argmax
from keras.utils import np_utils
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from tensorflow.keras.callbacks import EarlyStopping

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath_models', type = click.Path())
@click.argument('output_filepath_figures', type = click.Path())
@click.argument('lstm_config', type = str)
@click.argument('lr', type = float)
@click.argument('batch_size', type = int)
@click.argument('dropout', type = float)
def main(input_filepath, output_filepath_models, output_filepath_figures, lstm_config , lr, batch_size, dropout):
    step = int(window_size*step_config + 1)
    model_name = f'ws_{window_size}_step_{step}'
    data_dir = f'{input_filepath}/{model_name}/processed_numerical_{model_name}'
    all_data = np.load(f'{data_dir}.npz', allow_pickle = True)
    
    x_train = np.asarray(all_data['x_train']).astype('float32')
    y_train = np.asarray(all_data['y_train']).astype('int32')
   
    # weigh class weights for class imbalance
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    class_weights = dict(enumerate(class_weights))

if __name__ == '__main__':

    # # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).parents[2]
    sys.path.append(project_dir)
    # import all global variables and methods
    from src.variables_and_methods import *
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
