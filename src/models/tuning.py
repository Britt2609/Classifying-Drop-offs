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

from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt.pyll import scope

# TODO: put back es, dropout

def data():
    data_dir = 'data/processed/ws_16_step_9/processed_numerical_ws_16_step_9'
    all_data = np.load(f'{data_dir}.npz', allow_pickle = True)

    x_train_pre = np.asarray(all_data['x_train']).astype('float32')
    y_train_pre = np.asarray(all_data['y_train']).astype('int32')

    split = int(y_train_pre.shape[0]*0.7)

    x_val = x_train_pre[split:, :, :]
    y_val = y_train_pre[split:]

    x_train = x_train_pre[:split, :, :]
    y_train = y_train_pre[:split]
    return x_train, y_train, x_val, y_val

global x_train
global y_train
global x_val 
global y_val

x_train, y_train, x_val, y_val = data()



def main(params):

    print ('Params testing: ', params)
    input_filepath = 'data/processed models'
    logger = logging.getLogger(__name__)
    window_size = 16
    step_config = 0.5
    step = int(window_size*step_config + 1)
    model_name = f'ws_{window_size}_step_{step}'
    data_dir = 'data/processed/ws_16_step_9/processed_numerical_ws_16_step_9'
    
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    class_weights = dict(enumerate(class_weights))
    adam = Adam(learning_rate=params['lr'], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model = Sequential()
    if params['LSTM_config']['layers'] == 'two':
        model.add(LSTM(params['units1'], return_sequences=True, input_shape=(window_size, 4)))
        model.add(LSTM(params['LSTM_config']['units2']))
    else:
        model.add(LSTM(params['units1'], input_shape=(window_size, 4))) 

    model.add(Dropout(params['dropout']))
    model.add(Dense(1))
    model.add(Activation('sigmoid')) 

    logger.info("started training")
    # , 
    # PR because class imbalance, anything with a probability greater than 0.35 of being positive, we'll classify as positive. 
    model.compile(loss='binary_crossentropy',
                optimizer=adam, metrics = [TruePositives(), FalsePositives(), TrueNegatives(), FalseNegatives(), AUC(name='PR', curve = "PR", thresholds = [0.35])])

    #es = EarlyStopping(monitor='val_loss', mode='min', verbose = 1, patience = 3)
    history = model.fit(x_train, y_train,  batch_size=params['batch_size'], epochs=num_epochs, validation_data=(x_val, y_val), class_weight=class_weights, verbose = 2)
    
    validation_PR_max = np.amax(history.history['val_PR']) 
    validation_PR = history.history['val_PR']
    logger.info(f'Last valisation acc: {validation_PR}')
    logger.info(f'Best validation acc of epoch: {validation_PR_max}')
    return {'loss': -validation_PR_max, 'status': STATUS_OK}

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt, filename = "logs/tuning.log")

    project_dir = Path(__file__).parents[2]
    sys.path.append(project_dir)
    from src.variables_and_methods import *
    load_dotenv(find_dotenv())
    space = {
        'LSTM_config': hp.choice('num_layers', [
            {'layers': 'one'},
            {'layers': 'two', 'units2': hp.choice('units2', np.arange(32, 256, dtype =int))}
        ]),
        'units1': hp.choice('units1', np.arange(32, 256, dtype=int)),
        'dropout': hp.uniform('dropout', 0.5, 0.75),
        'lr': hp.uniform('lr', 0.0001, 0.01),
        'batch_size': hp.choice('batch_size', [2048, 4096])
    }
    
    best_run = fmin(main, space, algo=tpe.suggest, max_evals=15, trials=Trials())

    logger = logging.getLogger(__name__)
    
    # print("Evalutation of best performing model:")
    # print(best_model.evaluate(x_test, y_test))
    # logger.info("Evalutation of best performing model:")
    # logger.info(best_model.evaluate(x_test, y_test))
    logger.info("Best performing model chosen hyper-parameters:")
    logger.info(best_run)

