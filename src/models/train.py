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
from tensorflow.keras.metrics import TruePositives, TrueNegatives, FalsePositives, FalseNegatives, Recall, AUC, Precision
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
    logger = logging.getLogger(__name__)
    step = int(window_size*step_config + 1)
    model_name = f'ws_{window_size}_step_{step}'
    data_dir = f'{input_filepath}/{model_name}/processed_numerical_{model_name}'
    all_data = np.load(f'{data_dir}.npz', allow_pickle = True)
    
    x_train = np.asarray(all_data['x_train']).astype('float32')
    y_train = np.asarray(all_data['y_train']).astype('int32')


    split = int(y_train.shape[0]*fraction_validate)

    x_val = x_train[split:, :, :]
    y_val = y_train[split:]

    x_train = x_train[:split, :, :]
    y_train = y_train[:split]

    # weigh class weights for class imbalance
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    class_weights = dict(enumerate(class_weights))

    datestring = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M')

    model = Sequential()
    elements = lstm_config.split('_')
   
    if elements[0] == '1L':
        model.add(LSTM(int(elements[1]),input_shape=(window_size, len(dynamic_idx))))
    elif elements[0] == '2L':
        model.add(LSTM(int(elements[1]), return_sequences=True, input_shape=(window_size, len(dynamic_idx))))
    else:
        raise Exception(f"Invalid LSTM coniguration : {elements}")
    
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))

    # Custom adam optimizer
    adam = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    logger.info("started training")
    # Compile Model, with custom loss and optimizer
    # miss_rate_keras, fbeta_keras, fall_out_keras
    model.compile(loss='binary_crossentropy',
                optimizer=adam, metrics = [TruePositives(name="TP_Score"), FalsePositives(), TrueNegatives(), FalseNegatives(), AUC(name='AUC_PR', curve = "PR", thresholds = [0.35]), Recall(name='recall'), Precision(name='precision')])

    # Plot the Model to Image File
    # keras.utils.plot_model(model, to_file='..\..\models\danish_model_' + ACTIVATION_LAYER + '_' + str(num_epochs) + 'ep_' + datestring + '.png', show_shapes=True)
     # for saving files
    # early stopping
    es = EarlyStopping(monitor='val_AUC_PR', mode='min', verbose=1, patience=10)
    
    # Fit the Model
    history = model.fit(x_train, y_train, class_weight=class_weights, batch_size=batch_size, epochs=num_epochs, validation_data=(x_val, y_val), verbose = 2)
    datestring = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M')
    
    # Serialize Model to JSON
    model_json = model.to_json()
   
    # make new directory for setup if necessary
    model_dir_name = lstm_config + f'_ws_{str(window_size)}' +  f'_step_{str(step)}' + f'_lr_{str(lr)}' + f'_bs_{str(batch_size)}'
    model_dir = output_filepath_models + '/' + model_dir_name
    
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    # save in right folder with the date
    file_name_model =  model_dir + '/' + model_dir_name + '_' +  datestring 
   
    with open(file_name_model + '.json', 'w') as json_file:
        json_file.write(model_json)
    
    # Serialize Weights to HDF5
    model.save_weights(file_name_model + '.h5')
    logger.info(f'Saved model and weights to file location: {file_name_model}')

    validation_PR = np.amax(history.history['val_AUC_PR']) 
    logger.info(f'max validation AUC_PR : {validation_PR}')
    scores = model.evaluate(x_val, y_val, verbose = verbose)

    logger.info("Finished training")
    
    # Evaluate the Model
    
    loss, TP, FP, TN, FN, AUC_PR = scores[0],  scores[1],  scores[2],  scores[3],  scores[4], scores[5]

    recall = compute_recall(TP, FP, TN, FN)
    miss_rate = compute_miss_rate(TP, FP, TN, FN)
    fbeta = compute_f_beta(TP, FP, TN, FN)
    fall_out = compute_fall_out(TP, FP, TN, FN)

    logger.info(f"----------------------------------------------------------------------------------------------------")
    logger.info(f"LSTM configuration : {lstm_config}, Window size : {window_size}, step = {step}, learning rate = {lr}, batch size = {batch_size}, dropout = {dropout}")
    logger.info(f"TP = {TP}, FP = {FP}, TN = {TN}, FN = {FN} ")
    logger.info(f"loss = {loss}, AUC_PR = {AUC_PR}")
    logger.info(f"recall = {recall}, miss_rate = {miss_rate}, fbeta = {fbeta}, fall_out = {fall_out} ")
    logger.info(f"----------------------------------------------------------------------------------------------------")

    
    file_name_train_history = f'{output_filepath_figures}/train_history_{datestring}.csv'
    train_hist_df = pd.DataFrame(history.history) 

    with open(file_name_train_history, mode='w') as f:
        train_hist_df.to_csv(f)
        

    # Make figure of loss and metrics 
    if make_figures:
        # Loss
        plt.figure(0)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Cross Binary Entropy Loss During Training')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper right')
        plt.savefig(f'{output_filepath_figures}/{model_dir_name}_{datestring}_loss.png', bbox_inches='tight')
        
        # AUC_PR
        plt.figure(1)
        plt.plot(history.history['AUC_PR'])
        plt.plot(history.history['val_AUC_PR'])
        plt.title('AUC-PR Score During Training')
        plt.xlabel('Epoch')
        plt.ylabel('AUC-PR')
        plt.legend(['Training', 'Validation'], loc='upper right')
        plt.savefig(f'{output_filepath_figures}/{model_dir_name}_{datestring}_AUC_PR.png', bbox_inches='tight')
        
        # Recall
        plt.figure(2)
        plt.plot(history.history['recall'])
        plt.plot(history.history['val_recall'])
        plt.title('Recall Score During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend(['Training', 'Validation'], loc='upper right')
        plt.savefig(f'{output_filepath_figures}/{model_dir_name}_{datestring}_recall.png', bbox_inches='tight')
       
        # Precision
        plt.figure(3)
        plt.plot(history.history['precision'])
        plt.plot(history.history['val_precision'])
        plt.title('Precision Score During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend(['Training', 'Validation'], loc='upper right')
        plt.savefig(f'{output_filepath_figures}/{model_dir_name}_{datestring}_precision.png', bbox_inches='tight')
        
        # True Positives
        plt.figure(4)
        plt.plot(history.history['TP_Score'])
        plt.plot(history.history['val_TP_Score'])
        plt.title('Amount of Correct Triggers During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Correct triggers')
        plt.legend(['Training', 'Validation'], loc='upper right')
        plt.savefig(f'{output_filepath_figures}/{model_dir_name}_{datestring}_AUC_PR.png', bbox_inches='tight')

        logger.info('made figures and stored store in directory reports/figures')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt, filename = "logs/training.log")

    # # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).parents[2]
    sys.path.append(project_dir)
    # import all global variables and methods
    from src.variables_and_methods import *
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
