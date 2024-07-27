
# -*- coding: utf-8 -*-
"""
    File name: test_model.py
    Author: Maike Nützel
    Date created: 14/05/2023
    Date last modified: 
    Python Version: 3.9
"""
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime
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
from sklearn.utils import class_weight
import sys
import matplotlib.pyplot as plt
import glob
import os
import csv
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_absolute_error, mean_squared_error

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('input_filepath_models', type = click.Path())
@click.argument('output_filepath_figures', type = click.Path())
@click.argument('output_filepath_results', type = click.Path())
@click.argument('lstm_config', type = str)
@click.argument('lr', type = float)
@click.argument('batch_size', type = int)
@click.argument('dropout', type = float)
def main(input_filepath, input_filepath_models, output_filepath_figures, output_filepath_results, lstm_config, lr, batch_size, dropout):
    step = int(window_size*step_config + 1)
    model_dir = input_filepath_models + '/' + lstm_config + f'_ws_{str(window_size)}_' +  f'_step_{str(step)}_' + f'_lr_{str(lr)}_' + f'_bs_{str(batch_size)}'
    print(f'model_dir is {model_dir}')
    
    if not get_latest_file:
        datestring = 'here' 
        file_name_model =  model_dir + '/' + model_dir + '_' +  datestring 
    
    logger = logging.getLogger(__name__)

    # for saving files
    datestring = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M')
    model_dir_name = lstm_config + f'_ws_{str(window_size)}' +  f'_step_{str(step)}' + f'_lr_{str(lr)}' + f'_bs_{str(batch_size)}'
    model_dir = input_filepath_models + '/' + model_dir_name
    
    data_dir_name = f'ws_{window_size}_step_{step}'
    data_dir = f'{input_filepath}/{data_dir_name}/processed_numerical_{data_dir_name}'
    all_data = np.load(f'{data_dir}.npz', allow_pickle = True)
    
    x_test = np.asarray(all_data['x_test']).astype('float32')
    y_test = np.asarray(all_data['y_test']).astype('int32')

    # # Finding the most recent files
    list_of_jsons = glob.glob(f'{model_dir}/*.json')
    list_of_jsons.sort(key=os.path.getctime, reverse=True)

    list_of_h5 = glob.glob(f'{model_dir}/*.h5')
    list_of_h5.sort(key=os.path.getctime, reverse=True)
    
    # Load json and create model
    if get_latest_file :
        json_file = open(list_of_jsons[0], 'r')
    else:
        json_file = open(file_name_model + '.json', 'r')

    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)

    if get_latest_file :
        loaded_model.load_weights(list_of_h5[0])
        logger.info(f"Loaded most recent model: {str(list_of_h5[0])}")
    else:
        loaded_model.load_weights(model_name_h5)
        
    
    # # Custom adam optimizer
    adam = keras.optimizers.Adam(learning_rate = lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
  
    # Compute class weights for the training data
    # Assuming you have access to y_train or the class distribution
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_test), y_test)
    class_weights = dict(enumerate(class_weights))
    pos_weight = class_weights[1]  # Assuming the positive class is labeled as 1
    
    # Define the custom loss function with the positive class weight
    custom_loss = lambda y_true, y_pred: weighted_binary_crossentropy(y_test, y_pred, pos_weight)
    
    model.compile(loss='binary_crossentropy',
                optimizer=adam, metrics = [TruePositives(name="TP_Score"), FalsePositives(), TrueNegatives(), FalseNegatives(), AUC(name='AUC_PR', curve = "PR", thresholds = [0.35]), Recall(name='recall'), Precision(name='precision')])
        
    # calculate predicted probabilities
    scores_pred = loaded_model.predict(x_test, verbose=verbose)
    y_pred_proba = scores_pred[:, 0]  # assuming positive class is at index 0

    # Calculate R2
    r2 = r2_score(y_test, y_pred_proba)
    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred_proba)
    # Calculate MAPE
    #mape = np.mean(np.abs((y_test - y_pred_proba) / y_test)) * 100
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_proba))

    # Print or use these values as needed
    print("R2 Score:", r2)
    print("MAE:", mae)
    #print("MAPE:", mape)
    print("RMSE:", rmse)

    # Plot real-predicted scatter plot
    y_test_jittered = y_test + np.random.uniform(-0.05, 0.05, size=len(y_test))
    #plt.figure(figsize=(10, 8))  # Adjust figure size as needed
    #plt.scatter(y_test_jittered, y_pred_proba, color='blue')
    #plt.xlabel('True Labels')
    #plt.ylabel('Predicted Probabilities')
    #plt.title('Real-Predicted Scatter Plot')
    #plt.grid(True)
    #plt.savefig('/gpfs/home4/bvanleeuwen/pred_drop_off/reports/figures/real_predicted_scatter_plot.png')  # Save figure
    plt.figure(figsize=(10, 8))  # Adjust figure size as needed
    plt.hexbin(y_test_jittered, y_pred_proba, gridsize=50, cmap='Blues', bins='log')
    plt.colorbar(label='log10(N)')
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Probabilities')
    plt.title('Real-Predicted Scatter Plot')
    plt.savefig('/gpfs/home4/bvanleeuwen/pred_drop_off/reports/figures/real_predicted_scatter_heatmap.png')  # Save figure


    scores = loaded_model.evaluate(x_test, y_test, verbose = verbose)

    logger.info("Finished training")
    
    # Evaluate the loaded_model
    
    loss, TP, FP, TN, FN, AUC_PR = scores[0],  scores[1],  scores[2],  scores[3],  scores[4], scores[5]

    recall = compute_recall(TP, FP, TN, FN)
    miss_rate = compute_miss_rate(TP, FP, TN, FN)
    fbeta = compute_f_beta(TP, FP, TN, FN)
    fall_out = compute_fall_out(TP, FP, TN, FN)
    
    print("recall: " + str(recall))
    print("miss_rate: " + str(recall))
    print("fbeta: " + str(recall))
    print("fall_out: " + str(recall))
    print("loss, TP, FP, TN, FN, AUC_PR: ")
    print(loss, TP, FP, TN, FN, AUC_PR)

    logger.info(f"----------------------------------------------------------------------------------------------------")
    logger.info(f"LSTM configuration : {lstm_config}, Window size : {window_size}, step = {step}, learning rate = {lr}, batch size = {batch_size}, dropout = {dropout}")
    logger.info(f"TP = {TP}, FP = {FP}, TN = {TN}, FN = {FN} ")
    logger.info(f"loss = {loss}, AUC_PR = {AUC_PR}")
    logger.info(f"recall = {recall}, miss_rate = {miss_rate}, fbeta = {fbeta}, fall_out = {fall_out} ")
    logger.info(f"----------------------------------------------------------------------------------------------------")

   
    # if the results file exits, open it else make df
    try: 
        df_results = pd.read_csv(file_name_results)
    except Exception:
        df_results = pd.DataFrame(columns = ['Configuration', 'loss' ,'AUC_PR' ,'TP' , 'FP' , 'TN' , 'FN' , 'recall' , 'miss_rate' , 'fbeta' , 'fall_out' ])

    # add the new results to (existing) dataframe
    results_row = {'Configuration' : str(model_dir_name), 'loss' : loss, 'AUC_PR' : AUC_PR,'TP' : TP, 'FP' : FP, 'TN' : TN, 'FN' : FN, 'recall' : recall, 'miss_rate' : miss_rate, 'fbeta' : fbeta, 'fall_out' : fall_out }
    df_results_new = df_results.append(results_row, ignore_index=True)


    # save csv file
    df_results_new.to_csv(file_name_results, sep = ',')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt, filename = "logs/test.log")

    # # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).parents[2]
    sys.path.append(project_dir)
    # import all global variables and methods
    from src.variables_and_methods import *
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
