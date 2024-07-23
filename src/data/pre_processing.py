# -*- coding: utf-8 -*-
"""
    File name: pre_processing.py
    Author: Maike Nützel
    Date created: 13/05/2023
    Date last modified: 
    Python Version: 3.9
"""

import click
import logging
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
import scipy
import sys
from tensorflow.keras import backend as K
from skimage.util.shape import view_as_windows
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier



def load_pulled_data(input_filepath):
    """ Loads all npz data from file and concats into Numpy array 'df'.
    """
    logger = logging.getLogger(__name__)

    try:
        file = next(Path(input_filepath).glob("*.npz"))
        df = np.load(file, allow_pickle=True)
        logger.info(f'loaded {file.name}')
    except StopIteration:
        raise RuntimeError("No .npz file present!")
    return df


def convert_relative_coordinates(df, unique_count):
    """ Makes each coordinate realtive to the starting point so that location is
    irrelevant.
    """
    logger = logging.getLogger(__name__)

    first_lat = []
    first_long = []
    k = 0
    postitive_data_points = []
    # find the first lat and long for each MMSI track
    for i in range(len(unique_count)):
        if df[k, index_variable('Label_df')] == 1:
            postitive_data_points.append(df[k, index_variable('MMSI')])
        first_lat.append(df[k, index_variable('Latitude')])
        first_long.append(df[k, index_variable('Longitude')])
        k += unique_count[i]

    logger.info(f"positive data labels of whole set is {len(postitive_data_points)}")
    k = 0
    last_count = 0

    # Change each lat and long into relative with the first lat and long for each MMSI track
    for i in range(len(df)):
        df[i, index_variable('Latitude')] = df[i, index_variable('Latitude')] - first_lat[k]
        df[i, index_variable('Longitude')] = df[i, index_variable('Longitude')] - first_long[k]
        if (last_count - i) == unique_count[k]:
            last_count = i
            k += 1
    return df

def split_sliding_window(df, window_size, step):
    logger = logging.getLogger(__name__)
    MMSIs = df[:, index_variable('MMSI')]
    window_shape = (window_size, number_of_variables + 1)
    logger.info(f"window shape is {window_shape}")
    df_MMSIs = np.split(df, np.where(np.diff(MMSIs))[0] + 1)
    split_df = None  # Initialize split_df outside the loop
    
    for df_MMSI in df_MMSIs:
    
        if window_size <= df_MMSI.shape[0]:
            df_2_MMSI = view_as_windows(df_MMSI, window_shape, step=step)
            df_2_MMSI = np.squeeze(df_2_MMSI)
            
            if df_2_MMSI.ndim < 3:
                df_2_MMSI = df_2_MMSI.reshape(-1, *df_2_MMSI.shape)  # Ensure df_2_MMSI has the same number of dimensions as split_df
                
            if split_df is None:
                split_df = df_2_MMSI
            else:
                split_df = np.concatenate([split_df, df_2_MMSI], axis=0)

    logger.info(f'shape split_df is is {split_df.shape}')
    x = split_df[:, :, :index_variable('Label_df')]
    y = split_df[:, 0, [index_variable('MMSI'), index_variable('Date'), index_variable('Label_df')]]
    return x, y, split_df





def split_data(x_train, y_train):
    """ Split data into test and training sets based on 'fraction_train'
    """
    logger = logging.getLogger(__name__)
    
    train_length = int(x_train.shape[0] * fraction_train)
    logger.info(f"x_train.shape[0] is {x_train.shape[0]}")
    logger.info(f'train length is {train_length}')
    # shuffle
    p = np.random.permutation(y_train.shape[0])
    x_train = x_train[p, :, :]
    y_train = y_train[p, :]

    y_test = y_train[train_length:, :]
    x_test = x_train[train_length:, :, :]

    y_train = y_train[:train_length + 1, :]
    x_train = x_train[:train_length + 1, :, :]

    logger.info(f"y_test shape is {y_test.shape}")

    # remove MMSIs and date for training
    x_train = x_train[ : , :, dynamic_idx]
    y_train = y_train[:, index_variable('Label_y')].astype(int) 

    # keep only dynamic information
    x_test = x_test[: , :, dynamic_idx]
    y_test = y_test[:, index_variable('Label_y')].astype(int) 

    number_ones_train = np.sum(y_train)
    length_train = y_train.shape[0]
    percentage_pos_train = (number_ones_train/length_train)*100
    number_ones_test = np.sum(y_test)
    length_test = y_test.shape[0]
    percentage_pos_test = (number_ones_test/length_test)*100

    logger.info(f"There exits {percentage_pos_train} % positive datapoints in train and {percentage_pos_test} % in test. ")
    logger.info(f"shapes are: x_train = {x_train.shape}, y_train = {x_train.shape}, x_test =  {x_test.shape},  y_test =  {y_test.shape}")

    return x_train, x_test, y_train, y_test

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn interim pulled data from (../interim) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info(f"----------------------------------------------------------------------------------------------------")
    
    step = int(window_size*step_config + 1)
    logger.info(f'window size : {window_size}, step = {step}')
    sorted_data = load_pulled_data(input_filepath)
    df = sorted_data['sorted_data']
    logger.info("loaded sorted data")
    
    # all MMSIs
    MMSIs= df[:, index_variable('MMSI')]
    # unique_vals: the unique MMSI numbers    
    # unique count: the length of the unique MMSI 
    unique_vals, unique_count = np.unique(MMSIs, return_counts=True)

    df = convert_relative_coordinates(df, unique_count)
    
    logger.info("converted to relative coordiates")
    df = normalize_variables(df)

    logger.info("normalised")
    
    
    # Calculate the correlation matrix
    data = pd.DataFrame(df).iloc[:, 2:]
    data = data.apply(pd.to_numeric, errors='coerce')
    corr_matrix = data.corr(method='pearson')  

    #fig_width = corr_matrix.shape[1]
    fig_width, fig_height = 5, 5
    #fig_height = corr_matrix.shape[0]

    # Create a heatmap using Matplotlib
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='nearest')
    
    # Add text annotations to each box
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='white')
    plt.colorbar(label='Correlation')
    plt.title('Correlation Heatmap')
    plt.xticks(np.arange(data.shape[1]), ['Latitude', 'Longitude' , 'SOG', 'COG', 'Label'], rotation=90)
    plt.yticks(np.arange(data.shape[1]), ['Latitude', 'Longitude' , 'SOG', 'COG', 'Label'])
    plt.tight_layout()
    plt.savefig("/gpfs/home4/bvanleeuwen/pred_drop_off/reports/figures/correlation_heatmap.png")
            
            

    x_train, y_train, split_df = split_sliding_window(df, window_size, step)
    x_train, x_test, y_train, y_test = split_data( x_train, y_train)
    
    # Reshape xtrain and xtest to be 2D arrays
    x_train_reshaped = x_train.reshape(x_train.shape[0], -1)  # Reshape to (457, 16*4)
    x_test_reshaped = x_test.reshape(x_test.shape[0], -1)     # Reshape to (114, 16*4)


    # train and test feed forward nn
    #feedforward_nn = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', random_state=42)
    #feedforward_nn.fit(x_train_reshaped, y_train)
    #y_pred_proba = feedforward_nn.predict_proba(x_test_reshaped)[:, 1]
    #y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Get the unique values and their counts
    unique_values, counts = np.unique(y_test, return_counts=True)
    
    # Print the unique values and their counts
    for value, count in zip(unique_values, counts):
        print(f"Value: {value}, Count: {count}")
        
    
    logreg = LogisticRegression(class_weight='balanced')
    #logreg = LogisticRegression()
    logreg.fit(x_train_reshaped, y_train)
    y_pred = logreg.predict(x_test_reshaped)
    y_pred_proba = logreg.predict_proba(x_test_reshaped)[:, 1]
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("TN: ", tn)
    print("FP: ", fp)
    print("FN: ", fn)
    print("TP: ", tp)
    
    # Evaluate the model
    print("Classification Report logreg baseline:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix logreg baseline:")  
    print(confusion_matrix(y_test, y_pred))

    # Calculate R2
    r2 = r2_score(y_test, y_pred_proba)
    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred_proba)
    # Calculate MAPE
    #mape = np.mean(np.abs((y_test - y_pred_proba) / y_test)) * 100
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_proba))

    # Print or use these values as needed
    print("logreg R2 Score:", r2)
    print("logreg MAE:", mae)
    #print("logreg MAPE:", mape)
    print("logreg RMSE:", rmse)


    # logger.info(f"split and filled in batches. x_train shape has {x_train.shape}, y_train shape has {y_train.shape} \n x_test shape has {x_test.shape} and y_test shape has {y_test.shape}")

    # # make new directory for setup if necessary
    # model_dir = output_filepath + '/' + f'ws_{str(window_size)}' + f'_step_{step}' 
    # Path(model_dir).mkdir(parents=True, exist_ok=True)
    # # save in right folder with the date
    # file_name_model =  model_dir + '/' + 'processed_numerical' + f'_ws_{str(window_size)}' + f'_step_{step}' + '.npz'

    # # save 
    # np.savez(file_name_model, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    # logger.info(f'making preprocessed data set from interim data and stored {file_name_model}')
    # logger.info(f"----------------------------------------------------------------------------------------------------")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt, filename='logs/pre_processing.log')

    project_dir = Path(__file__).parents[2]
    sys.path.append(project_dir)
    from src.variables_and_methods import *
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()

