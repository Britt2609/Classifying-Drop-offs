# -*- coding: utf-8 -*-
"""
    File name: pull_data.py
    Author: Maike Nützel
    Date created: 13/05/2023
    Date last modified: 
    Python Version: 3.9
"""

import click
import logging
from pathlib import Path
import glob
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
import random
from datetime import datetime
from src.variables_and_methods import df_columns
from src.variables_and_methods import index_variable


def make_MMSI_annonymous(df):
    MMSIs = df['MMSI'].unique()
    for MMSI in MMSIs:
        df.loc[df['MMSI'] == MMSI, 'MMSI'] = random.randint(100000000, 999999999)
    return df

def load_data(data_source, input_filepath, pos_or_neg , short):
    logger = logging.getLogger(__name__)
    """ Loads the positive datapoints.
    data_source == 'sweden' : 7 points of crashes at the coast of sweden
    data_source == 'madesmart' : 17 dropoffs in the Dutch part of the North Sea . 
    """

    df = pd.DataFrame() 
    logger.info(f"loaded files f'{input_filepath}/{pos_or_neg}/{data_source}/*.csv'")
    try: 
        files = glob.glob(f'{input_filepath}/{pos_or_neg}/{data_source}/*.csv')
    except FileNotFoundError:
        print(f" '{input_filepath}/{pos_or_neg}/{data_source}/*.csv' is not a valid filepath")

    for file in files:
        # alternative filetype
        logger.info(f'Loading : {str(file)}')
        # if data_source == 'sweden' and pos_or_neg == 'negative':
        #     df_temp = pd.read_csv(file, usecols=[1, 2, 3, 4, 5, 6])
        #     df_temp = df_temp.rename(columns={'# Timestamp': 'Date and UTC Time'})
        
        if data_source == 'madesmart' and pos_or_neg == 'positive':
            if short:
                df_temp = pd.read_csv(file, usecols=['Date and UTC Time', 'Latitude', 'Longitude' , 'SOG', 'COG'])
                # Randomly sample half of the DataFrame
                #half_length = len(df) // 2
                #df_temp = df_temp.sample(n=half_length)
            else:
                df_temp = pd.read_csv(file, usecols=['Date and UTC Time', 'Latitude', 'Longitude' , 'SOG', 'COG'])
            # Add column of randomly generated MMSIs since data is anonymous.            
            # note 1: each file consists of 1 MMSI
            df_temp["MMSI"] = random.randint(100000000, 999999999)
            # date format changed from '2021-04-10 02:06:07' ->  '10/04/2021 02:06:07'
            df_temp['Date and UTC Time'] = pd.to_datetime(df_temp['Date and UTC Time']).dt.strftime('%d/%m/%Y %H:%M:%S').apply(str)
        
        elif data_source == 'madesmart' and pos_or_neg == 'negative':
            if short:
                df_temp = pd.read_csv(file, usecols = df_columns)
                # Sample half of the DataFrame
                half_length = len(df_temp) // 2
                df_temp = df_temp.iloc[:int(half_length)]
            else:
                df_temp = pd.read_csv(file, usecols = df_columns)

        # else: # data_source == 'sweden' and pos_or_neg = 'positive' <- no longer in use. 
        #     df_temp = pd.read_csv(file, usecols = df_columns)
        
        # put columns in right order,
        df_temp = df_temp.reindex(columns = df_columns)
        # concat to full df 
        df = pd.concat([df, df_temp], ignore_index=True)
    return df

def remove_short_tracks(df, threshold):
    """ Removes all tracks from df that have less datapoints then 
    the threshold given as parameter 'threshold'. 
    """
    df_new = df.copy()
    logger = logging.getLogger(__name__)
    for MMSI, df_track in df.groupby('MMSI'):
        amount_points = df_track.shape[0]
        # df_new = df
        if amount_points < threshold:
            df_new = df.drop(df_track.index, inplace = False)
            logging.info(f"deleting track from MMSI {MMSI} with label {df_track['label'].unique()[0]}. # points: {amount_points}")
    return df_new

def convert_UTC_to_date(df):
    """Converts UTC Time and Date to an int version of date.
    fills in at same column in df.
    """
    logger = logging.getLogger(__name__)
    # new number of rows and columns
    n_rows, n_cols = df.shape
    for i in range(n_rows):
        df[i, index_variable('Date')] =  int(df[i, index_variable('Date')][6:10])*10000 + int(df[i, index_variable('Date')][3:5])*100 + int(df[i, index_variable('Date')][:2])
    return df

def sort_data(df):
    """ Sort data by MMSI then date
    """
    return df[np.lexsort((df[:, index_variable('Date')], df[:, index_variable('MMSI')]))]


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../interim).
    """
    data_source = "madesmart" # madesmart or sweden
    short = True

    logger = logging.getLogger(__name__)
    logger.info('loading data set from raw data')
    
    # load negative data from file  
    df_negative = load_data(data_source, input_filepath, pos_or_neg = "negative", short = short)
    df_negative['label'] = 0
    logger.info(f'loaded negative data')
    
    # convert MMSIs to random numbers
    df_negative = make_MMSI_annonymous(df_negative)
    logger.info(f'made negative df MMSIs annonymous')

    # load posive data from file
    df_positive = load_data(data_source, input_filepath, pos_or_neg = "positive", short =  short)
    df_positive['label'] = 1
    logger.info(f'loaded positive (already annonymous) data')

    
    np_positive = df_positive.values
    np_negative = df_negative.values

    np_positive = convert_UTC_to_date(np_positive)
    np_negative = convert_UTC_to_date(np_negative)

    # Find overlap in dates of the positive and negative dataset

    dates_to_delete = list(np.unique(np.intersect1d(np_positive[:, index_variable('Date')], np_negative[:,index_variable('Date')])))
    
    # Undersampling 1-10
    size_before = df_negative.shape[0]
    logger.info(f"df_positive.shape[0]: {df_positive.shape[0]}, df_negative.shape[0] : {df_negative.shape[0]}")
    n = min(df_positive.shape[0]*100, df_negative.shape[0])
    df_negative = df_negative.sample(n=n, random_state=101)
    logger.info(f"Undersampled {df_negative.shape[0] - size_before} rows smaller ")
    df = pd.concat([df_negative, df_positive], ignore_index = True)
    
    num_rows = df.shape[0]
    
    # get rid of nan rows (in speed and course) 
    df = df.dropna()
    num_deletions = num_rows - df.shape[0]
    logger.info(f'number of deletions are : {num_deletions}')
    
    # Delete tracks which are smaller than window_size
    window_size = 16
    df = remove_short_tracks(df, window_size)
    
    # change dataframe to numpy array
    df = df.values
    logger.info(f"converted Pandas dataframe to Numpy array")
    logger.info(f"check: MMSI: {df[0, index_variable('MMSI')]},  Date : {df[0, index_variable('Date')]}, Latitude : {df[0, index_variable('Latitude')]},  Longitude: {df[0, index_variable('Longitude')]},  SOG : {df[0, index_variable('SOG')]}, COG : {df[0, index_variable('COG')]}")
    logger.info(f"dates to delete: {dates_to_delete}")
    size_before = df.shape[0]
    if dates_to_delete:
        for date in dates_to_delete:
            row_num = list(np.where(np.asarray(np.any(df == date, axis=1)) == True))
            logger.info(f"index of dates_to_delete are {row_num}")
            df = np.delete(df, row_num, axis=0)
    logger.info(f"deleted duplicate dates, df is now {df.shape[0] - size_before} rows smaller (could be zero due to undersampling) ")
    # convert UTC Time and Date to int Date
    df = convert_UTC_to_date(df)
    logger.info(f"converted UTC and Date to int date, df shape is {df.shape}")


    # sort by MMSI then Date
    df = sort_data(df)
    
    #save to interim
    logger.info(f"Sorted array by MMSI, then Date")

    np.savez(f'{output_filepath}/all_tracks_sorted.npz', sorted_data = df)
    logger.info(f'saved all tracks to {output_filepath}/all_tracks_sorted.npz')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt, filename='logs/pulling.log')

    # not used in this stub but often useful for finding various files
    project_dir = str(Path(__file__).parents[2])

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

