# -*- coding: utf-8 -*-
"""
    File name: variables_and_methods.py
    Author: Maike Nützel
    Date created: 16/05/2023
    Date last modified: 
    Python Version: 3.9
"""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.math import logical_and
import numpy as np
import pandas as pd 

# performance measures
global beta

# running setting
global verbose
global data_source 

# pre-processing settings
global number_of_variables
global df_columns
global static_idx 
global dynamic_idx

# AIS constants 
global lat_max 
global long_max 
global sog_max 
global cog_max

# model settings 
global four_hot_encoding
global num_epochs
global fraction_train
global fraction_validate 
global activation_layer
global validation_split
global make_figures
global get_latest_file
global window_size
global step_config


# data description:
def describe_synonyms_metrics():
    dict = {
        'TP' : 'True positive',
        'TN' : 'True negatives',
        'FN' : 'False negatives',
        'FP' : 'False positives',
        'recall' : 'Sensitivity, True Positive Rate (TPR)',
        'specificity' : 'Selectivity, True Negative Rate (TNR)',
        'miss_rate' : 'Type II error, False Negative Rate (FNR)',
        'fall_out' : 'Type I error, False Positive Rate (FPR)',
        'precision' : 'Positive Predicive Value (PPV)',
        'NPV' : 'Negative Predictive Value',
        'FDR' : 'False Discovery Rate ',
        'FOR' : 'False Omission Rate ',
        f'f_{beta}' : 'F-beta score (e.g. F-1, F-2)',
        'J' : 'Youden’s J Statistic/Index (J), (Bookmaker) Informedness',
        'markedness' : '-',
        'accuracy' : '-',
        'balanced_accuracy' : '-',
        'MCC' : 'Matthews Correlation Coefficient (MCC)',
        'cohen' : 'Cohen’s kappa',
        'FM' : 'Fowlkes-Mallows Index, G-mean 1',
        'G_mean_2' : '-',
        'threat_score' : 'Critical Success Index'
    }
    return dict


def describe_metrics():
    dict = {
        'TP' : 'True positive',
        'TN' : 'True negatives',
        'FN' : 'False negatives',
        'FP' : 'False positives',
        'recall' : 'Out of all the drop-offs that took place, what fraction was classified as drop-offs?',
        'specificity' : 'Out of all the normal movements, what fraction was classified as normal?',
        'miss_rate' : 'What fraction of drop-offs did the model miss?',
        'fall_out' : 'What is the fraction false alerts?',
        'precision' :'Out of all the movements classfied as drop-offs, what fraction was actually drop-offs?',
        'NPV' : 'Out of all the movements  classified as normal, what fraction was actually normal',
        'FDR' : 'Out of all the movements classified as drop-offs, what fraction was actually normal?',
        'FOR' : 'TBA',
        f'f_{beta}' : 'Measure of how good the model is. F1: recall and precision equally as important, F2: recall twice as important.',
        'J' : 'TBA',
        'markedness' : 'TBA',
        'accuracy' : 'What fraction of the times was the model correct (Not to be used with inbalanced datasets) ',
        'balanced_accuracy' : 'TBA',
        'MCC' : 'TBA',
        'cohen' : 'TBA',
        'FM' : 'TBA',
        'G_mean_2' : 'TBA',
        'threat_score' : 'TBA'
    }
    return dict

def make_all_metrics_table(TP, FP, TN, FN):
    """ logs and returns the table of all DD scores
    """
    all_desciptions = describe_metrics()
    all_synonyms = describe_synonyms_metrics()
    df_all_desciptions  = pd.DataFrame.from_dict(all_desciptions, orient='index', columns=['Description'])
    df_all_synonyms  = pd.DataFrame.from_dict(all_synonyms, orient='index', columns=['Alternative names'])
    df = pd.concat([df_all_synonyms, df_all_desciptions], axis = 1)
    return df


# Metrics with backend:
# tp = K.sum(y_pos * y_pred_pos)
# tn = K.sum(y_neg * y_pred_neg)
# fp = K.sum(y_neg * y_pred_pos)
# fn = K.sum(y_pos * y_pred_neg)

def fp_keras(y_true, y_pred):
    return K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1 )))

def tn_keras(y_true, y_pred):
    return K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1 )))

def tp_keras(y_true, y_pred):
    return  K.sum(K.round(K.clip(y_true * y_pred, 0, 1 )))

def fn_keras(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1 )))

# def recall_keras(y_true, y_pred):
#     # TP/(TP + FN)
#     y_pred_pos = K.round(K.clip(y_pred, 0, 1))
#     y_pred_neg = 1 - y_pred_pos

#     y_pos = K.round(K.clip(y_true, 0, 1))
#     y_neg = 1 - y_pos

#     tp = K.sum(y_pos * y_pred_pos)
#     tn = K.sum(y_neg * y_pred_neg)

#     fp = K.sum(y_neg * y_pred_pos)
#     fn = K.sum(y_pos * y_pred_neg)
   
#     return tp / (tp + fn + K.epsilon())

def precision_keras(y_true, y_pred):
    # TP/(TP + FP)
    fp = get_FP(y_true, y_pred)
    tp = get_TP(y_true, y_pred)
    precision_keras = tp / (tp + fp + K.epsilon() )
    return precision_keras


def fbeta_keras(y_true, y_pred, beta=3):
    y_pred = K.clip(y_pred, 0, 1)

    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=1)
    fp = K.sum(K.round(K.clip(y_pred - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)

    p = tp / (tp + fp )
    r = tp / (tp + fn)

    num = (1 + beta ** 2) * (p * r)
    den = (beta ** 2 * p + r )

    return K.mean(num / den)

def miss_rate_keras(y_true, y_pred):
    # FN/(TP + FN)

    tp = tp_keras(y_true, y_pred)
    fn = fn_keras(y_true, y_pred)

    return fn / (tp + fn + K.epsilon() )  

def fall_out_keras(y_true, y_pred):
    # FP/(FP + TN)
    tn = tn_keras(y_true, y_pred)
    fp = fp_keras(y_true, y_pred)
    return fp / (fp + tn + K.epsilon())
    

# Metrics
def compute_recall(TP, FP, TN, FN):
    """ Computes the recall (=sensitivity = TPR) metric
    """
    try: 
        return TP/(TP + FN)
    except ZeroDivisionError:
        return 0 

def compute_specificity(TP, FP, TN, FN):
    """ Computes the specificity  (= selectivity = TNR) metric
    """
    try:
        return TN/(FP + TN)
    except ZeroDivisionError:
        return 0

def compute_miss_rate(TP, FP, TN, FN):

    """ Computes the miss rate (= FNR) metric
    """
    try:
        return FN/(TP + FN)
    except ZeroDivisionError:
        return 0

def compute_fall_out(TP, FP, TN, FN):

    """ Computes the fall-out (= FPR) metric
    """
    try:
        return FP/(FP + TN)
    except ZeroDivisionError:
        return 0
    

def compute_precision(TP, FP, TN, FN):
    """ Computes the precision (=PPV) metric
    """
    try:
        return TP/(TP + FP)
    except ZeroDivisionError:
        return 0 

def compute_NPV(TP, FP, TN, FN):
    """ Computes the negative predicive value (=NPV) metric
    """
    try:
        return TN/(FN + TN)
    except ZeroDivisionError:
        return 0 

def compute_FDR(TP, FP, TN, FN):
    """ Computes the false discovery rate (=FDR) metric
    """
    try:
        return FP/(TP + FP)
    except ZeroDivisionError:
        return 0 

def compute_FOR(TP, FP, TN, FN):
    """ Computes the false omission rate (=FOR) metric
    """
    try:
        return FN/(FN + TN)
    except ZeroDivisionError:
        return 0 


def compute_f_1(TP, FP, TN, FN):
    """ f1
    """
    precision = compute_precision(TP, FP, TN, FN)
    recall = compute_recall(TP, FP, TN, FN)
    try: 
        return (1 + 1**2) / (1/precision + 1**2/recall)
    except ZeroDivisionError:
        return 0

def compute_f_beta(TP, FP, TN, FN):
    """ Computes the f-beta score metric
    default is f-3.
    """
    precision = compute_precision(TP, FP, TN, FN)
    recall = compute_recall(TP, FP, TN, FN)
    try: 
        return (1 + beta**2) / (1/precision + beta**2/recall)
    except ZeroDivisionError:
        return 0

def compute_J(TP, FP, TN, FN):
    """ Computes the Youden's J metric (=J, Informedness)
    """
    specificity = compute_specificity(TP, FP, TN, FN)
    recall = compute_recall(TP, FP, TN, FN)
    try: 
        return recall + specificity - 1
    except ZeroDivisionError:
        return 0

def compute_markedness(TP, FP, TN, FN):
    """ Computes the Markedness (MK)
    """
    precision = compute_precision(TP, FP, TN, FN)
    NPV = compute_NPV(TP, FP, TN, FN)
    try: 
        return precision + NPV - 1
    except ZeroDivisionError:
        return 0

def compute_accuracy(TP, FP, TN, FN):
    """ Computes the accuracy (Acc)
    """
    try: 
        return (TP + TN)/ (TP + FP + TN + FN)
    except ZeroDivisionError:
        return 0

def compute_balanced_accuracy(TP, FP, TN, FN):
    """ Computes the balanced accuracy (BAcc)
    """
    specificity = compute_specificity(TP, FP, TN, FN)
    recall = compute_recall(TP, FP, TN, FN)
  
    try: 
        return 0.5*(recall + specificity)
    except ZeroDivisionError:
        return 0

def compute_MCC(TP, FP, TN, FN):
    """ Computes Mathews Correlation Coefficient (MCC)
    """

    try: 
        return (TP * TN - FP * FN)/((TP + FP)*(TN + FN)*(TP + FN)*(FP + TN))**0.5
    except ZeroDivisionError:
        return 0

def compute_cohen(TP, FP, TN, FN):
    """ Computes Cohen's kappa
    """
    accuracy = compute_accuracy(TP, FP, TN, FN)
    try: 
        Pe = (((TP + FP)*(TP + FN)) + ((FN + FN)*(FP + TN)))/ (TP + FP + TN + FN)**2
        return (accuracy - Pe)/(1 - Pe)
    except ZeroDivisionError:
        return 0

def compute_FM(TP, FP, TN, FN):
    """ Computes Fowlkes-Mallows Index (FM) (G-mean 1)
    """
    recall = compute_recall(TP, FP, TN, FN)
    precision = compute_precision(TP, FP, TN, FN)
    try: 
        return (recall * precision)**0.5
    except ZeroDivisionError:
        return 0

def compute_G_mean_2(TP, FP, TN, FN):
    """ Computes G-mean 2
    """
    recall = compute_recall(TP, FP, TN, FN)
    specificity = compute_specificity(TP, FP, TN, FN)
    try: 
        return (recall * specificity)**0.5
    except ZeroDivisionError:
        return 0

def compute_threat_score(TP, FP, TN, FN):
    """ Computes Threat score (=TS, Critical Success Index)
    """
    try: 
        return TP/((TP + FN) + FP)
    except ZeroDivisionError:
        return 0



# Other methods
def index_variable(variable : str):
    """ Returns the index of variable in the numpy array
    """
    if variable == 'MMSI':
        return 0
    if variable == 'Date':
        return 1
    if variable == 'Latitude':
        return 2
    if variable == 'Longitude':
        return 3 
    if variable == 'SOG':
        return 4
    if variable == 'COG':
        return 5
    if variable == 'Label_df':
        return 6
    if variable == 'Label_y':
        return 2 
    raise ValueError("Invalid variable")

def round_to_closest(x, base = 5):
    """Returns the value 'x' rounded to 'base'. 
    e.g. base = 5 : 43 -> 45, 42 -> 40
    """
    return base * round(x/base)

def convert_to_one_hot(dynamic_features_list):
    """Converts each dynamic feature to a one-hot-encoded vector
    and then concatenates all to make four-hot-encoded vector
    """
    one_hot_lat = np.zeros((2*lat_max, ))
    one_hot_lat[int(dynamic_features_list[0])] = 1
    four_hot_vector = one_hot_lat
    one_hot_long = np.zeros((2*long_max, ))
    one_hot_long[int(normal_variables[1])] = 1
    four_hot_vector = np.hstack((four_hot_vector, one_hot_long))
    one_hot_sog = np.zeros((sog_max, ))
    one_hot_sog[int(normal_variables[2])] = 1
    four_hot_vector = np.hstack((four_hot_vector, one_hot_sog))
    one_hot_cog = np.zeros((cog_max, ))
    one_hot_cog[int(normal_variables[3])] = 1
    four_hot_vector = np.hstack((four_hot_vector, one_hot_cog))
    return four_hot_vector

def compute_batch_count(split_size, unique_count):
    """Computes number of batches needed with a certain 'split_size'
    """
    # find number of batches
    batch_count = 0
    for i in unique_count:
        # if there are more then 'split_size' datapoints, cut off (hence increase batch_count)
        if i >= split_size:
            batch_count += (i - split_size - 1)
    return batch_count

def compute_size_four_hot():
    """ Computes the size of the four_hot_vector based on global variables 
    which are min and max values of each feature
    """
    # size of the four-for-vector
    return int(2*lat_max*100 + 2*long_max*100 + sog_max + cog_max/5)


def round_off_variables(df):
    """ Rounds off variables to appropriate size for four-hot encoding
    """
    df[:, index_variable('Latitude')] = np.round(df[:, index_variable('Latitude')].astype(np.double), decimals = 2)
    df[:, index_variable('Longitude')] = np.round(df[:, index_variable('Longitude')].astype(np.double), decimals = 2)
    df[:, index_variable('SOG')] = df[:, index_variable('SOG')].astype(int)
    for i in range(df.shape[0]):
        df[i, index_variable('COG')] = round_to_closest(df[i, index_variable('COG')], 5)
    return df

def normalize_variables(df):
    """Normalises variables between 0 and 1 for numeric estimation
    """
    df[:, [index_variable('Latitude')]] = ((df[:, [index_variable('Latitude')]] + long_max )/(2*lat_max))
    df[:, [index_variable('Longitude')]] = ((df[:, [index_variable('Longitude')]]+ long_max)/(2*long_max))
    df[:, [index_variable('SOG')]] = ((df[:, [index_variable('SOG')]]) / (sog_max))
    df[:, [index_variable('COG')]] = ((df[:, [index_variable('COG')]]) / (cog_max))
    df[:, [index_variable('MMSI')]] = (df[:, [index_variable('MMSI')]]).astype(int)
    df[:, [index_variable('Label_df')]] = (df[:, [index_variable('Label_df')]]).astype(int)
    df[:, [index_variable('Date')]] = (df[:, [index_variable('Date')]]).astype(int)
    return df


# Performance measures
beta = 3

# running settingw
verbose = False

# pre-processing settings
static_idx = [index_variable('MMSI'), index_variable('Date')]
dynamic_idx = [index_variable('Latitude'),index_variable('Longitude'), index_variable('SOG'), index_variable('COG')]
df_columns = ['MMSI', 'Date and UTC Time', 'Latitude', 'Longitude' , 'SOG', 'COG']
number_of_variables = len(df_columns)

# AIS constants
lat_max = 90.00 # lat_min = -90.00
long_max = 180.00 # long_min = -180.00
sog_max = 40 # based on max speed possible, SOG_min = 0
cog_max = 360 # COG_min = 0

# model settings
num_epochs =  50
fraction_train = 0.80
fraction_validate = 0.75
make_figures = True
get_latest_file = True
window_size=16
step_config=0.5

