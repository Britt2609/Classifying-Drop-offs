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
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import TruePositives, TrueNegatives, FalsePositives, FalseNegatives, Recall, AUC, Precision
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin


def weighted_binary_crossentropy(y_true, y_pred, pos_weight):
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    
    # Compute the weighted binary cross-entropy
    loss = -(pos_weight * y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
    return K.mean(loss, axis=-1)



@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath_models', type=click.Path())
@click.argument('output_filepath_figures', type=click.Path())
@click.argument('lstm_config', type=str)
@click.argument('lr', type=float)
@click.argument('batch_size', type=int)
@click.argument('dropout', type=float)
def main(input_filepath, output_filepath_models, output_filepath_figures, lstm_config, lr, batch_size, dropout):
    logger = logging.getLogger(__name__)
    step = int(window_size * step_config + 1)
    model_name = f'ws_{window_size}_step_{step}'
    data_dir = f'{input_filepath}/{model_name}/processed_numerical_{model_name}'
    all_data = np.load(f'{data_dir}.npz', allow_pickle=True)
    
    x_train = np.asarray(all_data['x_train']).astype('float32')
    y_train = np.asarray(all_data['y_train']).astype('int32')

    split = int(y_train.shape[0] * fraction_validate)

    x_val = x_train[split:, :, :]
    y_val = y_train[split:]

    x_train = x_train[:split, :, :]
    y_train = y_train[:split]

    # Compute class weights for class imbalance
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    class_weights = dict(enumerate(class_weights))
    pos_weight = class_weights[1]  # Assuming the positive class is labeled as 1

    datestring = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M')

    model = Sequential()
    elements = lstm_config.split('_')
   
    if elements[0] == '1L':
        model.add(LSTM(int(elements[1]), input_shape=(window_size, len(dynamic_idx))))
    elif elements[0] == '2L':
        model.add(LSTM(int(elements[1]), return_sequences=True, input_shape=(window_size, len(dynamic_idx))))
    else:
        raise Exception(f"Invalid LSTM configuration: {elements}")
    
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))

    # Custom Adam optimizer
    adam = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    logger.info("Started training")
    # Compile model with WBCE loss
    model.compile(loss=lambda y_true, y_pred: weighted_binary_crossentropy(y_true, y_pred, pos_weight),
                  optimizer=adam, 
                  metrics=[TruePositives(name="TP_Score"), FalsePositives(), TrueNegatives(), FalseNegatives(), 
                           AUC(name='AUC_PR', curve="PR", thresholds=[0.35]), Recall(name='recall'), Precision(name='precision')])

    # Early stopping
    es = EarlyStopping(monitor='val_AUC_PR', mode='min', verbose=1, patience=10)
    
    # Fit the model
    history = model.fit(x_train, y_train, class_weight=class_weights, batch_size=batch_size, epochs=num_epochs, 
                        validation_data=(x_val, y_val), verbose=2, callbacks=[es])
    datestring = datetime.strftime(datetime.now(), '%Y-%m-%d-%H-%M')
    
    # Serialize model to JSON
    model_json = model.to_json()
   
    # Create new directory for setup if necessary
    model_dir_name = lstm_config + f'_ws_{str(window_size)}' + f'_step_{str(step)}' + f'_lr_{str(lr)}' + f'_bs_{str(batch_size)}'
    model_dir = output_filepath_models + '/' + model_dir_name
    
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    # Save in right folder with the date
    file_name_model = model_dir + '/' + model_dir_name + '_' + datestring
   
    with open(file_name_model + '.json', 'w') as json_file:
        json_file.write(model_json)
    
    # Serialize weights to HDF5
    model.save_weights(file_name_model + '.h5')
    logger.info(f'Saved model and weights to file location: {file_name_model}')

    validation_PR = np.amax(history.history['val_AUC_PR']) 
    logger.info(f'Max validation AUC_PR: {validation_PR}')
    scores = model.evaluate(x_val, y_val, verbose=verbose)

    logger.info("Finished training")
    
    # Evaluate the model
    loss, TP, FP, TN, FN, AUC_PR = scores[0], scores[1], scores[2], scores[3], scores[4], scores[5]

    recall = compute_recall(TP, FP, TN, FN)
    miss_rate = compute_miss_rate(TP, FP, TN, FN)
    fbeta = compute_f_beta(TP, FP, TN, FN)
    fall_out = compute_fall_out(TP, FP, TN, FN)

    logger.info(f"----------------------------------------------------------------------------------------------------")
    logger.info(f"LSTM configuration: {lstm_config}, Window size: {window_size}, step = {step}, learning rate = {lr}, batch size = {batch_size}, dropout = {dropout}")
    logger.info(f"TP = {TP}, FP = {FP}, TN = {TN}, FN = {FN}")
    logger.info(f"Loss = {loss}, AUC_PR = {AUC_PR}")
    logger.info(f"Recall = {recall}, Miss rate = {miss_rate}, F-beta = {fbeta}, Fall-out = {fall_out}")
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
        plt.title('Weighted Binary Cross-Entropy Loss During Training')
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
        plt.plot
