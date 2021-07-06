# General
import pandas as pd
import geopandas as gpd
import numpy as np
import os

# Modelling
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, Ridge
import mord
import math
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import LabelEncoder
## LSTM (/NN)
#from keras.models import Sequential
#from keras.layers import LSTM, Dense, Flatten, Dropout
#from keras.optimizers import Adam
#import tensorflow
#from keras.wrappers.scikit_learn import KerasClassifier
#from keras.utils import np_utils

# Sample weights
from sklearn.utils.class_weight import compute_sample_weight

# Synthetic minority oversampling
from imblearn.over_sampling import SMOTENC
from imblearn.over_sampling import SMOTE


### FUNCTIONS - modelling

def test_model_random_walk(df_y_train, predict_period):
    """
    Run random walk (naïve) model on the prediction or test set
    """

    # Use most recent value as predicted values
    prediction = [df_y_train.iloc[-1, 0]] * predict_period
    
    return prediction


def test_model_avg_3_weeks(df_y_train, df_y_predict, predict_period, n_samples_week, target):
    """
    Run model that uses the average of the previous 3 weeks on the prediction or test set
    """

    # Use average of last 3 weeks (for same time stamps) as predicted values
    df_hist = pd.concat([df_y_train, df_y_predict], 0)
    df_hist_1week = df_hist.shift(n_samples_week)
    df_hist_2week = df_hist.shift(2*n_samples_week)
    df_hist_3week = df_hist.shift(3*n_samples_week)
    df_hist_all = pd.concat([df_hist_1week, df_hist_2week, df_hist_3week], 1)
    df_hist_all = df_hist_all[df_hist_all.index.isin(df_y_predict.index)]
    
    if target == "count":
        # Average
        prediction = df_hist_all.mean(axis = 1)
    elif target == "level":
        # Majority class
        prediction = df_hist_all.mode(axis = 1)
    
    return prediction


def train_model_ridge_regression(df_X_train, df_y_train, Y_name, target, thresholds_all = None, 
                                 thresholds_one = None, use_smote = False):     
    
    """
    Train linear regression model using L2-regularization.
    
    thresholds_scaled: scaled version of the thresholds that matches the scaled target variable 
    use_smote: True/False (synthetic minority oversampling)
    """
    
    if use_smote:
        # Perform synthetic minority oversampling
        if thresholds_all is not None:
            X_train, y_train = perform_smote(df_X_train, df_y_train, Y_name, target, thresholds_all = thresholds_all)
        elif thresholds_one is not None:
            X_train, y_train = perform_smote(df_X_train, df_y_train, Y_name, target, thresholds_one = thresholds_one)
    else:
        # Convert data to numpy array
        X_train = np.array(df_X_train)
        y_train = np.array(df_y_train)
        
    # Initialize model
    model = Ridge()
      
    # Fit model
    model.fit(X_train, y_train)
    
    return model


def test_model_ridge_regression(model, df_X_test):
    """
    Run trained linear regression model using L2-regularization on the prediction or test set
    """
    
    # Convert data to numpy array
    X_test = np.array(df_X_test)
    
    # Predict data
    prediction = model.predict(df_X_test)
    
    return prediction


def train_model_ordinal_regression(df_X_train, df_y_train, Y_name, target, use_smote = False):     
    
    """
    Train ordinal regression model
    
    use_smote: True/False (synthetic minority oversampling)
    """
    
    # Perform synthetic minority oversampling
    if use_smote:
        X_train, y_train = perform_smote(df_X_train, df_y_train, Y_name, target, thresholds)
        
        # Convert y_train to integer dtype (necessary for this model)
        y_train = y_train.astype(int)
        
    else:
        # Convert data to numpy array
        X_train = np.array(df_X_train)
        y_train = np.array(df_y_train)
        y_train = y_train.reshape(len(y_train))
        
        # Convert y_train to integer dtype (necessary for this model)
        y_train = y_train.astype(int)
    
    # Initialize model (all threshold based model)
    model = mord.LogisticAT()
              
    # Fit model
    if use_sample_weights:
        model.fit(X_train, y_train, sample_weight = sample_weights)
    else:
        model.fit(X_train, y_train)
    
    return model


def test_model_ordinal_regression(model, df_X_test):
    """
    Run trained ordinal regression model on the prediction or test set
    """

    # Convert data to numpy array
    X_test = np.array(df_X_test)
    
    # Predict data
    prediction = model.predict(X_test)
    
    return prediction


def perform_smote(df_X_train, df_y_train, Y_name, target, thresholds_all = None, thresholds_one = None):
    
    # Convert predictor variables to numpy array
    X_train = np.array(df_X_train)

    # Column indices of categorical features
    cat_index = np.array(df_X_train.dtypes != 'float64')
    
    if target == 'count':
        if thresholds_all is not None:
            y_train = get_crowd_levels(df_y_train, Y_name, thresholds_all = thresholds_all)
        if thresholds_one is not None:
            y_train = get_crowd_levels(df_y_train, Y_name, thresholds_one = thresholds_one)
        y_train_both = pd.concat([df_y_train[Y_name], y_train], 1)
        y_train_both.columns = ['count', 'level']
        avg_count = y_train_both.groupby('level')['count'].mean()
    elif target == 'level':
        # Convert to category dtype (if not done before)
        y_train = df_y_train[Y_name].astype('category')

    # Only oversample if there are at least 3 samples
    if all(y_train.value_counts() >= 3):
        
        # If no categorical predictors: regular SMOTE
        if sum(cat_index) == 0:
            sm = SMOTE(k_neighbors=2) 
            X_train, y_train = sm.fit_resample(X_train, y_train) 
                 
        # If there are also categorical predictors: SMOTENC
        elif sum(cat_index) != X_train.shape[1]:
            sm = SMOTENC(categorical_features = cat_index, k_neighbors=2) 
            X_train, y_train = sm.fit_resample(X_train, y_train) 
        
        if target == 'count':
            # Convert target variable back to continuous value (mean of level)
            y_train = np.where(y_train[len(df_y_train):] == -1.0, 
                                avg_count[0], np.where(y_train[len(df_y_train):] == 0.0,
                                avg_count[1], avg_count[2]))
            y_train = pd.Series(np.concatenate([np.array(df_y_train[Y_name]), y_train.reshape(len(y_train))]).astype('float'))
            
        # Randomly select part of the new samples to dampen the oversampling
        drop_idx = y_train[len(df_y_train):].sample(frac = (1-0.5)).index.values
        y_train = np.delete(np.array(y_train), drop_idx)
        X_train = np.delete(X_train, drop_idx, axis = 0)
                      
    else:
        print("Warning: SMOTE not used because there are not enough samples for all crowd levels (at least 3 per level).")
        
        # Convert target variable to numpy array
        y_train = np.array(df_y_train)
        
    return X_train, y_train


def train_model_LSTM_regression(df_X_train, df_y_train, prediction_window,
                               tune_hyperparameters, batch_size, epochs, neurons, drop_out_perc, learning_rate):
    """
    Train LSTM regression model.
    
    tune_hyperparameters: True/False (perform grid search to tune hyperparameters)
    
    List (tune_hyperparameters = True) or single values (tune_hyperparameters = False) of value(s) for:
    batch_size: number of samples shown to the model at one iteration 
    epochs: number of epochs used to train the model
    neurons: number of neurons in the LSTM layer
    drop_out_perc: percentage of neurons used in the drop out layers
    learning_rate: learning rate used for weight updates
    """
    
    # Crop training data so that the number of samples is divisible by the prediction window
    new_length = int(prediction_window * math.floor(len(df_y_train)/prediction_window))
    
    df_y_train = df_y_train[len(df_y_train)-new_length:]
    df_X_train = df_X_train.iloc[len(df_X_train)-new_length:, :]
    
    # Number of features
    n_features = len(df_X_train.columns)
    
    # Convert data to numpy array
    X_train = np.array(df_X_train)
    y_train = np.array(df_y_train)

    # Reshape predictor variables into 3D array
    X_train = X_train.reshape(int(len(df_X_train) / prediction_window), prediction_window, n_features)
    
    # Reshape target variable into 2D array
    y_train = y_train.reshape(int(len(y_train) / prediction_window), prediction_window)
        
    if tune_hyperparameters:
        
        # Single values also have to be in a list
        prediction_window = [prediction_window]
        n_features = [n_features]
        
        # Determine optimal hyperparameters based on training data 
        opt_hp = hyperparameter_search_LSTM_regression(X_train, y_train, batch_size, epochs, neurons, drop_out_perc,
                                                       learning_rate, prediction_window, n_features)
    
        # Store optimal hyperparameters
        hyperparameters = opt_hp
    
        # Select optimal hyperparameters
        drop_out_perc = opt_hp['drop_out_perc']
        learning_rate = opt_hp['learning_rate']
        neurons = opt_hp['neurons']
        batch_size = opt_hp['batch_size']
        epochs = opt_hp['epochs']
        prediction_window = opt_hp['prediction_window']
        n_features = opt_hp['n_features']
        
    # Create sequential model
    model = Sequential()
                    
    # Add dropout layer
    model.add(Dropout(drop_out_perc, input_shape=(prediction_window, n_features)))
    
    # Add dense layer
    model.add(LSTM(neurons, activation='relu', return_sequences = True)) 
            
    # Add dropout layer
    model.add(Dropout(drop_out_perc))
                
    # Add an output layer with one unit
    model.add(Dense(1))
    
    # Optimizer
    optimizer = Adam(lr=learning_rate)
    
    # Compile the model 
    model.compile(optimizer=optimizer, loss='mse')

    # Fit the model with the selected hyperparameters
    model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs)
    
    if tune_hyperparameters:
        return model, hyperparameters
    else:
        return model
                   
    
def initialize_LSTM_regression(batch_size, epochs, neurons, drop_out_perc, learning_rate, prediction_window, n_features):
    """
    Initialize the LSTM model for regression
    """
    
    # Create sequential model
    init_model = Sequential()
    
    # Add dropout layer
    init_model.add(Dropout(drop_out_perc, input_shape=(prediction_window, n_features)))
    
    # Add LSTM layer
    init_model.add(LSTM(neurons, activation='relu', return_sequences = True))
    
    # Add dropout layer
    init_model.add(Dropout(drop_out_perc))
     
    # Add an output layer with one unit
    init_model.add(Dense(1))
    
    # Optimizer
    optimizer = Adam(lr=learning_rate)
    
    # Compile the model 
    init_model.compile(optimizer=optimizer, loss='mse')
    
    return init_model


def test_model_LSTM_regression(model, df_X_test):
    """
    Run trained LSTM regression model on the prediction or test set
    """
    
    # Convert data to numpy array
    X_test = np.array(df_X_test)
    
    # Reshape predictor variables into 3D array
    X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1])
    
    # Predict new data 
    prediction = model.predict(X_test).flatten()

    return prediction

        
def hyperparameter_search_LSTM_regression(X_train, y_train, batch_size, epochs, neurons, drop_out_perc, 
                                          learning_rate, prediction_window, n_features):
    """ 
    Perform a hyperparameter grid search on the training data (only for LSTM regression model)
    """
    
    # Initialize LSTM
    init_model = KerasClassifier(build_fn=initialize_LSTM_regression, verbose=0)

    # Initialize a dictionary with the hyper parameter options
    param_grid = dict(batch_size = batch_size, epochs = epochs, neurons = neurons,
                     drop_out_perc = drop_out_perc, learning_rate = learning_rate, prediction_window = prediction_window,
                      n_features = n_features)

    # Train the model and find the optimal hyper parameter settings
    grid_result = GridSearchCV(estimator=init_model, param_grid=param_grid, n_jobs=-1, cv=3, 
                               scoring = 'neg_mean_squared_error', refit = True)
 
    # Fit the model to retrieve the best parameters
    grid_result.fit(X_train, y_train)
        
    # Select the best hyper parameters 
    opt_hp = grid_result.best_params_
                
    return opt_hp

    
def train_model_LSTM_classification(df_X_train, df_y_train, prediction_window, batch_size, epochs, neurons, drop_out_perc,
                                   learning_rate):
    """
    Train LSTM classification model.
    
    prediction_window: the prediction window in terms of samples/time steps (equal to predict_period)
    batch_size: number of samples shown to the model at one iteration 
    epochs: number of epochs used to train the model
    neurons: number of neurons in the LSTM layer
    drop_out_perc: percentage of neurons used in the drop out layers
    learning_rate: learning rate used for updating the weights
    """   
    
    # Crop training data so that the number of samples is divisible by the prediction window
    new_length = int(prediction_window * math.floor(len(df_y_train)/prediction_window))
    
    df_y_train = df_y_train[len(df_y_train)-new_length:]
    df_X_train = df_X_train.iloc[len(df_X_train)-new_length:, :]
    
    # Number of features
    n_features = len(df_X_train.columns)
    
    # Convert data to numpy array
    X_train = np.array(df_X_train)
    y_train = np.array(df_y_train)
    
    # Number of classes
    n_classes = len(np.unique(y_train))

    # Reshape predictor variables into 3D array
    X_train = X_train.reshape(int(len(X_train) / prediction_window), prediction_window, n_features)
    
    # Reshape target variable into 2D array
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_y_train = encoder.transform(y_train)
    y_train = np_utils.to_categorical(encoded_y_train)
    y_train = y_train.reshape(int(len(y_train) / prediction_window), prediction_window, n_classes)
                                  
    # Create sequential model
    model = Sequential()
                    
    # Add dropout layer
    model.add(Dropout(drop_out_perc, input_shape=(prediction_window, n_features)))
    
    # Add dense layer
    model.add(LSTM(neurons, activation='relu', return_sequences = True)) 
            
    # Add dropout layer
    model.add(Dropout(drop_out_perc))
                
    # Add an output layer with as many units as classes
    model.add(Dense(n_classes, activation = "softmax"))
    
    # Optimizer
    optimizer = Adam(lr=learning_rate)
    
    # Compile the model 
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    # Fit the model with the chosen hyperparameters
    model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs)
                                               
    return model


def test_model_LSTM_classification(model, df_X_test):
    """
    Run trained LSTM classification model on the prediction or test set
    """
    
    # Convert data to numpy array
    X_test = np.array(df_X_test)
    
    # Reshape predictor variables into 3D array
    X_test = X_test.reshape(1, X_test.shape[0], X_test.shape[1])
    
    # Predict data 
    prediction = model.predict(X_test)
         
    # Reshape predictions
    prediction = prediction.reshape(prediction.shape[1], prediction.shape[2])
                
    # Predict class with highest probability
    prediction = np.argmax(prediction, axis = 1)
    prediction = np.where(prediction == 0, -1.0, np.where(prediction == 1, 0.0, 1.0))
    
    return prediction


def train_model_RF_regression(df_X_train, df_y_train):     
    
    """
    Train Random Forest regression model
    """
    
    # Convert data to numpy array
    X_train = np.array(df_X_train)
    y_train = np.array(df_y_train)
    
    # Reshape target variable into 1D array
    y_train = y_train.reshape(len(y_train))
    
    # Initialize model
    model = RandomForestRegressor()
      
    # Fit model
    model.fit(X_train, y_train)
    
    return model


def test_model_RF_regression(model, df_X_test):
    """
    Run trained Random Forest regression model on the prediction or test set
    """
    
    # Convert data to numpy array
    X_test = np.array(df_X_test)
    
    # Predict data
    prediction = model.predict(df_X_test)
    
    return prediction


def train_model_RF_classification(df_X_train, df_y_train, use_sample_weights = False, use_smote = False):     
    
    """
    Train Random Forest classification model
    
    use_sample_weights: True/False
    use_smote: True/False (synthetic minority oversampling)
    """
    
    # Perform synthetic minority oversampling
    if use_smote:
        X_train, y_train = perform_smote(df_X_train, df_y_train)
        
    else:
        # Convert data to numpy array
        X_train = np.array(df_X_train)
        y_train = np.array(df_y_train)
        y_train = y_train.reshape(len(y_train))
    
    # Calculate sample weights
    if use_sample_weights:
        sample_weights = compute_sample_weight('balanced', y_train)
        
    # Initialize model (all threshold based model)
    model = RandomForestClassifier()
              
    # Fit model
    if use_sample_weights:
        model.fit(X_train, y_train, sample_weight = sample_weights)
    else:
        model.fit(X_train, y_train)
    
    return model


def test_model_RF_classification(model, df_X_test):
    """
    Run trained Random Forest classification model on the prediction or test set
    """

    # Convert data to numpy array
    X_test = np.array(df_X_test)
    
    # Predict data
    prediction = model.predict(X_test)
    
    return prediction
    
    
### FUNCTIONS - post operational modelling

def unscale_y(y_scaled, Y_scaler):
    """
    Undo scaling on the target variable.
    """
    
    # Undo scaling using the scaler object
    y_unscaled = Y_scaler.inverse_transform(y_scaled)
    
    # Convert back to right format
    df_y_unscaled = pd.DataFrame(y_unscaled, columns = y_scaled.columns, index = y_scaled.index)
    df_y_unscaled.index.name = 'datetime'
    
    return df_y_unscaled


def prepare_final_dataframe(df, df_raw, data_source, target, model_version, data_version):
    """
    Preparing for the output to be stored in the database. 
    
    df_raw: the raw dataframe containig the target variable
    data_source: the data source for which the predictions are made, e.g. 'resono'
    """

    df_final = df.copy()
    
    if data_source == 'resono':
        
        if target == "count":
            # Set negative predictions to zero
            df_final[df_final < 0] = 0
    
        # Long format
        df_final = pd.melt(df_final.reset_index(), id_vars = 'datetime')
        df_final = df_final.sort_values("datetime", ascending = True)
    
        # Rename columns
        if target == 'count':
            df_final = df_final.rename(columns = {'value': 'total_count_predicted', 'datetime': 'measured_at',
                                                  'variable': 'location_name'})
            # Round to integers
            df_final['total_count_predicted'] = df_final['total_count_predicted'].round()
            
        elif target == "level":
            df_final = df_final.rename(columns = {'value': 'crowd_level_predicted', 'datetime': 'measured_at',
                                                  'variable': 'location_name'})
             # Round to integers
            df_final['crowd_level_predicted'] = df_final['crowd_level_predicted'].round()
    
    # Information on the predictions
    df_final['model_version'] = model_version
    df_final['data_version'] = data_version
    df_final['predicted_at'] = datetime.now()
    
    # Merge with original raw data frame
    df_final = df_raw.merge(df_final, how = 'right')
    
    # Select prediction time slots
    if data_source == "resono":
        df_final = df_final[df_final["measured_at"].isin(df.index)]
    
    return df_final

