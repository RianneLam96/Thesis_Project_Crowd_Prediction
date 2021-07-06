# General
import pandas as pd
import geopandas as gpd
import numpy as np
import os

# Evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report, confusion_matrix
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sn


### FUNCTIONS - backtesting 

def prepare_backtesting(start_test, predict_period, freq, df, Y_name, n_samples_week, target,
                       y_scaler):
    '''
    Prepare the data frames for backtesting given the starting timestamp for the predictions.
    '''
    
    # Define end timestamp of predictions
    end_test = pd.date_range(start_test, periods = predict_period, freq = freq)
    end_test = end_test[len(end_test)-1]
    
    # Training data
    df_X_train_bt, df_y_train_bt = get_train_df(df, Y_name, start_test)
    
    # Data frame to fill in with predictions
    df_y_predict_bt = get_future_df(start_test, predict_period, freq)
    
    # Drop ground truth values
    df_X_predict_bt = df.drop(Y_name, 1)

    # Select features for prediction period
    df_X_predict_bt = df_X_predict_bt[start_test:end_test]
    
    # Save ground truth values
    df_y_ground_truth_bt = df[[Y_name]][start_test:end_test]
    
    # If days are missing from ground truth, also discard them in data frame to predict
    df_y_predict_bt = df_y_predict_bt[df_y_predict_bt.index.isin(df_y_ground_truth_bt.index)]
    
    # Save scaled ground truth values
    df_y_ground_truth_bt_scaled = df_y_ground_truth_bt
    
    # Unscale the ground truth data if continuous
    if target == "count":
        df_y_ground_truth_bt = unscale_y(df_y_ground_truth_bt, y_scaler)
        
    return df_y_predict_bt, df_y_train_bt, df_y_ground_truth_bt, df_y_ground_truth_bt_scaled, df_X_train_bt, df_X_predict_bt


def test_model_avg_3_weeks_bt(df_y_train, df_y_predict, df_y_ground_truth_scaled, predict_period, n_samples_week, target):
    """
    Run model that uses the average of the previous 3 weeks on the prediction or test set
    """

    # Use average of last 3 weeks (for same time stamps) as predicted values
    df_hist = pd.concat([df_y_train, df_y_ground_truth_scaled], 0)
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
        prediction = df_hist_all.mode(axis = 1).iloc[:, 0]
    
    return prediction


def evaluate(pred, ground_truth, target, count_to_level = False, Y_name = None, thresholds = None, print_metrics=True):
    """
    Evaluate a prediction and ground_truth.
    
    count_to_level: convert results to crowd level and show confusion matrix using the given thresholds
    """
    
    if target == 'count':
        
        # fill NaNs with zeroes
        pred = pred.fillna(method = "ffill")
        pred = pred.fillna(method = "bfill")
        ground_truth = ground_truth.fillna(method = "ffill")
        ground_truth = ground_truth.fillna(method = "bfill")
        
        # Set negative predictions to zero
        pred[pred < 0] = 0
        ground_truth[ground_truth < 0] = 0
    
        # Calculate error metrics
        rmse = mean_squared_error(ground_truth, pred, squared=False)
        mae = mean_absolute_error(ground_truth, pred)
      
        # Calculate error metrics only for crowded moments (p75)   
        busy = np.percentile(ground_truth, 75)
        ground_truth_busy = ground_truth[ground_truth > busy].dropna()
        pred_busy = pred[ground_truth > busy].dropna()
        rmse_busy = mean_squared_error(ground_truth_busy, pred_busy, squared=False)
        mae_busy = mean_absolute_error(ground_truth_busy, pred_busy)
        
        # Store error metrics in dict
        error_metrics = dict({'rmse': rmse, 'rmse_busy': rmse_busy, 'mae': mae, 'mae_busy': mae_busy})
    
        if print_metrics:
            print(f"Root mean squared error: {rmse.round(1)}")
            print(f"Root mean squared error (crowded): {rmse_busy.round(1)}")
            print(f"Mean absolute error: {mae.round(1)}")
            print(f"Mean absolute error (crowded): {mae_busy.round(1)}")
            
        if count_to_level:
            pred = get_crowd_levels(pred, Y_name, thresholds)
            ground_truth = get_crowd_levels(ground_truth, Y_name, thresholds)
            
            # Confusion matrix
            conf_mat = confusion_matrix(ground_truth, pred)
            
            error_metrics['conf_mat'] = conf_mat
            
    elif target == "level":
        
        # Set dtype to category
        pred = pred.astype('category')
        ground_truth = ground_truth.astype('category')
        
        # Forward fill NaNs
        pred = pred.fillna(method = "ffill")
        ground_truth = ground_truth.fillna(method = "ffill")
        
        # Confusion matrix
        conf_mat = confusion_matrix(ground_truth, pred)
        
        # Classification report (recall, precision, F1)
        class_report = classification_report(ground_truth, pred, output_dict = True)
        class_report = pd.DataFrame(class_report).transpose()
        
        error_metrics = dict({"conf_mat": conf_mat, "class_report": class_report})
        
        if print_metrics:
            print(f"Confusion matrix: {conf_mat}")
            print(f"Classification report: {class_report}")
            
    return error_metrics
        
    
def visualize_backtesting(df_y_ground_truth_bt, df_y_benchmark, df_y_model, target, Y_name, error_metrics, 
                          count_to_level = False):
    '''
    Count: Plot the ground truth, benchmark and model predictions in one graph. 
    Optional: Plot the confusion matrix for the model predictions.
    
    Level: Plot the confusion matrix for the model predictions.
    '''

    if target == "count":
        fig, ax = plt.subplots(figsize = (30, 5))
        df_y_ground_truth_bt.plot(ax = ax)
        df_y_benchmark.plot(ax = ax)
        df_y_model.plot(ax = ax)
        plt.legend(["Ground truth", "Average past 3 weeks", "Model"])
        plt.ylabel("Visitor count (relative)")
        plt.xlabel("Date")
        
        plt.close()
        
        if count_to_level:
            sn.set(font_scale=1.5)
            fig2 = sn.heatmap(error_metrics['conf_mat'], 
                         annot=True, cbar = False, fmt='g', cmap = 'Blues').get_figure()
            plt.ylabel('True')
            plt.xlabel('Predicted')
            plt.xticks(rotation = 45)
            
            plt.close()
            
            return fig, fig2

    elif target == "level":
        sn.set(font_scale=1.5)
        fig = sn.heatmap(error_metrics['conf_mat'], 
                         annot=True, cbar = False, fmt='g', cmap = 'Blues').get_figure()
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.xticks(rotation = 45)
    
    return fig 
    
    
def feature_importance(weights, feature_names):
    ''' 
    Plot the feature weights
    
    feat_imp: array of weights/feature importances of the fitted model 
           (e.g. model.coef_ for regression model, model.feature_importances_ for RF model)
    feature_names: list of names for the features
    '''
     
    # Dataframe with feature weights/importance
    feat_imp = pd.DataFrame(weights, index = feature_names, 
                            columns =["Importance"]).sort_values("Importance")
    feat_imp['Sign'] = np.where(feat_imp <= 0, 'neg', 'pos')
    feat_imp['Feature'] = feat_imp.index
    feat_imp = pd.concat([feat_imp.head(5), feat_imp.tail(5)], 0)
    
    # Create figure of feature weights/importance
    fig = px.bar(feat_imp, x = "Importance",  y = 'Feature', labels={'value':'Importance', 'index':'Feature'}, 
             color = 'Sign', color_discrete_map={'neg':'red', 'pos':'blue'}, orientation = 'h')
    fig.update_layout(showlegend=False)
    
    return feat_imp, fig


def backtesting_results_all_locations(locations, RMSE_models, RMSE_benchmarks):
    ''' 
    Create a dataframe with the backtesting results summarized for all locations.
    '''
    
    df_results = pd.DataFrame()
    df_results["Location"] = locations
    df_results["RMSE_model"] = RMSE_models
    df_results["RMSE_benchmark"] = RMSE_benchmarks
    df_results["RMSE_difference"] = np.subtract(df_results["RMSE_model"], df_results["RMSE_benchmark"])
    
    return df_results
    


