# Short-term Crowdedness Predictions for Public Locations in Amsterdam

This repository shows the code that I produced for my thesis project on predicting crowdedness at the City of Amsterdam.
With this code, we can make 2 hour-ahead predictions for the estimated visitor count at various public locations in Amsterdam. These locations are mostly parks and squares, but can also be shopping streets or market places. 

![](media/examples/emojis.png)

---


## Project Folder Structure

1) [`code`](./code): Folder containing a public version of the project code.
2) [`figures`](./figures): Folder containing figures used to illustrate the project.

---


## How it works

In the main notebook, resono_2h_predictions.ipynb predictions can be generated based on some settings. These settings can be given as arguments in the notebook (more explanation on the different arguments is given in the notebook). This notebook makes use of two files with functions prediction_model_helpers.py and resono_2h_predictions.py. The notebook executes the following steps:
1) Reading in the data from the database (crowdedenss data and external factors)
2) Preprocessing the data 
3) Training a prediction model
4) Generating the predictions 
5) Outputting the predictions to the database

**Backtesting**: several experiments were performed where the use of different predictor variables and prediction models were compared based on validation/test data.  The code for this falls under the backtesting section of the notebook. 

---
## Acknowledgements

Some functions in prediction_model_helpers.py were written by Shayla Jansen (@City of Amsterdam). 


