# Thesis Project: Impact of Weather Data on Electrical Power Grid Load Forecasting

## Required Libraries:

* **Numpy**

* **Pandas**

* **MatplotLib**

* **Darts**

## Basic usage:

For each model `x`, there are three scripts: `x_tune_zurichdata.py`, `x_train_zurichdata.py`, `x_test_zurichdata.py`, which ought to be run in that order.

You can optionally run the scripts with 1 or 2 parameters. The first parameter is the forecast horizon, i.e. how many datapoints the model forecasts. The second parameter is the input length, i.e. how many datapoints the model uses for inputs.

* `x_tune_zurichdata.py` finds the optimal hyperparameters for the model, and saves the parameters to a *json* file.
* `x_train_zurichdata.py` trains using the model using the hyperparameters from the *json*, and stores the model to a *pkl* file
* `x_test_zurichdata.py` loads the model and tests it on the validation set, and can optionally plot its related graphs.
