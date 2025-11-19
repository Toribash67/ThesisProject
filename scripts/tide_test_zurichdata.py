print("Importing libraries...")
import warnings

warnings.filterwarnings("ignore")

import logging

logging.disable(logging.CRITICAL)

import torch
import numpy as np
import pandas as pd

import sys

import optuna

import json

import time

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler

from darts.models import TiDEModel
from darts.metrics import mape, smape, rmse, mae, mase
from darts.utils.statistics import check_seasonality, plot_acf
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from darts.explainability import TFTExplainer
from darts.models.filtering.moving_average_filter import MovingAverageFilter

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from darts.datasets import ElectricityConsumptionZurichDataset
from functools import reduce

import matplotlib as mpl
import matplotlib.pyplot as plt

# before starting, we define some constants
chunk_size = 4
minutes_per_datapoint = 15*chunk_size
daySize = 1440//minutes_per_datapoint

hyperparameters_file = 'tide_hyperparameters.json'

figsize = (9, 6)
plt.style.use('seaborn-v0_8-paper')

plt.rcParams.update({
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False,     # don't setup fonts from rc parameters
    "legend.loc": "upper left",
    "lines.linewidth" : 1
    })

values = ElectricityConsumptionZurichDataset().load()

def reduce_series_by_averaging(covariates, chunk_size):
    num_chunks = len(covariates) // chunk_size
    new_time_index = covariates.time_index[chunk_size//2::chunk_size][:num_chunks]
    averaged_covariates = {}
    for covariate in covariates.columns:
        cov_values = covariates[covariate].values()
        averaged_cov_values = [np.mean(cov_values[i * chunk_size:(i + 1) * chunk_size]) for i in range(num_chunks)]
        averaged_covariates[covariate] = averaged_cov_values
    
    # Create a DataFrame for the reduced covariates
    averaged_cov_df = pd.DataFrame(averaged_covariates, index=new_time_index)
    
    # Convert to TimeSeries
    reduced_covariates = TimeSeries.from_dataframe(averaged_cov_df)
    
    return reduced_covariates

# Reduce the series

values = reduce_series_by_averaging(values, chunk_size)

target = values['Value_NE5']
past_covariates = values[['RainDur [min]', 'StrGlo [W/m2]', 'T [°C]', 'WD [°]', 'WVs [m/s]', 'WVv [m/s]', 'p [hPa]']]

target, _ = target.split_after(int(len(target)*0.65))

#60 20 20 split
train, validation = target.split_after(int(len(target)*0.6))
validation, test = validation.split_after(int(len(validation)*0.5))

transformer = Scaler()
train_transformed = transformer.fit_transform(train)
validation_transformed = transformer.transform(validation)
test_transformed = transformer.transform(test)

values_transformed = transformer.transform(target)

past_covariates_transformer = Scaler()
past_covariates_transformer.fit(past_covariates)
past_covariates_transformed = past_covariates_transformer.transform(past_covariates)

future_covariates_transformed_no_holidays = datetime_attribute_timeseries(values, attribute="weekday", one_hot=True)
future_covariates_transformed = future_covariates_transformed_no_holidays.add_holidays("CH", prov = "ZH")
holidays = future_covariates_transformed['holidays']

forecast_horizon = daySize
input_chunk_length = daySize*7

if(len(sys.argv) >= 2):
    forecast_horizon = int(sys.argv[1])
if(len(sys.argv) >= 3):
    input_chunk_length = int(sys.argv[2])

print(f"forecast_horizon: {forecast_horizon}, input_chunk_length: {input_chunk_length}")

with open(hyperparameters_file, 'r') as file:
    hyperparameters = json.load(file)
    
hidden_size = hyperparameters['hidden_size']
dropout_rate = hyperparameters['dropout_rate']

tide_model = TiDEModel(
    input_chunk_length=input_chunk_length,
    output_chunk_length=forecast_horizon,
    n_epochs=10,
    model_name='TiDE',
    hidden_size=hidden_size,
    dropout=dropout_rate,
    use_reversible_instance_norm=True,
    add_encoders={'cyclic': {'future': ['day','week','month']}},
    likelihood=QuantileRegression(),
    pl_trainer_kwargs={"gradient_clip_val": 1, "callbacks": [EarlyStopping(monitor="val_loss",patience=3, min_delta=0.0001,mode='min')]},
    save_checkpoints=True,
    force_reset=True,
    random_state=42,
)


def set_size(width_pt, fraction=1, subplots=(1, 1)):
    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27
    not_golden_ratio = (5**.5 - 1) / 1.8
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * not_golden_ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)

fig, ax = plt.subplots(1, 1, figsize=set_size(369.88583, subplots = (1,1)))
tide_model = TiDEModel.load(f"models/tide_zurich_model_{input_chunk_length}_{forecast_horizon}.pkl")

full_prediction_series = tide_model.predict(n=forecast_horizon, series = test_transformed[:7*daySize+11], past_covariates = past_covariates_transformed, future_covariates = future_covariates_transformed, verbose=False)

for i in np.arange(8*daySize+11,len(test_transformed)-forecast_horizon,daySize):
    predicted = tide_model.predict(n=forecast_horizon, series = test_transformed[:i], past_covariates = past_covariates_transformed, future_covariates = future_covariates_transformed, verbose=False)
    full_prediction_series = full_prediction_series.append(predicted)
    print(f"Progress: {i/len(test_transformed)}")
full_prediction_series = full_prediction_series[pd.Timestamp('2019-01-03 00:30:00'):pd.Timestamp('2019-12-24 23:30:00')]

full_prediction_series.to_csv('tide_timeseries_normal_covs.csv')

n = 4
naive = test_transformed.shift(daySize*7)[full_prediction_series.start_time():full_prediction_series.end_time()]
naive_error = rmse(test_transformed, naive[full_prediction_series.start_time():full_prediction_series.end_time()])
error = rmse(test_transformed, full_prediction_series)
for i in range(2,n+1):
    naive += values_transformed.shift((60/minutes_per_datapoint)*24*7*i)[full_prediction_series.start_time():full_prediction_series.end_time()]
naive /= n

naive.plot(ax = ax, label = 'Naive', linewidth = 1, color="#C44E52")
full_prediction_series.plot(ax = ax, label = 'Forecast', linewidth = 1,color="#5ED462")
values_transformed.plot(ax = ax, label = 'Actual', linewidth = 1, color = "#5E62D4")
ax.set_xlim([target.start_time(), target.end_time()])
#plt.show()


fig, ax = plt.subplots(1, 1, figsize=set_size(369.88583, subplots = (1,1)))
tide_model = TiDEModel.load(f"models/tide_zurich_model_no_covs_{input_chunk_length}_{forecast_horizon}.pkl")

full_prediction_series = tide_model.predict(n=forecast_horizon, series = test_transformed[:7*daySize+11], future_covariates = future_covariates_transformed,verbose=False)

for i in np.arange(8*daySize+11,len(test_transformed),daySize):
    predicted = tide_model.predict(n=forecast_horizon, series = test_transformed[:i], future_covariates = future_covariates_transformed,verbose=False)
    full_prediction_series = full_prediction_series.append(predicted)
    print(f"Progress: {i/len(test_transformed)}")
full_prediction_series = full_prediction_series[pd.Timestamp('2019-01-03 00:30:00'):pd.Timestamp('2019-12-24 23:30:00')]

full_prediction_series.to_csv('tide_timeseries_no_covs.csv')

naive.plot(ax = ax, label = 'Naive', linewidth = 1, color="#C44E52")
full_prediction_series.plot(ax = ax, label = 'predicted', linewidth = 1,color="#5ED462")
values_transformed.plot(ax = ax, label = 'Actual', linewidth = 1, color = "#5E62D4")
ax.set_xlim([target.start_time(), target.end_time()])
#plt.show()


fig, ax = plt.subplots(1, 1, figsize=set_size(369.88583, subplots = (1,1)))
tide_model = TiDEModel.load(f"models/tide_zurich_model_forecast_covs_{input_chunk_length}_{forecast_horizon}.pkl")

full_prediction_series = tide_model.predict(n=forecast_horizon, series = test_transformed[:7*daySize+11], future_covariates=future_covariates_transformed.stack(past_covariates_transformed), verbose=False)

for i in np.arange(8*daySize+11,len(test_transformed),daySize):
    predicted = tide_model.predict(n=forecast_horizon, series = test_transformed[:i], future_covariates=future_covariates_transformed.stack(past_covariates_transformed), verbose=False)
    full_prediction_series = full_prediction_series.append(predicted)
    print(f"Progress: {i/len(test_transformed)}")
full_prediction_series = full_prediction_series[pd.Timestamp('2019-01-03 00:30:00'):pd.Timestamp('2019-12-24 23:30:00')]

full_prediction_series.to_csv('tide_timeseries_full_covs.csv')

naive.plot(ax = ax, label = 'Naive', linewidth = 1, color="#C44E52")
full_prediction_series.plot(ax = ax, label = 'predicted', linewidth = 1,color="#5ED462")
values_transformed.plot(ax = ax, label = 'Actual', linewidth = 1, color = "#5E62D4")
ax.set_xlim([target.start_time(), target.end_time()])
#plt.show()


fig, ax = plt.subplots(1, 1, figsize=set_size(369.88583, subplots = (1,1)))
tide_model = TiDEModel.load(f"models/tide_zurich_model_no_holidays_{input_chunk_length}_{forecast_horizon}.pkl")

full_prediction_series = tide_model.predict(n=forecast_horizon, series = test_transformed[:7*daySize+11], future_covariates=future_covariates_transformed_no_holidays, verbose=False)

for i in np.arange(8*daySize+11,len(test_transformed),daySize):
    predicted = tide_model.predict(n=forecast_horizon, series = test_transformed[:i], future_covariates=future_covariates_transformed_no_holidays, verbose=False)
    full_prediction_series = full_prediction_series.append(predicted)
    print(f"Progress: {i/len(test_transformed)}")
full_prediction_series = full_prediction_series[pd.Timestamp('2019-01-03 00:30:00'):pd.Timestamp('2019-12-24 23:30:00')]

full_prediction_series.to_csv('tide_timeseries_no_holidays.csv')

naive.plot(ax = ax, label = 'Naive', linewidth = 1, color="#C44E52")
full_prediction_series.plot(ax = ax, label = 'predicted', linewidth = 1,color="#5ED462")
values_transformed.plot(ax = ax, label = 'Actual', linewidth = 1, color = "#5E62D4")
ax.set_xlim([target.start_time(), target.end_time()])
#plt.show()