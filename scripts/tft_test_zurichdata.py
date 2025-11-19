print("Importing libraries...")
import warnings

warnings.filterwarnings("ignore")

import logging

logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd

import sys

import optuna

import json

import time

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler

from darts.models import TFTModel, RNNModel, NaiveSeasonal
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

num_samples = 100

# before starting, we define some constants
chunk_size = 4
minutes_per_datapoint = 15*chunk_size
daySize = 1440//minutes_per_datapoint

hyperparameters_file = 'tft_hyperparameters.json'

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
 
# RainDur [min]:    Duration of precipitation (divided by 4 for conversion from hourly to quarter-hourly records)
# StrGlo [W/m2]:    Global solar irradiation
# T [°C]:           Temperature
# WD [°]:           Wind direction (Weird?)
# WVs [m/s]:        Wind vector speed
# p [hPa]:          Air pressure

#past_covariates['RainDur [min]'] = MovingAverageFilter(window=daySize, centered=False).filter(values['RainDur [min]'])
#train, validation = target.split_after(int(len(target)*0.2))
#validation, _ = validation.split_after(int(len(target)*0.1))

target, _ = target.split_after(int(len(target)*0.65))

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

future_covariates_transformed = datetime_attribute_timeseries(values, attribute="weekday", one_hot=True)


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
lstm_layers = hyperparameters['lstm_layers']
num_attention_heads = hyperparameters['num_attention_heads']
dropout_rate = hyperparameters['dropout_rate']

tft_model = TFTModel(
    input_chunk_length=input_chunk_length,
    output_chunk_length=forecast_horizon,
    hidden_size=hidden_size,
    lstm_layers=lstm_layers,
    num_attention_heads=num_attention_heads,
    dropout=dropout_rate,
    model_name="Fin_TFT",
    batch_size=16,
    n_epochs=100,
    add_relative_index=False,
    add_encoders={'cyclic': {'future': ['day','week','month','year']}},
    likelihood=QuantileRegression(),  # QuantileRegression is set per default
    pl_trainer_kwargs={"callbacks": [EarlyStopping(monitor="val_loss",patience=0,min_delta=0.01,mode='min')]},
    #loss_fn=MSELoss(),
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


tft_model = TFTModel.load(f"models/tft_zurich_model_{input_chunk_length}_{forecast_horizon}.pkl")

full_prediction_series = tft_model.predict(n=forecast_horizon, series = test_transformed[:7*daySize+11], past_covariates = past_covariates_transformed, future_covariates = future_covariates_transformed,num_samples = num_samples, verbose=False)

for i in np.arange(8*daySize+11,len(test_transformed)-forecast_horizon,daySize):
    predicted = tft_model.predict(n=forecast_horizon, series = test_transformed[:i], past_covariates = past_covariates_transformed, future_covariates = future_covariates_transformed,num_samples = num_samples, verbose=False)
    full_prediction_series = full_prediction_series.append(predicted)
    print(f"Progress: {i/len(test_transformed)}")

full_prediction_series.quantile(0.1).to_csv('tft_timeseries_10_normal_covs.csv')
full_prediction_series.quantile(0.5).to_csv('tft_timeseries_50_normal_covs.csv')
full_prediction_series.quantile(0.9).to_csv('tft_timeseries_90_normal_covs.csv')

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
ax.set_xlim([values_transformed.start_time(), values_transformed.end_time()])
#plt.show()

tft_model = TFTModel.load(f"models/tft_zurich_model_no_covs_{input_chunk_length}_{forecast_horizon}.pkl")

full_prediction_series = tft_model.predict(n=forecast_horizon, series = test_transformed[:7*daySize+11], future_covariates = future_covariates_transformed,num_samples = num_samples,verbose=False)

for i in np.arange(8*daySize+11,len(test_transformed)-forecast_horizon,daySize):
    predicted = tft_model.predict(n=forecast_horizon, series = test_transformed[:i], future_covariates = future_covariates_transformed,num_samples = num_samples,verbose=False)
    full_prediction_series = full_prediction_series.append(predicted)
    print(f"Progress: {i/len(test_transformed)}")
full_prediction_series = full_prediction_series[pd.Timestamp('2019-01-03 00:30:00'):pd.Timestamp('2019-12-24 23:30:00')]

full_prediction_series.quantile(0.1).to_csv('tft_timeseries_10_no_covs.csv')
full_prediction_series.quantile(0.5).to_csv('tft_timeseries_50_no_covs.csv')
full_prediction_series.quantile(0.9).to_csv('tft_timeseries_90_no_covs.csv')

naive.plot(ax = ax, label = 'Naive', linewidth = 1, color="#C44E52")
full_prediction_series.plot(ax = ax, label = 'predicted', linewidth = 1,color="#5ED462")
values_transformed.plot(ax = ax, label = 'Actual', linewidth = 1, color = "#5E62D4")
ax.set_xlim([values_transformed.start_time(), values_transformed.end_time()])
#plt.show()


tft_model = TFTModel.load(f"models/tft_zurich_model_forecast_covs_{input_chunk_length}_{forecast_horizon}.pkl")

full_prediction_series = tft_model.predict(n=forecast_horizon, series = test_transformed[:7*daySize+11], future_covariates=future_covariates_transformed.stack(past_covariates_transformed), num_samples = num_samples, verbose=False)

for i in np.arange(8*daySize+11,len(test_transformed),daySize):
    predicted = tft_model.predict(n=forecast_horizon, series = test_transformed[:i], future_covariates=future_covariates_transformed.stack(past_covariates_transformed), num_samples = num_samples, verbose=False)
    full_prediction_series = full_prediction_series.append(predicted)
    print(f"Progress: {i/len(test_transformed)}")
full_prediction_series = full_prediction_series[pd.Timestamp('2019-01-03 00:30:00'):pd.Timestamp('2019-12-24 23:30:00')]

full_prediction_series.quantile(0.1).to_csv('tft_timeseries_10_full_covs.csv')
full_prediction_series.quantile(0.5).to_csv('tft_timeseries_50_full_covs.csv')
full_prediction_series.quantile(0.9).to_csv('tft_timeseries_90_full_covs.csv')

naive.plot(ax = ax, label = 'Naive', linewidth = 1, color="#C44E52")
full_prediction_series.plot(ax = ax, label = 'predicted', linewidth = 1,color="#5ED462")
values_transformed.plot(ax = ax, label = 'Actual', linewidth = 1, color = "#5E62D4")
ax.set_xlim([values_transformed.start_time(), values_transformed.end_time()])
#plt.show()

tft_model = TFTModel.load(f"models/tft_zurich_model_no_holidays_{input_chunk_length}_{forecast_horizon}.pkl")

full_prediction_series = tft_model.predict(n=forecast_horizon, series = test_transformed[:7*daySize+11], future_covariates=future_covariates_transformed_no_holidays, num_samples = num_samples, verbose=False)

for i in np.arange(8*daySize+11,len(test_transformed),daySize):
    predicted = tft_model.predict(n=forecast_horizon, series = test_transformed[:i], future_covariates=future_covariates_transformed_no_holidays, num_samples = num_samples, verbose=False)
    full_prediction_series = full_prediction_series.append(predicted)
    print(f"Progress: {i/len(test_transformed)}")
full_prediction_series = full_prediction_series[pd.Timestamp('2019-01-03 00:30:00'):pd.Timestamp('2019-12-24 23:30:00')]

full_prediction_series.quantile(0.1).to_csv('tft_timeseries_10_no_holidays.csv')
full_prediction_series.quantile(0.5).to_csv('tft_timeseries_50_no_holidays.csv')
full_prediction_series.quantile(0.9).to_csv('tft_timeseries_90_no_holidays.csv')

naive.plot(ax = ax, label = 'Naive', linewidth = 1, color="#C44E52")
full_prediction_series.plot(ax = ax, label = 'predicted', linewidth = 1,color="#5ED462")
values_transformed.plot(ax = ax, label = 'Actual', linewidth = 1, color = "#5E62D4")
ax.set_xlim([values_transformed.start_time(), values_transformed.end_time()])
#plt.show()