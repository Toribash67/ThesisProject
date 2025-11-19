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

from darts.models import TFTModel
from darts.metrics import mape, smape, rmse, mae
from darts.utils.statistics import check_seasonality, plot_acf
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from darts.explainability import TFTExplainer

from darts.datasets import ElectricityConsumptionZurichDataset

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import matplotlib as mpl
import matplotlib.pyplot as plt

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

#train, validation = target.split_after(int(len(target)*0.2))
#validation, _ = validation.split_after(int(len(target)*0.1))

target, _ = target.split_after(int(len(target)*0.65))

train, validation = target.split_after(int(len(target)*0.6))
validation, test = validation.split_after(int(len(validation)*0.5))

transformer = Scaler()
train_transformed = transformer.fit_transform(train)
validation_transformed = transformer.transform(validation)

values_transformed = transformer.transform(target)

past_covariates_transformer = Scaler()
past_covariates_transformer.fit(past_covariates)
past_covariates_transformed = past_covariates_transformer.transform(past_covariates)

future_covariates_transformed_no_holidays = datetime_attribute_timeseries(values, attribute="weekday", one_hot=True)
future_covariates_transformed = future_covariates_transformed_no_holidays.add_holidays("CH", prov = "ZH")

forecast_horizon = daySize
input_chunk_length = daySize*7

if(len(sys.argv) >= 2):
	forecast_horizon = int(sys.argv[1])
if(len(sys.argv) >= 3):
	input_chunk_length = int(sys.argv[2])

print(f"forecast_horizon: {forecast_horizon}, input_chunk_length: {input_chunk_length}")

#values_transformed.plot(linewidth = 1)
#past_covariates_transformed.plot(linewidth = 1)
#future_covariates_transformed.plot(linewidth = 1)
#plt.show()

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
    n_epochs=10,
    add_relative_index=False,
    add_encoders={'cyclic': {'future': ['day','week','month']}},
    likelihood=QuantileRegression(),
    pl_trainer_kwargs={"callbacks": [EarlyStopping(monitor="val_loss",patience=3,min_delta=0.0001,mode='min')]},
    save_checkpoints=True,
    force_reset=True
)

print("Training...")
start_time = time.time()
tft_model.fit(train_transformed, past_covariates = past_covariates_transformed, future_covariates=future_covariates_transformed, val_series=validation_transformed,val_past_covariates = past_covariates_transformed, val_future_covariates = future_covariates_transformed, verbose=True)
print("Train time: %s seconds" % (time.time() - start_time))
tft_model = TFTModel.load_from_checkpoint('Fin_TFT', best=True)
tft_model.save(f"models/tft_zurich_model_{input_chunk_length}_{forecast_horizon}.pkl")

tft_model = TFTModel(input_chunk_length=input_chunk_length,output_chunk_length=forecast_horizon,hidden_size=hidden_size,lstm_layers=lstm_layers,num_attention_heads=num_attention_heads,dropout=dropout_rate,model_name="Fin_TFT",batch_size=16,n_epochs=10,add_relative_index=False,add_encoders={'cyclic': {'future': ['day','week','month']}},likelihood=QuantileRegression(),pl_trainer_kwargs={"callbacks": [EarlyStopping(monitor="val_loss",patience=3,min_delta=0.001,mode='min')]},save_checkpoints=True,force_reset=True)
tft_model.fit(train_transformed, future_covariates=future_covariates_transformed, val_series=validation_transformed, val_future_covariates = future_covariates_transformed, verbose=True)
print("Train time: %s seconds" % (time.time() - start_time))
tft_model = TFTModel.load_from_checkpoint('Fin_TFT', best=True)
tft_model.save(f"models/tft_zurich_model_no_covs_{input_chunk_length}_{forecast_horizon}.pkl")

tft_model = TFTModel(input_chunk_length=input_chunk_length,output_chunk_length=forecast_horizon,hidden_size=hidden_size,lstm_layers=lstm_layers,num_attention_heads=num_attention_heads,dropout=dropout_rate,model_name="Fin_TFT",batch_size=16,n_epochs=10,add_relative_index=False,add_encoders={'cyclic': {'future': ['day','week','month']}},likelihood=QuantileRegression(),pl_trainer_kwargs={"callbacks": [EarlyStopping(monitor="val_loss",patience=3,min_delta=0.001,mode='min')]},save_checkpoints=True,force_reset=True)
tft_model.fit(train_transformed, future_covariates=future_covariates_transformed.stack(past_covariates_transformed), val_series=validation_transformed, val_future_covariates = future_covariates_transformed.stack(past_covariates_transformed), verbose=True)
print("Train time: %s seconds" % (time.time() - start_time))
tft_model = TFTModel.load_from_checkpoint('Fin_TFT', best=True)
tft_model.save(f"models/tft_zurich_model_forecast_covs_{input_chunk_length}_{forecast_horizon}.pkl")

tft_model = TFTModel(input_chunk_length=input_chunk_length,output_chunk_length=forecast_horizon,hidden_size=hidden_size,lstm_layers=lstm_layers,num_attention_heads=num_attention_heads,dropout=dropout_rate,model_name="Fin_TFT",batch_size=16,n_epochs=10,add_relative_index=False,add_encoders={'cyclic': {'future': ['day','week','month']}},likelihood=QuantileRegression(),pl_trainer_kwargs={"callbacks": [EarlyStopping(monitor="val_loss",patience=3,min_delta=0.001,mode='min')]},save_checkpoints=True,force_reset=True)
tft_model.fit(train_transformed, future_covariates=future_covariates_transformed_no_holidays, val_series=validation_transformed, val_future_covariates = future_covariates_transformed_no_holidays, verbose=True)
print("Train time: %s seconds" % (time.time() - start_time))
tft_model.load_from_checkpoint(model_name='Fin_TFT', best=True)
tft_model.save(f"models/tft_zurich_model_no_holidays_{input_chunk_length}_{forecast_horizon}.pkl")