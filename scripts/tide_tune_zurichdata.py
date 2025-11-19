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
from darts.metrics import mape, smape, rmse, mae, mse
from darts.utils.statistics import check_seasonality, plot_acf
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from darts.explainability import TFTExplainer

from darts.datasets import ElectricityConsumptionZurichDataset

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer

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

class CustomEarlyStopping(EarlyStopping):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_val_loss = float('inf')

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        current_val_loss = trainer.callback_metrics.get("val_loss")
        if current_val_loss is not None and current_val_loss < self.min_val_loss:
            self.min_val_loss = current_val_loss

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

future_covariates_transformed = datetime_attribute_timeseries(values, attribute="weekday", one_hot=True)
future_covariates_transformed = future_covariates_transformed.add_holidays("CH", prov = "ZH")

forecast_horizon = daySize
input_chunk_length = daySize*7

if(len(sys.argv) >= 2):
	forecast_horizon = int(sys.argv[1])
if(len(sys.argv) >= 3):
	input_chunk_length = int(sys.argv[2])

print(f"forecast_horizon: {forecast_horizon}, input_chunk_length: {input_chunk_length}")

def print_callback(study, trial):
     print(f"Current value: {trial.value}, Current params: {trial.params}")
     print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

def objective(trial):
    # Define hyperparameter search space using trial object
    hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64, 128, 256])
    dropout_rate = trial.suggest_float("dropout_rate", 0, 0.5)
    
    custom_early_stopping = CustomEarlyStopping(monitor="val_loss", patience=3, min_delta=0.0001, mode='min')

    tide_model = TiDEModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=forecast_horizon,
        n_epochs=10,
        model_name='Opt_TiDE',
        hidden_size=hidden_size,
        dropout=dropout_rate,
        use_reversible_instance_norm=True,
        add_encoders={'cyclic': {'future': ['day','week','month']}},
        likelihood=QuantileRegression(),  # QuantileRegression is set per default
        pl_trainer_kwargs={"gradient_clip_val": 1, "callbacks": [custom_early_stopping]},
        save_checkpoints=True,
        force_reset=True,
        random_state=42,
    )

    tide_model.fit(train_transformed[:len(train_transformed)//10],
        past_covariates = past_covariates_transformed,
        future_covariates=future_covariates_transformed,
        val_series=validation_transformed,
        val_past_covariates = past_covariates_transformed,
        val_future_covariates = future_covariates_transformed,
        verbose=True
    )

    return custom_early_stopping.min_val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20, callbacks=[print_callback])

with open(hyperparameters_file, 'w') as file: 
    json.dump(study.best_trial.params, file, indent=4)