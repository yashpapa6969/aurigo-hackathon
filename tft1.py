import pandas as pd

data = [
    {
        "time_idx": i,
        "group_id": "series_1",
        "day_of_week": 1,
        "weekend_indicator": False,
        "holiday_indicator": False,
        "month": 8,
        "quarter": 3,
        "cost_lag_1": 500.0,
        "cost_lag_7": 490.0,
        "cost_lag_30": 470.0,
        "cost_rolling_7": 495.0,
        "cost_rolling_14": 485.0,
        "cost_rolling_30": 480.0,
        "staff_count": 50,
        "overtime_hours": 8.5,
        "production_volume": 1000,
        "marketing_spend": 1500.0,
        "daily_sales": 3000.0 + i * 10,  # just a dummy increasing pattern
        "inventory_level": 250.0,
        "shipping_cost": 200.0,
        "weather_temperature": 25.0,
        "weather_precipitation": 12.5,
        "exchange_rate": 1.12,
        "commodity_price_index": 105.0,
    }
    for i in range(100)  # example of 100 days
]

df = pd.DataFrame(data)
df["day_of_week"] = df["day_of_week"].astype(str)  # Convert day_of_week to string
df["holiday_indicator"] = df["holiday_indicator"].astype(str)  # Convert holiday_indicator to string
df["weekend_indicator"] = df["weekend_indicator"].astype(str)  # Convert weekend_indicator to string
df["month"] = df["month"].astype(str)  # Convert month to string
df["quarter"] = df["quarter"].astype(str)  # Convert quarter to string

import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_lightning.metrics import MeanSquaredError  # Import the appropriate metric
from pytorch_forecasting.metrics import QuantileLoss
# define maximum encoder (history) and prediction (forecast) lengths
max_encoder_length = 30
max_prediction_length = 7

# define the dataset
training_cutoff = df["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    df[lambda x: x.time_idx <= training_cutoff],  # use data up to cutoff
    time_idx="time_idx",
    target="daily_sales",
    group_ids=["group_id"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    
    # static covariates (example if group_id was something that doesn't change)
    static_categoricals=[],  # e.g. ["group_id"] if you had multiple series
    static_reals=[],         # e.g. ["store_size"] if it never changes
    
    # known covariates: we can know them in the future
    time_varying_known_categoricals=[
        "day_of_week",
        "weekend_indicator",
        "holiday_indicator",
        "month",
        "quarter",
    ],
    # force booleans or integers to be categorical if desired
    variable_groups={},  # if you have categorical groupings
    time_varying_known_reals=[
        # these are numerical features that you can know for future dates
        # e.g. you might know "exchange_rate" or "commodity_price_index" in future
        "exchange_rate",
        "commodity_price_index",
    ],
    
    # observed covariates: only known until the current day (cannot be known in future)
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=[
        "cost_lag_1",
        "cost_lag_7",
        "cost_lag_30",
        "cost_rolling_7",
        "cost_rolling_14",
        "cost_rolling_30",
        "staff_count",
        "overtime_hours",
        "production_volume",
        "marketing_spend",
        "inventory_level",
        "shipping_cost",
        "weather_temperature",
        "weather_precipitation",
    ],
    
    # label encoding
    categorical_encoders={
        "day_of_week": NaNLabelEncoder(),
        # if "weekend_indicator" is bool, you might turn it into int for a categorical
        "weekend_indicator": NaNLabelEncoder(),
        "holiday_indicator": NaNLabelEncoder(),
        "month": NaNLabelEncoder(),
        "quarter": NaNLabelEncoder(),
    },
    
    # how to fill missing values
    target_normalizer=None,  # you could apply a normalizer (e.g., TorchNormalizer())
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# create validation dataset by taking data beyond the training cutoff
validation = TimeSeriesDataSet.from_dataset(training, df, predict=False, stop_randomization=True)

# create dataloaders for model
batch_size = 32
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)



import torch
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

# define the model
tft = TemporalFusionTransformer(
    # config
    hidden_size=128,
    lstm_layers=1,
    dropout=0.1,
    output_size=1,  # because we're predicting a single value: daily_sales
    loss=MeanSquaredError(),  # Use a PyTorch Lightning metric instead of nn.Module
    # from dataset:
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=training.static_categoricals,
    static_reals=training.static_reals,
    time_varying_known_categoricals=training.time_varying_known_categoricals,
    time_varying_known_reals=training.time_varying_known_reals,
    time_varying_unknown_categoricals=training.time_varying_unknown_categoricals,
    time_varying_unknown_reals=training.time_varying_unknown_reals,
    # to ensure correct variable encodings:
    embedding_sizes=training.embedding_sizes,
    embedding_paddings=training.embedding_paddings,
    # how to scale continuous variables:
    scaler=training.scaler,
    # network config
    learning_rate=1e-3,
)

# callbacks for training
early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")
lr_logger = LearningRateMonitor(logging_interval='step')

# trainer
trainer = pl.Trainer(
    max_epochs=30,
    gpus=0,  # set to 1 if you have a GPU
    callbacks=[early_stop_callback, lr_logger],
)

# fit the model
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)



# load the best model if you have ModelCheckpoint, for example:
# best_tft = TemporalFusionTransformer.load_from_checkpoint("path/to/best.ckpt")

# or just use the current (already-fitted) model `tft`
predictions, x = tft.predict(val_dataloader, return_x=True)

# `predictions` will be a tensor of shape [batch_size, max_prediction_length]
# Reconstruct actual dates/indices:
actuals = torch.cat([y for x, y in iter(val_dataloader)])

raw_predictions = tft.predict(df, mode="raw")


