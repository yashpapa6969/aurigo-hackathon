from pydantic import BaseModel, Field
from typing import Optional, Type
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
import pandas as pd
from pytorch_forecasting.data import GroupNormalizer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import matplotlib.pyplot as plt

class DailyCostModel(BaseModel):
    day_of_week: int = Field(..., description="Day of the week as an integer (1 = Monday, 7 = Sunday).")
    weekend_indicator: bool = Field(..., description="Indicates if the day is a weekend (True/False).")
    holiday_indicator: bool = Field(..., description="Indicates if the day is a public or regional holiday (True/False).")
    month: int = Field(..., description="Month of the year as an integer (1 = January, 12 = December).")
    quarter: int = Field(..., description="Quarter of the year as an integer (1 to 4).")
    cost_lag_1: float = Field(..., description="Daily cost value from the previous day.")
    cost_lag_7: float = Field(..., description="Daily cost value from 7 days ago.")
    cost_lag_30: float = Field(..., description="Daily cost value from 30 days ago.")
    cost_rolling_7: float = Field(..., description="7-day rolling average of daily cost.")
    cost_rolling_14: float = Field(..., description="14-day rolling average of daily cost.")
    cost_rolling_30: float = Field(..., description="30-day rolling average of daily cost.")
    staff_count: int = Field(..., description="Number of staff working on the given day.")
    overtime_hours: Optional[float] = Field(None, description="Total overtime hours worked on the given day.")
    production_volume: Optional[int] = Field(None, description="Volume of production output on the given day.")
    marketing_spend: Optional[float] = Field(None, description="Total marketing spend on the given day (in relevant currency).")
    daily_sales: Optional[float] = Field(None, description="Total daily sales revenue (in relevant currency).")
    inventory_level: Optional[float] = Field(None, description="Level of inventory available on the given day.")
    shipping_cost: Optional[float] = Field(None, description="Total shipping costs incurred on the given day (in relevant currency).")
    weather_temperature: Optional[float] = Field(None, description="Average temperature on the given day (in Celsius or Fahrenheit).")
    weather_precipitation: Optional[float] = Field(None, description="Amount of precipitation on the given day (in mm or inches).")
    exchange_rate: Optional[float] = Field(None, description="Exchange rate applicable on the given day (if relevant).")
    commodity_price_index: Optional[float] = Field(None, description="Commodity price index value on the given day (if relevant).")

    class Config:
        json_schema_extra = {
            "example": {
                "day_of_week": 1,  # 1 for Monday
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
                "daily_sales": 3000.0,
                "inventory_level": 250.0,
                "shipping_cost": 200.0,
                "weather_temperature": 25.0,
                "weather_precipitation": 12.5,
                "exchange_rate": 1.12,
                "commodity_price_index": 105.0,
            }
        }

class TFTLightningModel(pl.LightningModule):
    def __init__(self, tft_model):
        super().__init__()
        self.model = tft_model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        # Check if y_hat is a tuple and extract the first element if necessary
        if isinstance(y_hat, tuple):
            y_hat = y_hat[0]  # Adjust this based on your model's output structure

        loss = self.model.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        # Check if y_hat is a tuple and extract the first element if necessary
        if isinstance(y_hat, tuple):
            y_hat = y_hat[0]  # Adjust this based on your model's output structure

        loss = self.model.loss(y_hat, y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.03)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str) -> Type['TFTLightningModel']:
        # Attempt to load the model from a checkpoint
        try:
            tft_model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
            model = cls(tft_model)  # Wrap the TFT model in the TFTLightningModel
            return model
        except KeyError as e:
            if str(e) == '__special_save__':
                print("Warning: '__special_save__' key not found in checkpoint. This may indicate a version mismatch or a corrupted checkpoint.")
            else:
                print(f"Error loading model from checkpoint: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise

def prepare_data(csv_path="sample_daily_cost_dataset.csv"):
    """
    Prepare the data for the TFT model by:
    1. Loading the CSV
    2. Adding time_idx and group columns
    3. Converting categorical variables to string type
    4. Ensuring time_idx is integer type
    """
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Clean numeric columns by removing spaces and converting to float
    numeric_columns = [
        'cost_lag_1', 'cost_lag_7', 'cost_lag_30', 
        'cost_rolling_7', 'cost_rolling_14', 'cost_rolling_30',
        'staff_count', 'overtime_hours', 'production_volume',
        'marketing_spend', 'daily_sales', 'inventory_level',
        'shipping_cost', 'weather_temperature', 'weather_precipitation',
        'exchange_rate', 'commodity_price_index'
    ]
    
    for col in numeric_columns:
        # Remove spaces and convert to float
        df[col] = df[col].astype(str).str.replace(' ', '').astype(float)
    
    # Add time index (ensuring integer type)
    df['time_idx'] = range(len(df))
    df['time_idx'] = df['time_idx'].astype(int)
    
    # Convert categorical variables to string type
    df['day_of_week'] = df['day_of_week'].astype(str)
    df['month'] = df['month'].astype(str)
    df['quarter'] = df['quarter'].astype(str)
    
    # Add group column
    df['group'] = 'default'
    
    # Set target variable
    df['cost'] = df['daily_sales']
    
    # Convert boolean indicators to integers
    df['weekend_indicator'] = df['weekend_indicator'].astype(int)
    df['holiday_indicator'] = df['holiday_indicator'].astype(int)

    # Fill or drop NA values in the target variable
    df['cost'] = pd.to_numeric(df['cost'], errors='coerce')  # Ensure cost is numeric
    df['cost'].fillna(df['cost'].mean(), inplace=True)  # Fill NA with mean or use df.dropna(subset=['cost']) to drop

    return df

def create_tft_model(training_data: pd.DataFrame, 
                    prediction_length: int = 7,
                    max_prediction_length: int = 7,
                    max_encoder_length: int = 30):
    """
    Create the TFT model with the prepared dataset
    """
    # Ensure target variable is numeric and handle non-numeric values
    training_data['cost'] = pd.to_numeric(training_data['cost'], errors='coerce')
    training_data = training_data.dropna(subset=['cost'])  # Drop any remaining NA values

    # Calculate training cutoff
    training_cutoff = training_data["time_idx"].max() - max_prediction_length

    # Create training dataset
    training = TimeSeriesDataSet(
        training_data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="cost",
        group_ids=["group"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=[
            "day_of_week",
            "month",
            "quarter"
        ],
        time_varying_known_reals=[
            "weekend_indicator",
            "holiday_indicator",
            "staff_count",
            "marketing_spend",
            "cost_lag_7",
            "cost_lag_30",
            "cost_rolling_7",
            "cost_rolling_14",
            "cost_rolling_30"
        ],
        time_varying_unknown_reals=[
            "cost",
            "overtime_hours",
            "production_volume",
            "daily_sales",
            "inventory_level",
            "shipping_cost",
            "weather_temperature",
            "weather_precipitation",
            "exchange_rate",
            "commodity_price_index"
        ],
        target_normalizer=GroupNormalizer(
            groups=["group"],
            transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    # Create validation set
    validation = TimeSeriesDataSet.from_dataset(training, training_data, predict=True, stop_randomization=True)

    # Create dataloaders
    batch_size = 32
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    # Create the model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # Wrap the model in the LightningModule
    lightning_model = TFTLightningModel(tft)

    return lightning_model, train_dataloader, val_dataloader, training  # Return the training dataset

def train_tft_model(model, train_dataloader, val_dataloader):
    """
    Train the TFT model
    """
    # Configure trainer with callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=10,
        verbose=False,
        mode="min"
    )
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger("lightning_logs")

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="cpu",
        devices=1,
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    # Train the model
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    # After training
    trainer.save_checkpoint("trained_model.ckpt")

    return trainer, model

def make_predictions(model, data, training_dataset, mode="prediction", return_x=False):
    """
    Make predictions using the trained model.
    """
    model.eval()
    dataset = TimeSeriesDataSet.from_dataset(training_dataset, data, predict=True, stop_randomization=True)
    dataloader = dataset.to_dataloader(train=False, batch_size=32, num_workers=0)

    predictions = []
    for batch in dataloader:
        x, _ = batch  # Assuming the second element is the target, which we don't need for prediction
        with torch.no_grad():
            output = model(x)  # Get the output from the model
            
            # Access the 'prediction' attribute from the output
            if hasattr(output, 'prediction'):
                output = output.prediction  # Access the predictions attribute
            else:
                raise ValueError("Output does not have 'prediction' attribute.")

            # Ensure the output shape is correct
            if output.shape[1] != 7:
                raise ValueError(f"Expected output shape [batch_size, 7], but got {output.shape}")

            predictions.append(output)  # Append the output tensor directly
    predictions = torch.cat(predictions, dim=0)  # Concatenate the tensors directly

    # Concatenate predictions if needed
    predictions = torch.mean(predictions, dim=0,keepdim=True)  # Concatenate the tensors directly

    if return_x:
        return predictions, x  # Return predictions and input data if requested
    return predictions

def load_model(checkpoint_path):
    """
    Load the trained model from a checkpoint.
    """
    model = TFTLightningModel.load_from_checkpoint(checkpoint_path)  # Ensure this is correct
    return model

# Prepare your data
data = prepare_data("sample_daily_cost_dataset.csv")

# Create the model
model, train_dataloader, val_dataloader, training_dataset = create_tft_model(data)

# Train the model
trainer, trained_model = train_tft_model(model, train_dataloader, val_dataloader)

# Make predictions
predictions = make_predictions(trained_model, data, training_dataset)

# Check the shape of predictions
print("Predictions Shape:", predictions.shape)
print(predictions)
# Assuming predictions is a tensor of shape [1, 7, 7]
predictions_np = predictions.squeeze(0).detach().numpy()  # Convert to NumPy array and remove batch dimension

# Plot each quantile
for i in range(predictions_np.shape[1]):
    plt.plot(predictions_np[:, i], label=f'Quantile {i+1}')

plt.title('Predictions Over Time')
plt.xlabel('Time Steps')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()

# Example: Using predictions for sales forecasting
predicted_sales = predictions_np[:, 3]  # Assuming the 4th column is the median prediction

# Make decisions based on predicted sales
for time_step, sales in enumerate(predicted_sales):
    print(f"Predicted sales for time step {time_step + 1}: {sales:.2f}")
    if sales > threshold:  # Define a threshold for action
        print("Consider increasing inventory and marketing efforts.")
    else:
        print("Maintain current inventory levels.")