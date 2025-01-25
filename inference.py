import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet
from tft import load_model, prepare_data, make_predictions  # Ensure this matches your tft.py
from tft import TFTLightningModel

def predict(model, data, training_dataset):
    # Custom prediction logic
    with torch.no_grad():
        predictions = model.predict(training_dataset)
    return predictions

def main():
    # Load the trained model
    model = TFTLightningModel.load_from_checkpoint("/Users/yashgupta/Desktop/augrio/trained_model.ckpt")  # Ensure this is correct

    # Prepare the data for inference
    data = prepare_data("sample_daily_cost_dataset.csv")

    # Create the training dataset for inference
    training_dataset = TimeSeriesDataSet.from_dataset(model.model.training, data, predict=True, stop_randomization=True)

    # Make predictions
    predictions = predict(model, data, data)

    # Output the predictions
    print("Predictions Shape:", predictions.shape)
    print("Predictions:\n", predictions)

if __name__ == "__main__":
    main()
