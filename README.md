# Daily Sales Forecasting with Temporal Fusion Transformer

## Overview

This project implements a daily sales forecasting model using the Temporal Fusion Transformer (TFT) architecture. The model leverages historical sales data along with various features such as marketing spend, weather conditions, and other relevant metrics to predict future sales.

## Features

- Data preparation for time series forecasting
- Implementation of the Temporal Fusion Transformer model
- Training and validation of the model using PyTorch Lightning
- Prediction capabilities with the trained model
- Visualization of predictions



## Usage

### Data Preparation

Prepare your data by placing your CSV file (e.g., `sample_daily_cost_dataset.csv`) in the project directory. The data should include features such as `day_of_week`, `weekend_indicator`, `holiday_indicator`, `month`, `quarter`, and various cost metrics.

### Training the Model

To train the model, run the following command:
