import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


import pickle
import pandas as pd
import numpy as np

def forecast_next_n_days(model_path, scaler_path, feature_names_path, data_path, n_days, target_column='daily_sales'):

    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    with open(feature_names_path, 'rb') as feature_file:
        feature_names = pickle.load(feature_file)
        
    data = pd.read_csv(data_path)

    categorical_columns = ['day_of_week', 'month', 'quarter']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    data = data.drop(columns=[target_column], errors='ignore')
    data = data.reindex(columns=feature_names, fill_value=0)

    last_known_data = data.iloc[-1].values


    predictions = []
    current_data = last_known_data.copy()

    for _ in range(n_days):
        scaled_data = scaler.transform([current_data])
        next_day_prediction = model.predict(scaled_data)[0]
        predictions.append(next_day_prediction)
        
        current_data = np.roll(current_data, shift=-1)
        current_data[-1] = next_day_prediction

    return predictions


def get_n_days_forecast(n_days):
    model_path = "/Users/yashgupta/Desktop/augrio/ml_models/random_forest_model.pkl"
    scaler_path = "/Users/yashgupta/Desktop/augrio/ml_models/scaler.pkl"
    feature_names_path = "/Users/yashgupta/Desktop/augrio/ml_models/feature_names.pkl"
    data_path = "/Users/yashgupta/Desktop/augrio/ml_models/sample_daily_cost_dataset.csv"

    pred = forecast_next_n_days(model_path, scaler_path, feature_names_path, data_path, n_days)
    return pred






def load_model(save_path):
    with open(save_path, 'rb') as f:
        model, label_encoder = pickle.load(f)
    return model, label_encoder

def predict_labor_needed(input_data, model, label_encoder):

    input_df = pd.DataFrame([input_data])
    input_df['Project Type'] = label_encoder.transform(input_df['Project Type'])
    input_df['Machinery Available'] = input_df['Machinery Available'].map({'Yes': 1, 'No': 0})
    input_df = input_df[['Project Type', 'Size (sq ft)', 'Deadline (days)', 
                         'Machinery Available', 'Productivity (sq ft/day/worker/machine)', 
                         'Machines Available']]

    # Predict labor needed
    prediction = model.predict(input_df)
    return int(np.round(prediction[0]))

def get_num_workers(input):
    model, label_encoder = load_model(save_path="/Users/yashgupta/Desktop/augrio/ml_models/xgboost_model.pkl")
    prediction = predict_labor_needed(input, model, label_encoder)
    return prediction