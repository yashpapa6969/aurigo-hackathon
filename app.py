import streamlit as st
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Added for plotting
import plotly.express as px  # Added for interactive plots
from dotenv import load_dotenv
load_dotenv()

# 1) For forecasting:
import pickle
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from ml_models.ml_pipeline import get_n_days_forecast, get_num_workers

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

    # If 'daily_sales' (or target_column) is present, remove it before predicting
    data = data.drop(columns=[target_column], errors='ignore')
    # Ensure columns match those used in training
    data = data.reindex(columns=feature_names, fill_value=0)

    last_known_data = data.iloc[-1].values

    predictions = []
    current_data = last_known_data.copy()

    for _ in range(n_days):
        scaled_data = scaler.transform([current_data])
        next_day_prediction = model.predict(scaled_data)[0]
        predictions.append(next_day_prediction)
        
        # Shift array left, place new prediction at the end
        current_data = np.roll(current_data, shift=-1)
        current_data[-1] = next_day_prediction

    return predictions




# 2) Labor/worker calculation:
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




# 3) Cost pipeline dependencies:
from cal import compute_scores_for_multiple_projects
from get_material import get_materials
from analysis import AnalysingBuildingData

# -- Removed the confusing import --
# from ml_models.ml_pipeline import get_n_days_forecast as pipeline_dummy

analyser = AnalysingBuildingData("componentInput.json", "materialInput.json", os.getenv('OPENAI_API_KEY'))

# Synonym-based approach
SYNONYMS = {
    "cement": ["portland cement", "ultra high performance concrete", "concrete"],
    "steel": ["steel", "rebar", "metal reinforcement"],
    "sand": ["sand", "fine aggregate"],
    "aggregate": ["aggregate", "crushed stone", "gravel"],
    "bitumen": ["asphalt", "bitumen"],
    # Add more as needed
}

def find_price_for_material(material_name, materials_list, default_price=50.0):
    mat_lower = material_name.lower().strip()
    # Direct substring check
    for entry in materials_list:
        actual_name = entry["material"].lower()
        price = entry["price"]
        if mat_lower in actual_name or actual_name in mat_lower:
            return price
    
    # If direct substring fails, try synonyms
    if mat_lower in SYNONYMS:
        possible_synonyms = SYNONYMS[mat_lower]
        for entry in materials_list:
            actual_name = entry["material"].lower()
            price = entry["price"]
            for syn in possible_synonyms:
                if syn in actual_name:
                    return price

    # Fallback if no match
    return default_price


def pipeline(projects_example, dynamic_weights, OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')):
    # 1) Compute multi-project scores
    res = compute_scores_for_multiple_projects(
        projects_example,
        dynamic_weights,
        OPENAI_API_KEY
    )
    response = res[0]["priority_score"]  # Example usage or retrieval

    # 2) Get materials from custom function / model
    materials = get_materials(
        f"give all the materials needed for constructing {projects_example[0]['tasks']} along with the price in per meter cube"
    )
    print("Materials from the model:", materials)

    # 3) Query final JSON from LLM
    response1 = analyser.query(materials["data"])  
    response1 = json.loads(response1)  
    print("Final JSON response:\n", response1)

    # 4) Build cost dictionary with synonyms
    cost = {}
    for structure_name, materials_used_dict in response1.items():
        structure_total = 0.0
        for mat_name, mat_info in materials_used_dict.items():
            quantity = mat_info["quantity"]
            matched_price = find_price_for_material(
                material_name=mat_name,
                materials_list=materials["data"], 
                default_price=50.0
            )
            structure_total += quantity * matched_price

        cost[structure_name] = structure_total

    print("Calculated cost dictionary:\n", cost)
    return cost


def plot_costs(cost_dict):
    """Plot the project costs."""
    structures = list(cost_dict.keys())
    costs = list(cost_dict.values())
    
    plt.figure(figsize=(10, 5))
    plt.bar(structures, costs, color='skyblue')
    plt.title('Project Costs by Structure')
    plt.xlabel('Structures')
    plt.ylabel('Cost (in currency)')
    plt.xticks(rotation=45)
    st.pyplot(plt)  # Display the plot in Streamlit

def plot_forecast(predictions):
    """Plot the forecasted values."""
    days = list(range(1, len(predictions) + 1))
    
    plt.figure(figsize=(10, 5))
    plt.plot(days, predictions, marker='o', linestyle='-', color='orange')
    plt.title('Forecasted Values Over Days')
    plt.xlabel('Days')
    plt.ylabel('Forecasted Value')
    st.pyplot(plt)  # Display the plot in Streamlit

def plot_workers_needed(workers_needed):
    """Plot the estimated number of workers needed."""
    project_types = list(workers_needed.keys())
    workers = list(workers_needed.values())
    
    fig = px.bar(x=project_types, y=workers, labels={'x': 'Project Type', 'y': 'Estimated Workers'},
                 title='Estimated Workers Needed by Project Type')
    st.plotly_chart(fig)  # Display the interactive plot in Streamlit

# ---------------- STREAMLIT APP ----------------
def main():
    st.title("All-in-One Project & Forecasting Dashboard")
    """
    This app showcases:
    1) Project scoring & cost pipeline
    2) N-days forecast using a saved ML model
    3) Estimation of labor needed using another saved model
    """

    # --- NAVIGATION BAR ---
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a section", ["Project Cost Pipeline", "Forecast Next N Days", "Estimate Workers Needed"])

    if app_mode == "Project Cost Pipeline":
        st.header("Project Cost Pipeline")

        # Let user define one project (or more) dynamically
        project_id = st.text_input("Project ID", value="101")
        tasks_input = st.text_area("Tasks (comma-separated)", 
            value="Build elevated track, Install stations, Lay electrical lines")
        timeline = st.number_input("Timeline (months)", value=18)
        budget = st.number_input("Budget (millions)", value=500)
        location = st.text_input("Location", value="Urban Metro Region")

        tasks_list = [t.strip() for t in tasks_input.split(",")]

        # Build a single project dictionary for demonstration
        user_project = {
            "projectId": project_id,
            "tasks": tasks_list,
            "timeline": timeline,
            "budget": budget,
            "location": location
        }

        projects_example = [user_project]

        st.subheader("Dynamic Weights")
        w_cost_benefit = st.slider("Cost-Benefit Weight", 0.0, 1.0, 0.25, 0.05)
        w_socio_impact = st.slider("Socio-Impact Weight", 0.0, 1.0, 0.30, 0.05)
        w_risk_resilience = st.slider("Risk-Resilience Weight", 0.0, 1.0, 0.20, 0.05)
        w_strategic_align = st.slider("Strategic Alignment Weight", 0.0, 1.0, 0.25, 0.05)

        dynamic_weights = {
            "w_cost_benefit": w_cost_benefit,
            "w_socio_impact": w_socio_impact,
            "w_risk_resilience": w_risk_resilience,
            "w_strategic_align": w_strategic_align
        }

        st.header("Compute Project Cost")
        if st.button("Compute Project Cost"):
            st.write("Running pipeline (LLM-based + cost matching)...")
            try:
                cost_result = pipeline(projects_example, dynamic_weights)
                st.subheader("Cost Dictionary Output")
                st.json(cost_result)
                plot_costs(cost_result)  # Plot costs after computation
            except Exception as e:
                st.error(f"Error during pipeline execution: {e}")

    elif app_mode == "Forecast Next N Days":
        st.header("Forecast Next N Days")
        st.write("Predict daily cost or other time-series values using a saved model.")
        n_days = st.number_input("Number of Days to Forecast", 1, 100, 7)
        if st.button("Run Forecast"):
            try:
                predictions = get_n_days_forecast(n_days)
                st.subheader("Forecasted Values:")
                st.write(predictions)
                plot_forecast(predictions)  # Plot forecasted values
            except Exception as e:
                st.error(f"Error during forecasting: {e}")

    elif app_mode == "Estimate Workers Needed":
        st.header("Estimate Number of Workers Needed")
        project_type_input = st.selectbox("Project Type", ["Road Construction", "Building Construction"])
        size_input = st.number_input("Size (sq ft)", value=15000)
        deadline_input = st.number_input("Deadline (days)", value=150)
        machinery_input = st.selectbox("Machinery Available", ["Yes", "No"])
        productivity_input = st.number_input("Productivity (sq ft/day/worker/machine)", value=35.0)
        machines_available_input = st.number_input("Machines Available", value=1)

        if st.button("Estimate Workers"):
            sample_input = {
                "Project Type": project_type_input,
                "Size (sq ft)": size_input,
                "Deadline (days)": deadline_input,
                "Machinery Available": machinery_input,
                "Productivity (sq ft/day/worker/machine)": productivity_input,
                "Machines Available": machines_available_input
            }
            try:
                workers_needed = get_num_workers(sample_input)
                st.success(f"Estimated Workers Needed: {workers_needed}")
                plot_workers_needed({project_type_input: workers_needed})  # Plot workers needed
            except Exception as e:
                st.error(f"Error during labor estimation: {e}")

    st.write("---")
    st.markdown("**Done!** This unified dashboard helps you do project scoring, cost analysis, forecasting, and labor estimation all in one place.")

if __name__ == "__main__":
    main()

  