# TempoCast - Team Minions


## Features



### How our 3D modelling system works:

- We can upload any 2D image or blueprint of our building/construction site.
- The 2D image is then analysed by the TRELLIS to create a 3D model.
- The 3D asset can then be accessed as a video 360 view, and a GLB format file
- Under the hood, TRELLIS uses Dino V2 to analyze images, and a VAE for encoding.
- We refine the model to understand, and interpret the 3D models with the images provided, in coontext of construction and infrastructure.
- The GLB file can then be convrted to a .gltf format file, to be accessible in softwares like Autocad Rivet or Grasshopper.

### Temporal Forecast, Budgeting, and Analysis model:

- Using the Seed Project Info provided, we are able to create a prioritization ladder for which projects require how much attention. The criteria for prioritization (weight) can be customized.
- Seed project info contains: list of projects, timeline, allocated budgets and location
- We have built a Temporal Data Forecastinf system using temporal fusion transformers (TFT) and random forest regressors. (XBoost + regression)
- We then come up with a material unit estimation in terms of volume and weight, and also the cost of total material given the industry average prices at that time.
- A forecast for the future, using an XBoost regression to analyse past data, is created to estimate future budget and resource allocation. It is accompanied by user intuitive graphs and metrics.
- Labour costs and estimates are also brought to light.

### YOLO safety working:

- YOLO is used to analyze the video, and analyze hwwat safety measures and practices are taking place.

### Data Preparation

Prepare your data by placing your CSV file (e.g., sample_daily_cost_dataset.csv) in the project directory. The data should include features such as day_of_week, weekend_indicator, holiday_indicator, month, quarter, and various cost metrics.

### Note: The PPT and the Video exist on this same repo

- Video link on YouTube: [here](https://www.youtube.com/watch?v=bPHXVP6WGUM)