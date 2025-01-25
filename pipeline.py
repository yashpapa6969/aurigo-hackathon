import os
import json
from dotenv import load_dotenv
load_dotenv()
from cal import compute_scores_for_multiple_projects
from get_material import get_materials
from analysis import AnalysingBuildingData
from ml_models.ml_pipeline import get_n_days_forecast, get_num_workers


analyser = AnalysingBuildingData("componentInput.json", "materialInput.json",os.getenv('OPENAI_API_KEY'))
def pipeline(projects_example, dynamic_weights, OPENAI_API_KEY= os.getenv('OPENAI_API_KEY')):
    res = compute_scores_for_multiple_projects(
        projects_example,
        dynamic_weights,
        OPENAI_API_KEY
    )
    response = res[0]["priority_score"]
    materials = get_materials(f"give all the materials needed for constructing {projects_example[1]['tasks']} along with the price in per meter cube")
    print(materials)
    
    response1 = analyser.query(materials["data"])
    response1 = json.loads(response1.content)
    print(response1)

  
    material2price = {data:value for list_data in materials["data"] for data, value in list_data.items()}  
    
    cost = {}
    for structure, value in response1.items():
        for item, data in value.items():
            if item in material2price.keys():
                if structure in cost.keys():
                    cost[structure] += data * material2price[item]*1000
                else:
                    cost[structure] = data * material2price[item]*1000
    
    print(cost)
    

projects_example = [
    {
        "projectId": "101",
        "tasks": ["Build elevated track", "Install stations", "Lay electrical lines"],
        "timeline": 18,
        "budget": 500,
        "location": "Urban Metro Region"
    },
    {
        "projectId": "202",
        "tasks": [
            "Clear and grade existing roadway",
            "Lay asphalt and expand lanes",
            "Install drainage and stormwater systems"
        ],
        "timeline": 12,
        "budget": 450,
        "location": "Suburban / Semi-urban"
    },
    {
        "projectId": "203",
        "tasks": [
            "Conduct geotechnical surveys and soil testing",
            "Build foundation piles and piers",
            "Assemble steel beams and girders"
        ],
        "timeline": 18,
        "budget": 750,
        "location": "Riverine / Rural Region"
    }
]

dynamic_weights = {
    "w_cost_benefit": 0.25,
    "w_socio_impact": 0.30,
    "w_risk_resilience": 0.20,
    "w_strategic_align": 0.25
}

pipeline(projects_example, dynamic_weights)
    