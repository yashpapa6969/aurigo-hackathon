import os
import json
from dotenv import load_dotenv
load_dotenv()
from cal import compute_scores_for_multiple_projects
from get_material import get_materials
from analysis import AnalysingBuildingData
from ml_models.ml_pipeline import get_n_days_forecast, get_num_workers


analyser = AnalysingBuildingData("componentInput.json", "materialInput.json",os.getenv('OPENAI_API_KEY'))
import os
import json
from dotenv import load_dotenv
load_dotenv()

# Example synonyms-based approach (Approach B). 
# Map known materials in LLM’s JSON to likely real-world equivalents from your model:
SYNONYMS = {
    "cement": ["portland cement", "ultra high performance concrete", "concrete"],
    "steel": ["steel", "rebar", "metal reinforcement"],
    "sand": ["sand", "fine aggregate"],
    "aggregate": ["aggregate", "crushed stone", "gravel"],
    "bitumen": ["asphalt", "bitumen"],
    # Add more as needed
}

def find_price_for_material(material_name, materials_list, default_price=50.0):
    """
    Attempts to match `material_name` (e.g. "Cement") with your materials_list (which may have strings like 
    "Portland Cement Concrete", "Asphalt", etc.).

    1) We do a simple substring check:
       if material_name.lower() in actual_material.lower() OR actual_material.lower() in material_name.lower().
    2) If still no direct substring match, we check synonyms.
    3) If no match is found, return `default_price` as a fallback.
    """
    mat_lower = material_name.lower().strip()

    # First, try direct substring matching
    for entry in materials_list:
        actual_name = entry["material"].lower()
        price = entry["price"]
        # Direct substring check in both directions
        if mat_lower in actual_name or actual_name in mat_lower:
            return price
    
    # If direct substring fails, try synonyms approach (Approach B):
    if mat_lower in SYNONYMS:
        # For each possible synonym in your list, see if it’s a substring in actual_name
        possible_synonyms = SYNONYMS[mat_lower]  # e.g. ["portland cement", "concrete"]
        for entry in materials_list:
            actual_name = entry["material"].lower()
            price = entry["price"]
            for syn in possible_synonyms:
                if syn in actual_name:
                    return price

    # If everything fails, return default price (Approach A)
    return default_price


def pipeline(projects_example, dynamic_weights, OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')):
    res = compute_scores_for_multiple_projects(
        projects_example,
        dynamic_weights,
        OPENAI_API_KEY
    )
    
    response = res[0]["priority_score"]  # Example usage
    
    # Get materials from some custom function (or from your model)
    materials = get_materials(
        f"give all the materials needed for constructing {projects_example[0]['tasks']} along with the price in per meter cube"
    )
    print("Materials from the model:", materials)

    # Query your final JSON from your LLM-based analyzer
    response1 = analyser.query(materials["data"])  
    response1 = json.loads(response1)  # The final JSON from the LLM
    print("Final JSON response:\n", response1)

    # Build cost dictionary
    cost = {}

    # Loop through each structure and its corresponding materials from response1
    for structure_name, materials_used_dict in response1.items():
        structure_total = 0.0

        for mat_name, mat_info in materials_used_dict.items():
            # 'mat_info' is a dict with "quantity" (and maybe "cost")
            quantity = mat_info["quantity"]

            # Attempt substring or synonyms-based match
            matched_price = find_price_for_material(
                material_name=mat_name,
                materials_list=materials["data"], 
                default_price=50.0  # Fallback if no match or synonyms found
            )

            if matched_price is not None:
                structure_total += quantity * matched_price
            else:
                # If there's truly no match, we can skip or use a default
                print(f"No match found for '{mat_name}' in our materials list! Using default price.")
                structure_total += quantity * 50.0

        cost[structure_name] = structure_total

    print("Calculated cost dictionary:\n", cost)
    return cost


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

#print(pipeline(projects_example, dynamic_weights))
    
