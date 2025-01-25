from openai import OpenAI
import os
import re

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def compute_scores_for_multiple_projects(projects, weights, openai_api_key):
    """
    Computes priority scores for multiple projects.
    
    :param projects: List of project data dicts, each like:
        {
          "projectId": "101",
          "tasks": ["Build elevated track", "Install stations"],
          "timeline": 18,
          "budget": 500,
          "location": "Urban Metro Region"
        }
    :param weights: Dict with the metric weights, e.g.:
        {
          "w_cost_benefit": 0.25,
          "w_socio_impact": 0.30,
          "w_risk_resilience": 0.20,
          "w_strategic_align": 0.25
        }
    :param openai_api_key: Your OpenAI API key.
    :return: A list of score results (one entry per project), each dict with:
        {
          "projectId": ...,
          "cost_benefit": ...,
          "socio_impact": ...,
          "risk_resilience": ...,
          "strategic_alignment": ...,
          "priority_score": ...
        }
    """
    results = []
    for project_data in projects:
        score = compute_project_score_with_openai(project_data, weights, openai_api_key)
        results.append(score)
    return results

def compute_project_score_with_openai(project_data, weights, openai_api_key):
    """
    Compute a priority score for an infrastructure project, combining:
      1) Cost-Benefit Ratio (cost_benefit)
      2) Socioeconomic Impact (socio_impact)
      3) Risk & Resilience (risk_resilience)
      4) Strategic Alignment (strategic_alignment)
    """

    # 1. Extract project data
    project_id = project_data.get("projectId", "Unknown")
    tasks = project_data.get("tasks", [])
    timeline_months = project_data.get("timeline", 12)
    budget_crores = project_data.get("budget", 1)  # Avoid zero division
    location = project_data.get("location", "Default Region")

    # 2. Compute Cost-Benefit Ratio (naive example)
    location_factor_map = {
        "Urban Metro Region": 1.2,
        "Suburban / Semi-urban": 1.0,
        "Riverine / Rural Region": 0.8
    }
    loc_factor = location_factor_map.get(location, 1.0)
    cost_benefit = (loc_factor * 10.0) / (budget_crores / 100.0)

    # 3. Socioeconomic Impact
    socio_impact = call_openai_for_metric(
        metric_name="socioeconomic impact",
        project_data=project_data,
        model="gpt-3.5-turbo",
        default_value=5.0
    )

    # 4. Risk & Resilience
    risk_resilience = call_openai_for_metric(
        metric_name="risk & resilience",
        project_data=project_data,
        model="gpt-3.5-turbo",
        default_value=5.0
    )

    # 5. Strategic Alignment (simple rule-based)
    strategic_alignment = calculate_strategic_alignment(tasks, location)

    # 6. Combine metrics via weighted sum
    w_cb = weights.get("w_cost_benefit", 0.25)
    w_si = weights.get("w_socio_impact", 0.30)
    w_rr = weights.get("w_risk_resilience", 0.20)
    w_sa = weights.get("w_strategic_align", 0.25)

    priority_score = (
        w_cb * cost_benefit +
        w_si * socio_impact +
        w_rr * risk_resilience +
        w_sa * strategic_alignment
    )
    priority_score = round(priority_score, 2)

    return {
        "projectId": project_id,
        "cost_benefit": round(cost_benefit, 2),
        "socio_impact": round(socio_impact, 2),
        "risk_resilience": round(risk_resilience, 2),
        "strategic_alignment": round(strategic_alignment, 2),
        "priority_score": priority_score
    }

def call_openai_for_metric(metric_name, project_data, model, default_value=5.0):
    """
    Prompts OpenAI to provide a numeric score (1-10).
    Uses regex fallback to extract a single number if model returns extra text.
    """
    system_prompt = (
        f"You are an expert infrastructure analyst. "
        f"Estimate the {metric_name} for a project on a scale of 1 to 10 (10 is highest). "
        f"ONLY OUTPUT THE NUMBER. No additional words or explanation."
    )
    user_message = (
        f"Project Data:\n"
        f"- Location: {project_data.get('location')}\n"
        f"- Tasks: {project_data.get('tasks')}\n"
        f"- Timeline (months): {project_data.get('timeline')}\n"
        f"- Budget: {project_data.get('budget')} crores\n\n"
        f"Please respond with ONLY a single integer or decimal number in the range 1 to 10. "
        f"No extra words. No units. If unsure, pick your best estimate."
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=5,
            temperature=0.2
        )
        text_out = response.choices[0].message.content.strip()

        # Attempt direct float parse
        try:
            value = float(text_out)
        except ValueError:
            # Use a regex to capture any numeric portion
            match = re.search(r"(\d+(\.\d+)?)", text_out)
            if match:
                value = float(match.group(1))
            else:
                return default_value

        # Clamp to [1..10]
        if value < 1: 
            value = 1.0
        elif value > 10:
            value = 10.0

        return value

    except Exception as e:
        print(f"[OpenAI Error] {metric_name}: {e}")
        return default_value


def calculate_strategic_alignment(tasks, location):
    """
    Example rule-based approach to measure strategic alignment (1-10).
    """
    alignment_score = 5.0
    tasks_str = " ".join(t.lower() for t in tasks)

    if any(word in tasks_str for word in ["metro", "rail"]):
        alignment_score += 2.0
    if "Urban" in location:
        alignment_score += 2.0

    if alignment_score > 10:
        alignment_score = 10.0
    return alignment_score


if __name__ == "__main__":
    # Example: multiple projects
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

    # Dynamic weights (modify these as needed)
    dynamic_weights = {
        "w_cost_benefit": 0.25,
        "w_socio_impact": 0.30,
        "w_risk_resilience": 0.20,
        "w_strategic_align": 0.25
    }

    # Retrieve API key from environment or any secure config
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    # Call our function to compute scores for multiple projects
    results = compute_scores_for_multiple_projects(
        projects_example,
        dynamic_weights,
        OPENAI_API_KEY
    )

    # Print or process the results
    print("Project Scoring Results:")
    for res in results:
        print(res)





# {
#   "projectId": "201",
#   "tasks": [
#     "Excavate tunnel sections",
#     "Lay underground tracks",
#     "Install signaling systems",
#     "Construct underground stations",
#     "Fit electrical lines for power supply",
#     "Set up ventilation and fire safety equipment"
#   ],
#   "timeline": 24,
#   "budget": 1800,
#   "location": "Urban Metro Region"
# }


# {
#   "projectId": "202",
#   "tasks": [
#     "Clear and grade existing roadway",
#     "Lay asphalt and expand lanes",
#     "Install drainage and stormwater systems",
#     "Construct pedestrian sidewalks",
#     "Implement traffic signals and signage",
#     "Conduct final safety inspections"
#   ],
#   "timeline": 12,
#   "budget": 450,
#   "location": "Suburban / Semi-urban"
# }


# {
#   "projectId": "203",
#   "tasks": [
#     "Conduct geotechnical surveys and soil testing",
#     "Build foundation piles and piers",
#     "Assemble steel beams and girders",
#     "Pour concrete deck slab",
#     "Install expansion joints and guard rails",
#     "Construct approach roads and ramps"
#   ],
#   "timeline": 18,
#   "budget": 750,
#   "location": "Riverine / Rural Region"
# }


# {
#   "projectId": "204",
#   "tasks": [
#     "Prepare unpaved roads with grading",
#     "Apply gravel and bitumen layers",
#     "Build small culverts for water crossing",
#     "Install solar-powered streetlights",
#     "Add basic signage and speed bumps",
#     "Coordinate with local communities on land access"
#   ],
#   "timeline": 9,
#   "budget": 250,
#   "location": "Remote Rural Area"
# }


# {
#   "projectId": "205",
#   "tasks": [
#     "Deploy underground fiber-optic cables",
#     "Install IoT sensors for traffic management",
#     "Retrofit existing roads with intelligent LED lighting",
#     "Build integrated command & control center",
#     "Upgrade drainage and sewage lines",
#     "Implement digital payment systems at tolls"
#   ],
#   "timeline": 15,
#   "budget": 1200,
#   "location": "Urban Metro Region"
# }


#  {'projectId': '101', 
#   'cost_benefit': 2.4, 
#   'socio_impact': 5.0, 
#   'risk_resilience': 5.0,
#     'strategic_alignment': 7.0, 
#     'priority_score': 4.85}