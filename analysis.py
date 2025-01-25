import json
from openai import OpenAI
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class AnalysingBuildingData:
    def __init__(self, components_json_path, materials_json_path, openai_api_key, model_name="gpt-4"):
        """
        Initialize the class by reading and combining JSON data from two files:
        - components_json_path
        - materials_json_path

        Provide your OpenAI API key and specify the model to use (default: gpt-4).
        """
        # Set up OpenAI API key and model name
        self.model_name = model_name

        # Read JSON files
        with open(components_json_path, 'r') as components_file:
         
            self.components_data = json.load(components_file)

        with open(materials_json_path, 'r') as materials_file:
            self.materials_data = json.load(materials_file)

        # Combine data for context
        self.combined_data = {
            "components": self.components_data,
            "materials": self.materials_data
        }

        # Prepare the system prompt for the model
        self.system_prompt = """
You are tasked with calculating the material allocation for various civil components based on their dimensions and raw material costs. For each component (e.g., Pillar, Beam, Slab, Foundation), you must calculate how many units of each raw material (e.g., Cement, Steel, Sand, Aggregate) are required, and report these values accurately in a JSON format.

The allocation of materials should be based on standard industry formulas that relate the component's volume to the material consumption for that type of component. Please follow these steps:

1. **Volume Calculation:**
   For each component, calculate its volume using the following formula based on its dimensions:
   - Volume = Height * Width * Length
   - Ensure you calculate the volume in cubic meters (m³).

2. **Material Allocation:**
   Based on the component’s volume, allocate the required amount of each material. For example (adjusting for each component type):
   - **Cement**: ~0.35 m³ per cubic meter of the component
   - **Steel**: ~0.1 m³ per cubic meter of the component
   - **Sand**: ~0.2 m³ per cubic meter of the component (often for slabs/foundations)
   - **Aggregate**: ~0.25 m³ per cubic meter of the component (often for foundations)

3. **Cost per Unit:**
   Each raw material has a cost associated with it per unit. Multiply the allocated amount by the material’s cost to determine the total cost.

4. **Output:**
   The output must be in **strict JSON**. Each component should be a key, and the value should be an object containing the materials and their respective quantities. No additional text outside the JSON. No special formatting or code fences.

Example of the final JSON structure (note: just illustrative):
{
  "Pillar": {
    "Cement": 0.35,
    "Steel": 0.1
  },
  "Beam": {
    "Cement": 0.25,
    "Steel": 0.1
  },
  "Slab": {
    "Cement": 0.3,
    "Steel": 0.2,
    "Sand": 0.4
  },
  "Foundation": {
    "Cement": 0.4,
    "Steel": 0.3,
    "Sand": 0.5,
    "Aggregate": 0.6
  }
}

Strictly use the exact same keys in the input materials. Only return JSON, without any extra symbols or explanations.
"""

    def query(self, user_query):
        """
        Query the model with the combined data as context plus your user query.
        The 'user_query' can contain specific instructions or updated conditions
        for calculation. The model should respond with strict JSON output.
        """

        # Convert the entire combined data to JSON (for context)
        data_as_json = json.dumps(self.combined_data, indent=2)

        # Create the user message with both context and user query
        user_content = f"Components: {data_as_json}, Materials: {user_query}"
        print(user_content)
        # Call the OpenAI ChatCompletion endpoint
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0,
            max_tokens=2000  # Adjust as needed)
        )

        # Return the model's message content (which should be JSON)
        return response.choices[0].message.content

    def get_original_data(self):
        """
        Optionally return the original JSON data if needed.
        """
        return self.combined_data

# import os
# # # Example usage
# if __name__ == "__main__":
#     # Initialize the Building Data Query System
#     building_query = AnalysingBuildingData(
#         "./componentInput.json", 
#         "./materialInput.json", 
#         os.getenv("OPENAI_API_KEY")
#     )

#     # Print original JSON data for reference
#     # print("Original Data:")
#     # print(json.dumps(building_query.get_original_data(), indent=2))

# #     # Query the data
#     query1 = "Pillar"
#     response1 = building_query.query(query1)
#     print("\nQuery 1 Response:", response1)

# print(response1.content)
# print(type(response1.content))
# print(json.dumps(response1.content, indent=2))