import chromadb
from langchain_chroma import Chroma
import json
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import uuid
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

class AnalysingBuildingData:
    def __init__(self, components_json_path, materials_json_path, openai_api_key, model_name="gpt-4"):
        openai.api_key = openai_api_key
        
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
        
        # Convert combined JSON to text chunks
        self.text_chunks = self._convert_json_to_text_chunks()
        
        # Initialize models
        self.llm = ChatOpenAI(
            model_name="gpt-4o", 
            temperature=0, 
            openai_api_key=openai_api_key, 
            max_tokens=4060
        )
        self.embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Initialize Chroma client and collection
        self.client = chromadb.Client()
        self.collection_name = 'building_data'
        self.collection = self.client.create_collection(self.collection_name)
        
        # Process data and create vector store
        self._process_data()
        self.vectorstore = Chroma(
            collection_name=self.collection_name, 
            client=self.client, 
            embedding_function=self.embedding_model
        )
        
        # Create RAG chain
        self.rag_chain = self._create_rag_chain()
    
    def _convert_json_to_text_chunks(self, chunk_size=500):
        """
        Convert combined JSON to text chunks for embedding
        """
        json_text = json.dumps(self.combined_data, indent=2)
        return [json_text[i:i + chunk_size] for i in range(0, len(json_text), chunk_size)]
    
    def _process_data(self):
        """
        Process data by embedding text chunks
        """
        for i, chunk in enumerate(self.text_chunks, 1):
            print(f"Processing chunk {i}")
            embedding = self.embedding_model.embed_documents([chunk])[0]
            document_id = str(uuid.uuid4())
            
            self.collection.add(
                documents=[chunk],
                metadatas=[{"content": chunk}],
                embeddings=[embedding],
                ids=[document_id]
            )
    
    def _create_rag_chain(self):
        """
        Create the Retrieval-Augmented Generation (RAG) chain using ChatPromptTemplate
        """
        # Custom ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are tasked with calculating the material allocation for various civil components based on their dimensions and raw material costs. For each component (e.g., Pillar, Beam, Slab, Foundation), you must calculate how many units of each raw material (e.g., Cement, Steel, Sand) are required, and report these values accurately in a JSON format. 
    
    The allocation of materials should be based on standard industry formulas that relate the component's volume to the material consumption for that type of component. Please follow these steps:

    1. **Volume Calculation:**
       For each component, calculate its volume using the following formula based on its dimensions:
       - Volume = Height * Width * Length
       - Ensure you calculate the volume in cubic meters (m³).

    2. **Material Allocation:**
       Based on the component’s volume, you need to allocate the required amount of each material. The raw material requirements should be determined as follows:
       - **Cement**: For a typical concrete component, approximately 0.35 cubic meters of cement is used per cubic meter of volume (adjust based on the component type).
       - **Steel**: For structural components like beams and pillars, approximately 0.1 cubic meters of steel is used per cubic meter of volume.
       - **Sand**: For slab and foundation components, approximately 0.2 cubic meters of sand is used per cubic meter of volume.
       - **Aggregate**: Similar to sand, foundation components may use approximately 0.25 cubic meters of aggregate per cubic meter of volume.

    3. **Cost per Unit:**
       Each raw material has a cost associated with it per unit (e.g., per cubic meter or kilogram). For each material, multiply the allocated amount by the material's cost to determine the total cost.

    4. **Output:**
       The output should be in strict JSON format. Each component should be a key, and the value should be an object containing the materials and their respective quantities. Each quantity should be calculated in the appropriate units (e.g., cubic meters, kilograms) for that component.

    Example Input:

    **Components:**
    [
        {{ "component": "Pillar", "dimensions": {{ "height": 4, "width": 0.5, "length": 0.5 }}}},
        {{ "component": "Beam", "dimensions": {{ "height": 0.3, "width": 0.4, "length": 6 }}}},
        {{ "component": "Slab", "dimensions": {{ "height": 0.2, "width": 6, "length": 6 }}}},
        {{ "component": "Foundation", "dimensions": {{ "height": 1.5, "width": 6, "length": 6 }}}}
    ]

    **Materials (with cost per unit):**
    [
        {{ "material": "Cement", "price_per_unit": 3500 }},  # per cubic meter
        {{ "material": "Steel", "price_per_unit": 60000 }},  # per cubic meter
        {{ "material": "Sand", "price_per_unit": 50000 }},   # per cubic meter
        {{ "material": "Aggregate", "price_per_unit": 40000 }}  # per cubic meter
    ]

    **Calculation:**
    - For **Pillar**, the volume is calculated as:
      Volume = 4 * 0.5 * 0.5 = 1 cubic meter.
      The materials allocation based on this volume will be:
      Cement: 0.35 cubic meters (for 1 cubic meter of Pillar).
      Steel: 0.1 cubic meters.
      
    - For **Beam**, the volume is:
      Volume = 0.3 * 0.4 * 6 = 0.72 cubic meters.
      The material allocation will be:
      Cement: 0.25 cubic meters.
      Steel: 0.1 cubic meters.

    - For **Slab**, the volume is:
      Volume = 0.2 * 6 * 6 = 7.2 cubic meters.
      The material allocation will be:
      Cement: 0.3 cubic meters.
      Steel: 0.2 cubic meters.
      Sand: 0.4 cubic meters.
      
    - For **Foundation**, the volume is:
      Volume = 1.5 * 6 * 6 = 54 cubic meters.
      The material allocation will be:
      Cement: 0.4 cubic meters.
      Steel: 0.3 cubic meters.
      Sand: 0.5 cubic meters.
      Aggregate: 0.6 cubic meters.

    **Output Format Example (in JSON):**
    {{
        "Pillar": {{
            "Cement": 0.35,
            "Steel": 0.1
        }},
        "Beam": {{
            "Cement": 0.25,
            "Steel": 0.15
        }},
        "Slab":{{
            "Cement": 0.3,
            "Steel": 0.2,
            "Sand": 0.4
        }},
        "Foundation": {{
            "Cement": 0.4,
            "Steel": 0.3,
            "Sand": 0.5,
            "Aggregate": 0.6
        }}
    }}

    Please ensure that the response is only in JSON format without any additional explanation or text no special characters NO MATTER WHAT dont give ```json.
    """),
    ("human", "Context: {context}")
])

        
        # Create the RAG chain using the prompt
        rag_chain = (
            {"context": self.vectorstore.as_retriever(search_kwargs={"k": 3}), 
             "question": RunnablePassthrough()}
            | prompt 
            | self.llm
        )
        
        return rag_chain

    def query(self, query):
        """
        Query the building data
        """
        return self.rag_chain.invoke(query)

    def get_original_data(self):
        """
        Return the original JSON data
        """
        return self.combined_data
import os

# Example usage
if __name__ == "__main__":
    # Initialize the Building Data Query System
    building_query = AnalysingBuildingData(
        "./componentInput.json", 
        "./materialInput.json", 
        os.getenv("OPENAI_API_KEY")
    )
    
    # Print original JSON data for reference
    # print("Original Data:")
    # print(json.dumps(building_query.get_original_data(), indent=2))
    
    # Query the data
    query1 = ""
    response1 = building_query.query(query1)
    print("\nQuery 1 Response:", response1.content)
    
print(response1.content)
print(type(response1.content))
print(json.dumps(response1.content, indent=2))