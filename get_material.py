from langchain.llms import OpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, Tool
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, chain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.tools import TavilySearchResults
from pydantic import BaseModel
from typing import List
import tavily

import os
import getpass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")
    

tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)


if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
llm = ChatOpenAI(model="gpt-4o")


class MaterialBase(BaseModel):
    material: str
    price: int

class MaterialResponse(BaseModel):
    data: list[MaterialBase]

prompt = ChatPromptTemplate(
    [
         ("system", """You are an expert construction planner with access to a vast knowledge base and real-time web search capabilities. The response must be in below JSON format strictly without any explanation and comments:
        {{
            "data": [
                {{
                    "material": "Material 1", 
                    "price": 1000
                }},
                {{
                    "material": "Material 2", 
                    "price": 1500
                }},
                {{
                    "material": "Material 3", 
                    "price": 2000
                }}
                // Add more materials and their prices as needed
            ]
        }}"""),
        ("human", "{user_input}"),
        ("placeholder", "{messages}"),
    ]
)


llm_with_tools = llm.bind_tools([tool])
llm_chain = prompt | llm_with_tools
parser = PydanticOutputParser(pydantic_object=MaterialResponse)

@chain
def tool_chain(user_input: str, config: RunnableConfig):
    input_ = {"user_input": user_input}
    ai_msg = llm_chain.invoke(input_, config=config)
    tool_msgs = tool.batch(ai_msg.tool_calls, config=config)
    response = llm_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]}, config=config)
    return response

    

def get_materials(user_input:str):
    response = tool_chain.invoke(user_input)
    parsed_response = parser.parse(response.content)
    return parsed_response.model_dump()
