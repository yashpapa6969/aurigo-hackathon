import os
import asyncio
import base64
from typing import List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, SystemMessage
from playwright.async_api import async_playwright, Page
from langchain_core.runnables import chain as chain_decorator
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
import nest_asyncio
import re

# Apply nest_asyncio to allow nested event loops in Jupyter notebooks
nest_asyncio.apply()

# Ensure the OpenAI API key is set in the environment variables
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Define TypedDicts for bounding boxes and predictions
from typing import List, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from playwright.async_api import Page


class BBox(TypedDict):
    x: float
    y: float
    text: str
    type: str
    ariaLabel: str


class Prediction(TypedDict):
    action: str
    args: Optional[List[str]]


# This represents the state of the agent
# as it proceeds through execution
class AgentState(TypedDict):
    page: Page  # The Playwright web page lets us interact with the web environment
    input: str  # User request
    img: str  # b64 encoded screenshot
    bboxes: List[BBox]  # The bounding boxes from the browser annotation function
    prediction: Prediction  # The Agent's output
    # A system message (or messages) containing the intermediate steps
    scratchpad: List[BaseMessage]
    observation: str  # The most recent response from a tool
# Load the JavaScript for marking the page
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'mark_page.js')
with open(file_path, 'r') as f:
    mark_page_script = f.read()

# Chain decorator to mark the page
@chain_decorator
async def mark_page(page):
    await page.evaluate(mark_page_script)
    for _ in range(10):
        try:
            bboxes = await page.evaluate("markPage()")
            break
        except Exception as e:
            print(f"Error during page evaluation: {e}")
            await asyncio.sleep(3)
    screenshot = await page.screenshot()
    await page.evaluate("unmarkPage()")
    return {
        "img": base64.b64encode(screenshot).decode(),
        "bboxes": bboxes,
    }

# Main execution
async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto("https://www.google.com")
        
        # Example logic for marking page
        try:
            result = await mark_page(page)
            print("Bounding boxes and screenshot captured!")
        except Exception as e:
            print(f"Failed during page annotation: {e}")

if __name__ == "__main__":
    asyncio.run(main())
