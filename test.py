import os
import asyncio
import base64
import json
import re
from typing import List, Optional
from typing_extensions import TypedDict

import nest_asyncio
from playwright.async_api import async_playwright, Page

# If using LangChain or direct OpenAI calls:
from openai import OpenAI
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")
client = OpenAI(api_key=openai_api_key)
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.runnables import chain as chain_decorator
from langchain_core.output_parsers import StrOutputParser

nest_asyncio.apply()

###############################################################################
# Set up your OpenAI key (optional for the LLM feature)
###############################################################################


###############################################################################
# TypedDict definitions (if you need them for bounding boxes / agent state)
###############################################################################
class BBox(TypedDict):
    x: float
    y: float
    text: str
    type: str
    ariaLabel: str

class Prediction(TypedDict):
    action: str
    args: Optional[List[str]]

class AgentState(TypedDict):
    page: Page
    input: str
    img: str
    bboxes: List[BBox]
    prediction: Prediction
    scratchpad: List[BaseMessage]
    observation: str

###############################################################################
# Optional: mark_page script for bounding box annotation
###############################################################################
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'mark_page.js')
try:
    with open(file_path, 'r') as f:
        mark_page_script = f.read()
except FileNotFoundError:
    # Fallback if mark_page.js not found
    mark_page_script = "function markPage() { return []; }"

@chain_decorator
async def mark_page(page):
    """
    A chain step that executes a JavaScript function in the browser to highlight elements.
    """
    await page.evaluate(mark_page_script)
    bboxes = []
    for _ in range(3):
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

###############################################################################
# 1) Use an LLM to get a list of similar projects
###############################################################################
async def fetch_similar_projects_llm(project_name: str) -> List[str]:
    """
    Calls an LLM (OpenAI, GPT-3.5/4, etc.) to get a short list of similar
    construction projects relevant to 'project_name'.
    """
    if not openai_api_key:
        # If no API key is set, just return mock data
        print("No OpenAI API key; returning mock results for demonstration.")
        return [
            f"{project_name} Phase 2",
            "Example Metro Extension",
            "Highway Overhaul Project",
            "Airport Rail Link",
            "Suburban Rail Upgrade"
        ]

    system_prompt = """You are an expert in large-scale infrastructure projects.
Given a construction project name, generate a short list of similar or related projects.
Return each project name on a separate line. No extra text."""

    user_prompt = f"Project Name: {project_name}\n\nProvide 5 similar or related projects."

    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.2,
    max_tokens=256)

    reply_content = response.choices[0].message.content.strip()
    # Split lines
    projects = [line.strip("-* ") for line in reply_content.split("\n") if line.strip()]
    return projects

###############################################################################
# 2) Crawl approximate cost from Google by going directly to search results
###############################################################################
async def fetch_project_costs_from_google(page: Page, project_names: List[str]) -> List[dict]:
    """
    For each project in project_names, navigate directly to 
    "https://www.google.com/search?q=<PROJECT> cost" and parse a snippet.
    """
    results = []

    for project in project_names:
        # Build the query
        query_string = f"{project} cost".replace(" ", "+")
        google_url = f"https://www.google.com/search?q={query_string}"

        # Navigate directly to the search results page
        await page.goto(google_url, timeout=60000)
        # Wait for the page to load
        await page.wait_for_load_state("networkidle")

        # Attempt to find a snippet in the results
        cost_snippet = "No snippet found"
        try:
            # Common snippet containers on Google
            snippet_elements = await page.query_selector_all("div.BNeawe, div.BNeawe iBp4i")
            snippet_texts = []
            for elem in snippet_elements:
                text_content = await elem.text_content()
                if text_content:
                    snippet_texts.append(text_content.strip())

            # We'll pick the first text that has a currency symbol or mentions cost
            for text in snippet_texts:
                if any(sym in text for sym in ["$", "₹", "£", "€"]) or "cost" in text.lower():
                    cost_snippet = text
                    break
        except Exception as e:
            cost_snippet = f"Failed to parse cost: {e}"

        results.append({
            "project_name": project,
            "approx_cost_info": cost_snippet
        })

    return results

###############################################################################
# Main Function
###############################################################################
async def main(
    project_name: str,
    output_json_path: str = "project_data.json",
    annotate_page: bool = False
):
    """
    1. Use an LLM to suggest similar projects for the given project_name.
    2. Use Playwright to scrape approximate cost info for the original project + similar projects.
    3. Optionally annotate the page for demonstration (bounding boxes & screenshot).
    4. Save the aggregated data to a JSON file.
    """

    # 1) (Optional) Get a list of similar projects from the LLM
    #    If you only want data on the original project, you can skip this step.
    similar_projects = await fetch_similar_projects_llm(project_name)
    print(f"Similar projects for '{project_name}':")
    for sp in similar_projects:
        print("  -", sp)

    # Combine original + similar so we fetch them all
    all_projects = [project_name] + similar_projects

    # 2) Launch browser and fetch approximate costs
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Optional step: annotate google.com with bounding boxes
        if annotate_page:
            await page.goto("https://www.google.com")
            try:
                annotation_result = await mark_page(page)
                print("Annotation successful. Screenshot captured in base64.")
            except Exception as e:
                print(f"Annotation failed: {e}")

        # Fetch approximate cost for each project (including the original project)
        project_costs = await fetch_project_costs_from_google(page, all_projects)

        await browser.close()

    # 3) Combine everything into a final JSON structure
    data_output = {
        "original_project": project_name,
        "similar_projects": similar_projects,
        "cost_data": project_costs
    }

    # 4) Save to JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data_output, f, indent=2, ensure_ascii=False)

    print(f"\nData saved to: {output_json_path}")
    return data_output

###############################################################################
# If you run this module directly:
###############################################################################
if __name__ == "__main__":
    # Example usage:
    project_name_input = "metro construction"
    asyncio.run(main(project_name_input, annotate_page=False))
