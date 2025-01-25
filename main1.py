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
import json
from openai import OpenAI

# Apply nest_asyncio to allow nested event loops in Jupyter notebooks
nest_asyncio.apply()

# Ensure the OpenAI API key is set in the environment variables
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Initialize OpenAI client
client = OpenAI()

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
async def main(
    project_name: str,
    output_json_path: str = "indian_project_data.json",
    annotate_page: bool = False
):
    """
    1. Use LLM to suggest similar Indian projects
    2. Fetch information from Indian sources about these projects
    3. Save the aggregated data to a JSON file
    """
    # Get similar Indian projects
    similar_projects = await fetch_similar_projects_llm(project_name)
    print(f"\nSimilar Indian projects for '{project_name}':")
    for sp in similar_projects:
        print("  -", sp)

    # Combine original + similar projects
    all_projects = [project_name] + similar_projects

    async with async_playwright() as p:
        # Launch browser with larger viewport and non-headless mode for testing
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            viewport={'width': 1280, 'height': 800}
        )
        page = await context.new_page()
        
        # Set longer default timeout
        page.set_default_timeout(60000)
        
        # Optional page annotation
        if annotate_page:
            await page.goto("https://www.google.co.in")
            try:
                annotation_result = await mark_page(page)
                print("Annotation successful. Screenshot captured in base64.")
            except Exception as e:
                print(f"Annotation failed: {e}")

        # Fetch project information from Indian sources
        project_info = await fetch_project_info_from_indian_sources(page, all_projects)

        # Combine everything into final JSON
        data_output = {
            "original_project": project_name,
            "similar_indian_projects": similar_projects,
            "project_details": project_info
        }

        # Save to JSON
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(data_output, f, indent=2, ensure_ascii=False)

        print(f"\nData saved to {output_json_path}")
        return data_output


async def fetch_similar_projects_llm(project_name: str) -> List[str]:
    """
    Calls an LLM to get a short list of similar Indian construction/infrastructure projects
    relevant to 'project_name'.
    """
    if not api_key:
        print("No OpenAI API key; returning mock results for demonstration.")
        return [
            f"{project_name} Phase 2",
            "Delhi Metro Extension",
            "Mumbai Coastal Road",
            "Bangalore Metro Phase 2",
            "Chennai Port-Maduravoyal Corridor"
        ]

    try:
        system_prompt = """You are an expert in Indian infrastructure and construction projects.
Given a project name, generate a short list of similar or related projects from India.
Focus on recent and ongoing projects. Return each project name on a separate line. No extra text."""

        user_prompt = f"Project Name: {project_name}\n\nProvide 5 similar or related Indian projects."

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=256
        )

        reply_content = response.choices[0].message.content.strip()
        projects = [line.strip("-* ") for line in reply_content.split("\n") if line.strip()]
        
        # Ensure we have at least one project
        if not projects:
            return [f"{project_name} Phase 2"]  # Fallback if no projects returned
            
        return projects[:5]  # Limit to 5 projects
        
    except Exception as e:
        print(f"Error getting similar projects: {e}")
        return [f"{project_name} Phase 2"]  # Fallback on error


async def fetch_project_info_from_indian_sources(page: Page, project_names: List[str]) -> List[dict]:
    """
    For each project, search Indian websites for relevant information including:
    - Project cost
    - Timeline
    - Current status
    - Key stakeholders
    """
    results = []
    # Focus on Google search since it's more reliable
    base_url = "https://www.google.co.in/search?q="

    for project in project_names:
        project_info = {
            "project_name": project,
            "cost_info": "",
            "timeline": "",
            "status": "",
            "stakeholders": ""
        }

        try:
            query_string = f"{project} infrastructure project india cost timeline status"
            encoded_query = query_string.replace(" ", "+")
            url = f"{base_url}{encoded_query}"

            # Navigate to Google search
            await page.goto(url, timeout=30000)
            await page.wait_for_load_state("networkidle", timeout=30000)

            # Wait for search results to load
            await page.wait_for_selector("div#search", timeout=5000)

            # Look for relevant information in search results
            snippets = await page.query_selector_all("div.g")
            
            for snippet in snippets:
                try:
                    # Get both the title and description text
                    title_elem = await snippet.query_selector("h3")
                    desc_elem = await snippet.query_selector("div.VwiC3b")
                    
                    if title_elem and desc_elem:
                        title = await title_elem.text_content()
                        desc = await desc_elem.text_content()
                        full_text = (title + " " + desc).lower()
                        
                        # Extract cost information
                        if any(word in full_text for word in ["₹", "crore", "lakh", "budget", "cost"]):
                            cost_match = re.search(r'(?:₹|rs\.?|cost of)\s*[\d,]+(?:\s*(?:crore|lakh|billion))?', full_text)
                            if cost_match:
                                project_info["cost_info"] = cost_match.group(0)
                        
                        # Extract timeline information
                        if any(word in full_text for word in ["complete", "deadline", "schedule", "timeline"]):
                            timeline_match = re.search(r'(?:complete|deadline|schedule|timeline).*?(?:202\d|203\d|by\s+\w+\s+\d{4})', full_text)
                            if timeline_match:
                                project_info["timeline"] = timeline_match.group(0)
                        
                        # Extract status information
                        if any(word in full_text for word in ["status", "progress", "phase", "complete"]):
                            status_match = re.search(r'(?:status|progress|phase|complete).*?[\.\n]', full_text)
                            if status_match:
                                project_info["status"] = status_match.group(0)
                        
                        # Extract stakeholder information
                        if any(word in full_text for word in ["minister", "contractor", "company", "authority"]):
                            stake_match = re.search(r'(?:minister|contractor|company|authority).*?[\.\n]', full_text)
                            if stake_match:
                                project_info["stakeholders"] = stake_match.group(0)

                except Exception as e:
                    print(f"Error processing snippet: {e}")
                    continue

        except Exception as e:
            print(f"Error processing project {project}: {e}")

        # Clean up the extracted information
        for key in project_info:
            if isinstance(project_info[key], str):
                project_info[key] = project_info[key].strip()
                if len(project_info[key]) > 200:
                    project_info[key] = project_info[key][:200] + "..."

        results.append(project_info)

    return results


if __name__ == "__main__":
    asyncio.run(main("Delhi Metro"))