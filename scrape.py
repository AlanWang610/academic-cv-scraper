from langchain_openai import ChatOpenAI
from browser_use import Agent
import asyncio
import sys
from dotenv import load_dotenv
import requests
import os
import json
from urllib.parse import urlparse
load_dotenv()

name = sys.argv[1]

task = f"""
### Prompt for scraping academic CV from an academic's personal website

**Objective:**
- Visit the personal website (self-hosted or institution-hosted) of {name} and scrape the academic CV. The CV is usually a PDF file.
- Start with search results that contain the phrases "CV" or "Curriculum Vitae" or "Resume" within the first 3 search results.
- It is best to search for the academic's name and the word "curriculum vitae" together.
- If the academic's CV can't be found on the website, try the next result in the search engine, since the institution-hosted website might not have it but another self-hosted one might.
- Visit at most 3 websites. If the CV is not found on the third website, return a null value as the CV URL.

**Output:**
A JSON object with the following fields:
- name: the name of the academic
- cv: the URL of the academic's CV

"""

async def main():
    agent = Agent(
        task=task,
        llm=ChatOpenAI(model="gpt-4o"),
    )
    result = await agent.run()
    
    # Get the last result from the agent's history
    result_data = result.history[-1].result if result.history else None
    print(result_data)
    cv_url = result_data.get('cv') if isinstance(result_data, dict) else None
    
    if cv_url:
        print(f"CV URL: {cv_url}")
        # Download the file
        response = requests.get(cv_url)
        
        # Determine file extension from URL or content-type
        parsed_url = urlparse(cv_url)
        path = parsed_url.path.lower()
        
        if path.endswith('.pdf'):
            ext = '.pdf'
        elif path.endswith('.doc') or path.endswith('.docx'):
            ext = os.path.splitext(path)[1]
        else:
            # Default to HTML if it's a webpage
            ext = '.html'
            
        # Create downloaded_CVs directory if it doesn't exist
        os.makedirs('downloaded_CVs', exist_ok=True)
            
        # Save file with academic's name in the downloaded_CVs directory
        output_file = os.path.join('downloaded_CVs', f"{name}{ext}")
        
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        print(f"File saved as: {output_file}")
    else:
        print("No CV URL found")

    return cv_url

# Usage
cv_url = asyncio.run(main())
