from langchain_openai import ChatOpenAI
from browser_use import Agent
import asyncio
import sys
from dotenv import load_dotenv
import requests
import os
import json
from urllib.parse import urlparse
import re
load_dotenv()

name = sys.argv[1]

task = f"""
### Prompt for scraping academic CV from an academic's personal website

**Objective:**
- Search for {name}'s academic CV, which is usually a PDF file.
- IMPORTANT: If any search result has ALL of these characteristics:
  1. The URL or title contains "cv", "curriculum-vitae", or "resume"
  2. The URL ends in .pdf
  3. It's from a university domain (.edu or .ac.uk) or professional website
  Then return that URL immediately without clicking or visiting the page.

- If no direct PDF link is found:
  - Check the first 5 search results containing "CV", "Curriculum Vitae", or "Resume"
  - Visit each website and look for CV links
  - Try the next result if CV isn't found on current site
  - Stop after checking 5 websites

**Search Strategy:**
1. First try: "{name} curriculum vitae filetype:pdf"
2. Look for the first 5 search results containing "cv", "Curriculum Vitae", or "Resume"
3. If no match, try: "{name} cv site:.edu"
4. Look for the first 5 search results containing "cv", "Curriculum Vitae", or "Resume"
5. If still no match, try regular search with "{name} curriculum vitae"
6. Look for the first 5 search results containing "cv", "Curriculum Vitae", or "Resume"
**Output:**
Return only the direct URL to the CV file (must start with https://).
If no CV is found, return null.

Example good URLs to return immediately:
- https://university.edu/faculty/smith/cv.pdf
- https://department.edu/people/smith_curriculum_vitae.pdf
- https://school.edu/faculty/cv/smith_2023.pdf
"""

async def main():
    agent = Agent(
        task=task,
        llm=ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3
        ),
    )
    result = await agent.run()
    
    # Get the last successful action result
    if result.history[-1]:
        # Get the content from the last history item
        content = str(result.history[-1])
        # Save raw content for debugging
        with open('test.txt', 'w') as f:
            f.write(content)
        # Extract URL from the content by finding text between the delimiters
        start_delimiter = "result=[ActionResult(is_done=True, success=True, extracted_content="
        end_delimiter = ", error=None, include_in_memory=False)]"
        if start_delimiter in content:
            start_idx = content.find(start_delimiter) + len(start_delimiter)
            end_idx = content.find(end_delimiter, start_idx) - 1
            if end_idx != -1:
                # Search for https:// and get everything after it until the end
                https_idx = content[start_idx:end_idx].find('https://')
                if https_idx != -1:
                    cv_url = content[start_idx+https_idx:end_idx].strip()
                else:
                    cv_url = None
            else:
                cv_url = None
        else:
            cv_url = None
        
        # Validate URL format
        if cv_url and cv_url.startswith('https://'):
            try:
                # Test the URL
                response = requests.get(cv_url)
                response.raise_for_status()  # Raise an exception for bad status codes
                
                # Check if it's a PDF, document, or webpage
                content_type = response.headers.get('content-type', '').lower()
                if ('application/pdf' in content_type or 
                    'application/msword' in content_type or 
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document' in content_type or
                    'text/html' in content_type):
                    
                    # Save the file
                    if 'text/html' in content_type:
                        ext = '.html'
                    elif 'pdf' in content_type:
                        ext = '.pdf'
                    else:
                        ext = '.docx'
                        
                    os.makedirs('downloaded_CVs', exist_ok=True)
                    output_file = os.path.join('downloaded_CVs', f"{name}{ext}")
                    
                    with open(output_file, 'wb') as f:
                        f.write(response.content)
                    
                    print(f"File saved as: {output_file}")
                    return cv_url
                
            except requests.RequestException as e:
                print(f"Error downloading CV: {e}")
                return None
        
    print("No valid CV URL found")
    return None

# Usage
cv_url = asyncio.run(main())
print(f"Final CV URL: {cv_url}")
