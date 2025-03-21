from langchain_openai import ChatOpenAI
from browser_use import Agent
import asyncio
import sys
from dotenv import load_dotenv
import requests
import os
import pandas as pd
import time

load_dotenv()

async def scrape_single_cv(name):
    """Async function to scrape a single CV"""
    task = f"""
    ### Prompt for scraping academic CV from an academic's personal website

    **Objective:**
    - Search for {name}'s academic CV, which is usually a PDF file.
    - IMPORTANT: Return URL immediately if either:
      1. Any search result has ALL of these characteristics:
         - URL/title contains "cv", "curriculum-vitae", or "resume"
         - URL ends in .pdf
         - From a university domain (.edu or .ac.uk) or professional website
      2. OR if you find a document that contains typical CV sections like:
         - "Education", "Academic Positions", "Academic Appointments"
         - "Employment", "Research", "Publications"
         - "Teaching", "Professional Experience", "Honors"
         Return that URL immediately without further searching.

    - If no direct match found:
      - Check first 5 search results containing "CV", "Curriculum Vitae", or "Resume"
      - Visit each website and look for CV-like documents
      - Stop and return URL immediately if you find a document with CV sections
      - Try next result if current site doesn't have CV
      - Stop after checking 5 websites

    **Search Strategy:**
    1. First try: "{name} curriculum vitae filetype:pdf"
    2. Look for first 5 results containing "cv", "vita", or "resume"
    3. If no match, try: "{name} cv site:.edu"
    4. Look for first 5 results containing "cv" or typical CV sections
    5. If still no match, try: "{name} curriculum vitae"

    **Output:**
    Return only the direct URL to the CV file (must start with https://).
    If no CV is found, return null.

    Example good URLs to return immediately:
    - https://university.edu/faculty/smith/cv.pdf
    - https://department.edu/people/smith_curriculum_vitae.pdf
    - https://school.edu/faculty/cv/smith_2023.pdf
    """
    agent = Agent(
        task=task,
        llm=ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3
        ),
        ignore_images=True
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

def process_csv_file(input_file):
    """Process all names in the CSV file"""
    try:
        # Read CSV file
        df = pd.read_csv(input_file)
        
        if 'academic_name' not in df.columns:
            raise ValueError("CSV must contain 'academic_name' column")
        
        results = []
        total = len(df)
        
        # Process each name
        for idx, row in df.iterrows():
            name = row['academic_name']
            print(f"\nProcessing {idx + 1}/{total}: {name}")
            
            # Run the async function for each name
            cv_url = asyncio.run(scrape_single_cv(name))
            
            results.append({
                'name': name,
                'cv_url': cv_url,
                'status': 'success' if cv_url else 'failed'
            })
            
            # Add a small delay between requests
            time.sleep(2)
        
        # Save results to a log file
        results_df = pd.DataFrame(results)
        results_df.to_csv('scraping_results.csv', index=False)
        print(f"\nProcessing complete. Results saved to scraping_results.csv")
        
        return results
        
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python scrape.py <path_to_csv_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        sys.exit(1)
        
    results = process_csv_file(input_file)
    if results:
        successful = sum(1 for r in results if r['status'] == 'success')
        total = len(results)
        print(f"Successfully processed {successful}/{total} CVs")

if __name__ == "__main__":
    main()
