import json
import os
import csv
import time
import random
from pathlib import Path
from collections import defaultdict
import openai
import dotenv

# Load environment variables
dotenv.load_dotenv()

# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def is_initial_name(name):
    """Check if name appears to be just an initial (with or without period)"""
    # Remove any periods and spaces
    cleaned = name.replace('.', '').replace(' ', '')
    return len(cleaned) == 1

def normalize_name_case(name):
    """Convert name to proper case if it appears to be all caps"""
    if name.isupper():
        return name.title()
    return name

def format_author_name(author):
    """Format author name as 'first last' if both parts exist"""
    if not author or len(author) < 2 or not all(author[:2]):
        return None
    first, last = author[:2]
    # Normalize case for each part
    first = normalize_name_case(first)
    last = normalize_name_case(last)
    return f"{first} {last}"

def extract_authors_from_line(line, initial_authors_data):
    """Extract author names from a single JSONL line and track initial-only names"""
    try:
        data = json.loads(line)
        authors = set()

        def process_author_list(author_list, work_title):
            """Process a list of authors and track those with initials"""
            for author in author_list:
                name = format_author_name(author)
                if name:
                    authors.add(name)
                    # Only store title if first name is an initial
                    first = author[0]
                    if is_initial_name(first):
                        initial_authors_data[name].add(work_title)

        # Process main article authors
        if "authors" in data:
            process_author_list(data["authors"], data.get("title", ""))

        # Process reference authors
        if "references" in data:
            for ref in data["references"]:
                if "authors" in ref:
                    process_author_list(ref["authors"], ref.get("title", ""))

        return authors

    except json.JSONDecodeError:
        return set()

def build_author_list(jsonl_file, initial_authors_data):
    """Build a set of unique author names from a JSONL file"""
    all_authors = set()
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_authors = extract_authors_from_line(line, initial_authors_data)
            all_authors.update(line_authors)
            
    return all_authors

def build_author_list_from_folder(folder_path):
    """Build a set of unique author names from all JSONL files in a folder"""
    all_authors = set()
    initial_authors_data = defaultdict(set)  # Maps author names to set of associated work titles
    folder = Path(folder_path)
    
    # Find all .jsonl files in the folder
    jsonl_files = folder.glob('*.jsonl')
    
    for file in jsonl_files:
        try:
            file_authors = build_author_list(file, initial_authors_data)
            all_authors.update(file_authors)
            print(f"Processed {file.name}, found {len(file_authors)} authors")
        except Exception as e:
            print(f"Error processing {file.name}: {str(e)}")
    
    return all_authors, initial_authors_data

def search_for_author_first_name(author_name, associated_works):
    """
    Search for the full first name of an author using the OpenAI API
    
    Args:
        author_name: The author name with initial (e.g., "A. Smith")
        associated_works: List of works associated with this author
    
    Returns:
        Full author name if found, or original name if not found
    """
    # Extract the initial and last name
    parts = author_name.split()
    if len(parts) < 2:
        return author_name  # Can't process without at least two parts
    
    initial = parts[0].replace('.', '')  # Remove period if present
    last_name = parts[-1]
    
    # Select up to 3 work titles to use in the search
    work_titles = list(associated_works)[:3]
    if not work_titles:
        return author_name  # No work titles to search with
    
    # Create a prompt for the API
    prompt = f"""
    I need to find the full first name of an academic author who appears with just an initial.
    
    Author with initial: {author_name}
    
    Associated works:
    {', '.join(f'"{title}"' for title in work_titles)}
    
    Please search for this author's full first name based on these works. Return ONLY the full name in the format "FirstName LastName" if you find it with high confidence.
    
    If you can't find the full name with high confidence, return exactly the original name: "{author_name}".
    
    Do not include any explanations or additional text, just return the name.
    """
    
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a research assistant that helps find full names of academic authors. You have access to search the web to find information."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract the full name from the response
        full_name = response.choices[0].message.content.strip()
        
        # Validate the response - it should contain the last name
        if last_name.lower() in full_name.lower() and full_name != author_name:
            print(f"Found full name for {author_name}: {full_name}")
            return full_name
        else:
            print(f"Could not find reliable full name for {author_name}")
            return author_name
            
    except Exception as e:
        print(f"Error searching for full name of {author_name}: {str(e)}")
        return author_name

def resolve_initial_names(authors, initial_authors_data, batch_size=10, delay=1):
    """
    Automatically resolve initial-only first names to full names
    
    Args:
        authors: Set of all author names
        initial_authors_data: Dictionary mapping author names to their associated works
        batch_size: Number of authors to process in each batch
        delay: Delay between API calls in seconds
    
    Returns:
        Updated set of author names with resolved full names
    """
    updated_authors = set(authors)  # Start with a copy of the original set
    
    # Get the list of authors with initial-only first names
    initial_authors = list(initial_authors_data.keys())
    total_authors = len(initial_authors)
    
    if total_authors == 0:
        print("No authors with initial-only first names found.")
        return updated_authors
    
    print(f"Automatically resolving full names for {total_authors} authors...")
    
    # Process authors in batches
    for batch_start in range(0, total_authors, batch_size):
        batch_end = min(batch_start + batch_size, total_authors)
        batch = initial_authors[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//batch_size + 1}/{(total_authors + batch_size - 1)//batch_size}")
        print(f"Authors {batch_start+1}-{batch_end} of {total_authors}")
        
        for i, author in enumerate(batch):
            print(f"  Processing {batch_start+i+1}/{total_authors}: {author}")
            
            # Get the associated works for this author
            works = initial_authors_data[author]
            
            # Search for the full name
            full_name = search_for_author_first_name(author, works)
            
            # If we found a different name, update the set
            if full_name != author:
                updated_authors.remove(author)
                updated_authors.add(full_name)
            
            # Add a delay to avoid rate limiting (except for the last author in the batch)
            if i < len(batch) - 1 and delay > 0:
                time.sleep(delay + random.uniform(0, 0.5))
        
        # Save progress after each batch
        save_progress(updated_authors, f"authors_resolved_batch_{batch_start//batch_size + 1}.csv")
        
        # Longer delay between batches
        if batch_end < total_authors:
            print(f"Batch complete. Waiting before next batch...")
            time.sleep(5 + random.uniform(0, 2))
    
    return updated_authors

def save_progress(authors, output_file):
    """Save the current progress to a CSV file"""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Author'])
        for author in sorted(authors):
            writer.writerow([author])
    print(f"Progress saved to {output_file}")

def save_authors_to_csv(authors, initial_authors_data, output_file):
    """Save the set of authors to a CSV file, marking those with initial-only first names"""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Author', 'Needs_Full_Name', 'Associated_Works'])  # Header
        for author in sorted(authors):
            needs_full_name = author in initial_authors_data
            associated_works = '; '.join(initial_authors_data.get(author, [])) if needs_full_name else ''
            writer.writerow([author, needs_full_name, associated_works])
    print(f"\nSaved {len(authors)} authors to {output_file}")

if __name__ == "__main__":
    folder_path = r"C:\Users\wangac\Documents\Tasks\Finance Academia Scraping\citation_gatherer\scraped_articles"
    output_file = "authors.csv"
    final_output = "authors_resolved.csv"
    
    # Build the initial author list
    authors, initial_authors_data = build_author_list_from_folder(folder_path)
    print(f"\nFound {len(authors)} unique authors total")
    print(f"Found {len(initial_authors_data)} authors with initial-only first names")
    
    # Save the initial list with flags for names that need resolution
    save_authors_to_csv(authors, initial_authors_data, output_file)
    
    # Automatically resolve initial names
    updated_authors = resolve_initial_names(authors, initial_authors_data)
    
    # Save the final resolved list
    with open(final_output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Author'])  # Simpler header for resolved list
        for author in sorted(updated_authors):
            writer.writerow([author])
    print(f"\nSaved {len(updated_authors)} authors to {final_output}")
    
    # Print summary of authors that couldn't be resolved
    unresolved = [author for author in initial_authors_data if author in updated_authors]
    if unresolved:
        print(f"\n{len(unresolved)} authors could not be resolved:")
        for author in sorted(unresolved):
            print(f"  - {author}")
