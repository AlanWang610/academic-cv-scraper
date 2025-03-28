import json
import os
import csv
from pathlib import Path

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

def extract_authors_from_line(line):
    """Extract author names from a single JSONL line"""
    try:
        data = json.loads(line)
        authors = set()

        def process_author_list(author_list):
            """Process a list of authors"""
            for author in author_list:
                name = format_author_name(author)
                if name:
                    authors.add(name)

        # Process main article authors
        if "authors" in data:
            process_author_list(data["authors"])

        # Process reference authors
        if "references" in data:
            for ref in data["references"]:
                if "authors" in ref:
                    process_author_list(ref["authors"])

        return authors

    except json.JSONDecodeError:
        return set()

def build_author_list(jsonl_file):
    """Build a set of unique author names from a JSONL file"""
    all_authors = set()
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line_authors = extract_authors_from_line(line)
            all_authors.update(line_authors)
            
    return all_authors

def build_author_list_from_folder(folder_path):
    """Build a set of unique author names from all JSONL files in a folder"""
    all_authors = set()
    folder = Path(folder_path)
    
    # Find all .jsonl files in the folder
    jsonl_files = folder.glob('*.jsonl')
    
    for file in jsonl_files:
        try:
            file_authors = build_author_list(file)
            all_authors.update(file_authors)
            print(f"Processed {file.name}, found {len(file_authors)} authors")
        except Exception as e:
            print(f"Error processing {file.name}: {str(e)}")
    
    return all_authors

def save_authors_to_csv(authors, output_file):
    """Save the set of authors to a CSV file"""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Author'])  # Header
        for author in sorted(authors):
            writer.writerow([author])
    print(f"\nSaved {len(authors)} authors to {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Build a list of authors from JSONL files')
    parser.add_argument('folder_path', help='Path to folder containing JSONL files')
    parser.add_argument('--output', default='authors.csv', help='Output CSV file (default: authors.csv)')
    
    args = parser.parse_args()
    
    # Build the author list
    authors = build_author_list_from_folder(args.folder_path)
    print(f"\nFound {len(authors)} unique authors total")
    
    # Save the author list
    save_authors_to_csv(authors, args.output)
    
    print("\nTo resolve initial-only first names, use find_author_first_names.py")
