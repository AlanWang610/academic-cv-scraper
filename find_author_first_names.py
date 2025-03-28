import json
import os
import csv
import time
import random
from pathlib import Path
from collections import defaultdict
import openai
import dotenv
import argparse
import pickle

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

def search_for_author_first_name():
    return

class AuthorNameResolver:
    def __init__(self, jsonl_file, progress_file=None):
        """
        Initialize the resolver for a specific JSONL file
        
        Args:
            jsonl_file: Path to the JSONL file to process
            progress_file: Path to save/load progress (defaults to jsonl_file + '.progress')
        """
        self.jsonl_file = Path(jsonl_file)
        self.progress_file = Path(progress_file) if progress_file else self.jsonl_file.with_suffix('.progress')
        self.output_file = self.jsonl_file.with_name(f"{self.jsonl_file.stem}_first_names.jsonl")
        
        # Initialize progress tracking
        self.progress = self._load_progress()
        
        # Count total lines for progress reporting
        self.total_lines = self._count_lines()
        
    def _count_lines(self):
        """Count the total number of lines in the JSONL file"""
        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    
    def _load_progress(self):
        """Load progress from file if it exists"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading progress file: {e}")
                return {'processed_lines': 0, 'name_mappings': {}}
        return {'processed_lines': 0, 'name_mappings': {}}
    
    def _save_progress(self):
        """Save current progress to file"""
        with open(self.progress_file, 'wb') as f:
            pickle.dump(self.progress, f)
        print(f"Progress saved to {self.progress_file}")
    
    def process_file(self, batch_size=10, delay=1):
        """
        Process the JSONL file line by line, resolving initial-only first names
        
        Args:
            batch_size: Number of authors to process in each batch
            delay: Delay between API calls in seconds
        """
        processed_lines = self.progress['processed_lines']
        name_mappings = self.progress['name_mappings']
        
        print(f"Starting from line {processed_lines + 1} of {self.total_lines}")
        print(f"Already resolved {len(name_mappings)} author names")
        print(f"Output will be written to {self.output_file}")
        
        # Create or append to the output file
        output_mode = 'a' if processed_lines > 0 else 'w'
        
        with open(self.jsonl_file, 'r', encoding='utf-8') as f_in, \
             open(self.output_file, output_mode, encoding='utf-8') as f_out:
            
            # Skip already processed lines
            for _ in range(processed_lines):
                next(f_in)
            
            # Process remaining lines
            for line_num, line in enumerate(f_in, processed_lines + 1):
                print(f"\nProcessing line {line_num}/{self.total_lines}")
                
                try:
                    # Process the line and write the updated version to the output file
                    updated_line = self._process_line(line, name_mappings, batch_size, delay)
                    f_out.write(updated_line + '\n')
                    
                    # Update progress
                    self.progress['processed_lines'] = line_num
                    self.progress['name_mappings'] = name_mappings
                    
                    # Save progress after each line
                    self._save_progress()
                    
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    # Save progress before exiting
                    self._save_progress()
                    raise
        
        print(f"\nProcessing complete. Resolved {len(name_mappings)} author names.")
        print(f"Output written to {self.output_file}")
        return name_mappings
    
    def _process_line(self, line, name_mappings, batch_size, delay):
        """
        Process a single line from the JSONL file
        
        Returns:
            Updated JSON line with resolved author names
        """
        try:
            data = json.loads(line)
            
            # Track authors with initial-only first names and their associated works
            initial_authors = defaultdict(set)
            
            # Process main article authors
            if "authors" in data:
                self._process_author_list(data["authors"], data.get("title", ""), initial_authors)
            
            # Process reference authors
            if "references" in data:
                for ref in data["references"]:
                    if "authors" in ref:
                        self._process_author_list(ref["authors"], ref.get("title", ""), initial_authors)
            
            # If we found any initial-only names, resolve them
            if initial_authors:
                print(f"Found {len(initial_authors)} authors with initial-only first names in this line")
                self._resolve_initial_names(initial_authors, name_mappings, batch_size, delay)
                
                # Update the data with resolved names
                if "authors" in data:
                    self._update_author_list(data["authors"], name_mappings)
                
                if "references" in data:
                    for ref in data["references"]:
                        if "authors" in ref:
                            self._update_author_list(ref["authors"], name_mappings)
            else:
                print("No authors with initial-only first names found in this line")
            
            # Return the updated line
            return json.dumps(data)
                
        except json.JSONDecodeError:
            print("Error: Invalid JSON in line")
            return line  # Return the original line if we can't parse it
    
    def _process_author_list(self, author_list, work_title, initial_authors):
        """Process a list of authors and track those with initials"""
        for author in author_list:
            name = format_author_name(author)
            if name:
                # Only store title if first name is an initial
                first = author[0]
                if is_initial_name(first):
                    initial_authors[name].add(work_title)
    
    def _update_author_list(self, author_list, name_mappings):
        """Update author list with resolved names"""
        for author in author_list:
            if len(author) >= 2 and all(author[:2]):
                name = format_author_name(author)
                if name in name_mappings and name_mappings[name] != name:
                    # Split the resolved name back into first and last
                    resolved_parts = name_mappings[name].split()
                    if len(resolved_parts) >= 2:
                        # Update the first name
                        author[0] = resolved_parts[0]
    
    def _resolve_initial_names(self, initial_authors, name_mappings, batch_size, delay):
        """Resolve initial-only first names to full names"""
        # Get the list of authors with initial-only first names
        authors_to_process = [author for author in initial_authors.keys() 
                             if author not in name_mappings]
        
        if not authors_to_process:
            print("All authors in this line have already been resolved")
            return
        
        print(f"Resolving {len(authors_to_process)} new authors")
        
        # Process authors in batches
        for batch_start in range(0, len(authors_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(authors_to_process))
            batch = authors_to_process[batch_start:batch_end]
            
            print(f"\nProcessing batch {batch_start//batch_size + 1}/{(len(authors_to_process) + batch_size - 1)//batch_size}")
            
            for i, author in enumerate(batch):
                print(f"  Processing {batch_start+i+1}/{len(authors_to_process)}: {author}")
                
                # Get the associated works for this author
                works = initial_authors[author]
                
                # Search for the full name
                full_name = search_for_author_first_name(author, works)
                
                # Store the mapping
                name_mappings[author] = full_name
                
                # Add a delay to avoid rate limiting (except for the last author in the batch)
                if i < len(batch) - 1 and delay > 0:
                    time.sleep(delay + random.uniform(0, 0.5))
            
            # Longer delay between batches
            if batch_end < len(authors_to_process):
                print(f"Batch complete. Waiting before next batch...")
                time.sleep(3 + random.uniform(0, 1))

def save_name_mappings_to_csv(name_mappings, output_file):
    """Save the name mappings to a CSV file"""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Original Name', 'Resolved Name', 'Changed'])
        
        for original, resolved in sorted(name_mappings.items()):
            changed = original != resolved
            writer.writerow([original, resolved, changed])
    
    print(f"Name mappings saved to {output_file}")

def apply_name_mappings_to_author_list(author_list_file, name_mappings_file, output_file):
    """
    Apply name mappings to an existing author list
    
    Args:
        author_list_file: CSV file with the original author list
        name_mappings_file: CSV file with name mappings
        output_file: Output CSV file for the updated author list
    """
    # Load name mappings
    name_mappings = {}
    with open(name_mappings_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name_mappings[row['Original Name']] = row['Resolved Name']
    
    # Load and update author list
    updated_authors = set()
    with open(author_list_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            author = row[0]
            updated_author = name_mappings.get(author, author)
            updated_authors.add(updated_author)
    
    # Save updated author list
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Author'])
        for author in sorted(updated_authors):
            writer.writerow([author])
    
    print(f"Updated author list saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Resolve initial-only first names in academic papers')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Process JSONL file command
    process_parser = subparsers.add_parser('process', help='Process a JSONL file to resolve author names')
    process_parser.add_argument('jsonl_file', help='Path to the JSONL file to process')
    process_parser.add_argument('--progress', help='Path to save/load progress (defaults to jsonl_file + .progress)')
    process_parser.add_argument('--output', help='Path to save name mappings CSV (defaults to jsonl_file + .mappings.csv)')
    process_parser.add_argument('--batch-size', type=int, default=10, help='Number of authors to process in each batch')
    process_parser.add_argument('--delay', type=float, default=1, help='Delay between API calls in seconds')
    
    # Apply mappings command
    apply_parser = subparsers.add_parser('apply', help='Apply name mappings to an author list')
    apply_parser.add_argument('author_list', help='Path to the original author list CSV')
    apply_parser.add_argument('name_mappings', help='Path to the name mappings CSV')
    apply_parser.add_argument('output', help='Path to save the updated author list CSV')
    
    args = parser.parse_args()
    
    if args.command == 'process':
        # Process JSONL file
        resolver = AuthorNameResolver(args.jsonl_file, args.progress)
        name_mappings = resolver.process_file(args.batch_size, args.delay)
        
        # Save name mappings to CSV
        output_file = args.output if args.output else Path(args.jsonl_file).with_suffix('.mappings.csv')
        save_name_mappings_to_csv(name_mappings, output_file)
        
    elif args.command == 'apply':
        # Apply name mappings to author list
        apply_name_mappings_to_author_list(args.author_list, args.name_mappings, args.output)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 
