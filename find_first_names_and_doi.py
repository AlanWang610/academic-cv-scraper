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
from langchain_openai import ChatOpenAI
from browser_use import Agent
import asyncio
from datetime import datetime, timedelta

# Load environment variables
dotenv.load_dotenv()

# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def is_initial_name(name):
    """Check if name appears to be just an initial (with or without period)"""
    # Remove any periods and spaces
    cleaned = name.replace('.', '').replace(' ', '')
    
    # Check if it's a single letter (like "J") - must be capitalized
    if len(cleaned) == 1 and cleaned.isupper():
        return True
    
    # Check if it's two-letter initials (like "JE" or "J.E.") - must be capitalized
    if len(cleaned) == 2 and cleaned.isupper():
        return True
    
    # Check if it's initials with periods (like "J.E." or "J. E.") - first letter must be capitalized
    if '.' in name and name[0].isupper():
        # Count the number of periods - if it's similar to the number of characters, it's likely initials
        period_count = name.count('.')
        letter_count = sum(1 for c in name if c.isalpha())
        # Cap to 2 letters with periods (like "J.E.")
        if period_count >= letter_count - 1 and letter_count <= 2:
            return True
    
    return False

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

async def search_for_author_first_name(author_names, doi, non_doi_json):
    """Function to search for the full first name of a list of authors"""
    if doi:
        task = f"""
        ### Prompt for finding author first names when only the first initials are given, DOI is provided

        **Objective:**
        - Use the direct DOI using the link: https://doi.org/{doi}
        - Find the first names of all the authors of the paper

        **Important Guidelines:**
        - If the DOI link works, use the information from that page
        - If you encounter a paywall or captcha, immediately try a Google search instead
        - Be satisfied with the author information you find on the first accessible page
        - Do not spend time visiting multiple pages if you already found the authors

        **Output Format:**
        You MUST return your answer EXACTLY in this format with square brackets and commas as shown:
        "[first_name1, last_name1], [first_name2, last_name2], [first_name3, last_name3]"

        For example:
        "[John, Smith], [Mary, Jones], [Robert, Williams]"

        DO NOT deviate from this format or add any additional text.
        If no author first names are found, return:
        "No author first names found for {author_names}"
        """
    else:
        task = f"""
        ### Prompt for finding author first names when only the first initials are given

        **Objective:**
        - Use the details here to find the paper online: {non_doi_json}
        - Find the first names of all the authors of the paper
        
        **Important Guidelines:**
        - Avoid PDF files when possible as they are harder to parse
        - Be satisfied with the author information you find on the first accessible page
        - If you encounter a paywall or captcha, immediately try a different search result
        - Do not visit more than 2 pages total - prioritize quality over completeness
        - It's better to return partial results than to spend too much time searching

        **Output Format:**
        You MUST return your answer EXACTLY in this format with square brackets and commas as shown:
        "[first_name1, last_name1], [first_name2, last_name2], [first_name3, last_name3]"

        For example:
        "[John, Smith], [Mary, Jones], [Robert, Williams]"

        DO NOT deviate from this format or add any additional text.
        If no author first names are found, return:
        "No author first names found for {author_names}"
        """
    agent = Agent(
        task=task,
        llm=ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
        ),
    )
    result = await agent.run()
    # Get the last successful action result
    if result and result.history and result.history[-1]:
        # Get the content from the last history item
        content = str(result.history[-1])
        # Extract URL from the content by finding text between the delimiters
        start_delimiter = "result=[ActionResult(is_done=True, success=True, extracted_content="
        end_delimiter = ", error=None, include_in_memory=False)]"
        if start_delimiter in content:
            start_idx = content.find(start_delimiter) + len(start_delimiter)
            end_idx = content.find(end_delimiter, start_idx) - 1
            if end_idx != -1:
                # Extract the text in the range
                extracted_text = content[start_idx:end_idx]
                return extracted_text
            else:
                return None
        else:
            return None
    return None

class AuthorNameResolver:
    def __init__(self, jsonl_file, progress_file=None):
        """
        Initialize the resolver for a specific JSONL file
        
        Args:
            jsonl_file: Path to the JSONL file to process
            progress_file: Path to save/load progress (defaults to jsonl_file + '.progress')
        """
        self.jsonl_file = Path(jsonl_file)
        self.output_file = self.jsonl_file.with_name(f"{self.jsonl_file.stem}_first_names.jsonl")
        
        # Initialize name mappings
        self.name_mappings = {}
        
        # Set to track processed DOIs
        self.processed_dois = set()
        
        # Load already processed DOIs from output file
        self._load_processed_dois()
        
        # Count total lines for progress reporting
        self.total_lines = self._count_lines()
        
        # Track the last time we made a browser-use call
        self.last_browser_call = None
        # Minimum time between browser calls (60 seconds)
        self.min_browser_interval = 60
        
    def _count_lines(self):
        """Count the total number of lines in the JSONL file"""
        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    
    def _load_processed_dois(self):
        """Load DOIs from the output file to track what's already been processed"""
        if not self.output_file.exists():
            print("No existing output file found. Starting from scratch.")
            return
        
        print(f"Loading processed DOIs from {self.output_file}")
        processed_count = 0
        
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        # Extract DOI if it exists
                        if "doi" in data and data["doi"]:
                            self.processed_dois.add(data["doi"])
                            processed_count += 1
                    except json.JSONDecodeError:
                        # Skip invalid lines
                        continue
        
            print(f"Loaded {len(self.processed_dois)} unique DOIs from output file")
            print(f"Found {processed_count} articles with DOIs out of {self._count_output_lines()} total lines")
        except Exception as e:
            print(f"Error loading processed DOIs: {e}")
    
    def _count_output_lines(self):
        """Count lines in the output file"""
        if not self.output_file.exists():
            return 0
        
        with open(self.output_file, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    
    async def process_file(self, batch_size=10, delay=1):
        """
        Process the JSONL file line by line, resolving initial-only first names
        
        Args:
            batch_size: Number of authors to process in each batch
            delay: Delay between API calls in seconds
        """
        print(f"Starting processing of {self.total_lines} lines")
        print(f"Already processed {len(self.processed_dois)} unique DOIs")
        print(f"Output will be written to {self.output_file}")
        
        # Create or append to the output file
        output_mode = 'a' if self.output_file.exists() else 'w'
        print(f"Opening output file in '{output_mode}' mode")
        
        # Check if output directory exists
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir)
        
        # Check if we have write permissions
        try:
            with open(self.output_file, output_mode, encoding='utf-8') as test_file:
                test_file.write("")
            print(f"Successfully opened output file for writing")
        except Exception as e:
            print(f"ERROR: Cannot write to output file: {e}")
            raise
        
        lines_processed = 0
        lines_written = 0
        new_dois_processed = 0
        
        try:
            with open(self.jsonl_file, 'r', encoding='utf-8') as f_in, \
                 open(self.output_file, output_mode, encoding='utf-8') as f_out:
                
                # Process all lines
                for line_num, line in enumerate(f_in, 1):
                    lines_processed += 1
                    
                    # Check if this line has a DOI we've already processed
                    skip_line = False
                    try:
                        data = json.loads(line)
                        if "doi" in data and data["doi"] and data["doi"] in self.processed_dois:
                            print(f"\nSkipping line {line_num}/{self.total_lines} - DOI already processed: {data['doi']}")
                            skip_line = True
                    except json.JSONDecodeError:
                        # Not valid JSON, we'll process it anyway
                        pass
                    
                    if skip_line:
                        continue
                    
                    print(f"\nProcessing line {line_num}/{self.total_lines}")
                    
                    try:
                        # Process the line and write the updated version to the output file
                        updated_line = await self._process_line(line, self.name_mappings, batch_size, delay)
                        
                        # Debug: check if updated_line is valid
                        if updated_line:
                            print(f"Writing updated line ({len(updated_line)} chars) to output file")
                            f_out.write(updated_line + '\n')
                            f_out.flush()  # Force write to disk
                            lines_written += 1
                            
                            # Track the DOI if it exists
                            try:
                                updated_data = json.loads(updated_line)
                                if "doi" in updated_data and updated_data["doi"]:
                                    self.processed_dois.add(updated_data["doi"])
                                    new_dois_processed += 1
                            except json.JSONDecodeError:
                                pass
                        else:
                            print("WARNING: Updated line is empty, not writing to output file")
                        
                    except Exception as e:
                        print(f"Error processing line {line_num}: {e}")
                        raise
            
            print(f"\nProcessing complete.")
            print(f"Lines processed: {lines_processed}")
            print(f"Lines written: {lines_written}")
            print(f"New DOIs processed: {new_dois_processed}")
            print(f"Total unique DOIs processed: {len(self.processed_dois)}")
            print(f"Names resolved: {len(self.name_mappings)}")
            print(f"Output written to {self.output_file}")
            
            # Verify the output file exists and has content
            if os.path.exists(self.output_file):
                file_size = os.path.getsize(self.output_file)
                print(f"Output file size: {file_size} bytes")
                if file_size == 0:
                    print("WARNING: Output file exists but is empty!")
            else:
                print("ERROR: Output file does not exist after processing!")
            
            return self.name_mappings
        
        except Exception as e:
            print(f"Fatal error during processing: {e}")
            raise
    
    async def _process_line(self, line, name_mappings, batch_size, delay):
        """
        Process a single line from the JSONL file
        
        Returns:
            Updated JSON line with resolved author names
        """
        try:
            data = json.loads(line)
            
            # Check if this is an error line
            if "llm_parsing_error" in data or "error" in data:
                print("Skipping error line:", data.get("llm_parsing_error", data.get("error", "Unknown error")))
                return line  # Return the original line without processing
            
            # Store the current work for reference in other methods
            self.current_work = data
            
            # Track works with their associated authors who have initial-only first names
            works_with_initial_authors = {}
            
            # Process main article authors
            main_authors_updated = False
            if "authors" in data:
                main_work = {
                    "title": data.get("title", ""),
                    "doi": data.get("doi"),
                    "authors": data["authors"]
                }
                initial_authors_in_work = self._collect_initial_authors(data["authors"])
                if initial_authors_in_work:
                    works_with_initial_authors[json.dumps(main_work)] = initial_authors_in_work
            
            # Process reference authors
            refs_updated = []
            if "references" in data:
                for i, ref in enumerate(data["references"]):
                    if "authors" in ref:
                        ref_work = {
                            "title": ref.get("title", ""),
                            "doi": ref.get("doi"),
                            "authors": ref["authors"]
                        }
                        initial_authors_in_ref = self._collect_initial_authors(ref["authors"])
                        if initial_authors_in_ref:
                            works_with_initial_authors[json.dumps(ref_work)] = initial_authors_in_ref
            
            # If we found any works with initial-only names, resolve them
            if works_with_initial_authors:
                total_initial_authors = sum(len(authors) for authors in works_with_initial_authors.values())
                print(f"Found {total_initial_authors} authors with initial-only first names across {len(works_with_initial_authors)} works")
                await self._resolve_works_with_initial_names(works_with_initial_authors, name_mappings, batch_size, delay)
                
                # Update the data with resolved names
                if "authors" in data:
                    main_authors_updated = self._update_author_list(data["authors"], name_mappings)
                    if main_authors_updated:
                        print(f"Updated main article authors with full first names")
                
                if "references" in data:
                    for i, ref in enumerate(data["references"]):
                        if "authors" in ref:
                            ref_updated = self._update_author_list(ref["authors"], name_mappings)
                            if ref_updated:
                                refs_updated.append(i)
                
                if refs_updated:
                    print(f"Updated authors in {len(refs_updated)} references with full first names")
                
                if main_authors_updated or refs_updated:
                    print("Successfully updated author names in the document")
                else:
                    print("No author names were updated in the document")
            else:
                print("No authors with initial-only first names found in this line")
            
            # Return the updated line
            return json.dumps(data)
                
        except json.JSONDecodeError:
            print("Error: Invalid JSON in line")
            return line  # Return the original line if we can't parse it
    
    def _collect_initial_authors(self, author_list):
        """Collect authors with initial-only first names"""
        initial_authors = []
        for author in author_list:
            if len(author) >= 2 and all(author[:2]):
                first_name = author[0]
                last_name = author[1]
                
                # Double-check that this is actually an initial
                if is_initial_name(first_name):
                    formatted_name = format_author_name(author)
                    initial_authors.append(formatted_name)
                    print(f"Identified initial first name: '{first_name}' in author '{formatted_name}'")
                else:
                    print(f"Skipping non-initial first name: '{first_name}' in author '{format_author_name(author)}'")
        
        if initial_authors:
            print(f"Found {len(initial_authors)} authors with initial-only first names: {initial_authors}")
        
        return initial_authors
    
    async def _resolve_works_with_initial_names(self, works_with_initial_authors, name_mappings, batch_size, delay):
        """Resolve initial-only first names for authors grouped by work"""
        # Process each work
        work_count = 0
        for work_json, authors in works_with_initial_authors.items():
            work_count += 1
            work = json.loads(work_json)
            
            # Skip authors that have already been resolved
            authors_to_process = [author for author in authors if author not in name_mappings]
            
            if not authors_to_process:
                print(f"All authors in work {work_count} have already been resolved")
                continue
            
            print(f"\nProcessing work {work_count}/{len(works_with_initial_authors)}: {work.get('title', 'Untitled')}")
            print(f"Resolving {len(authors_to_process)} authors in this work")
            
            # Apply rate limiting for browser-use calls
            current_time = datetime.now()
            if self.last_browser_call is not None:
                elapsed_seconds = (current_time - self.last_browser_call).total_seconds()
                if elapsed_seconds < self.min_browser_interval:
                    wait_time = self.min_browser_interval - elapsed_seconds
                    print(f"Rate limiting: Waiting {wait_time:.1f} seconds before next browser call...")
                    time.sleep(wait_time)
            
            # Extract DOI if available, otherwise use work details as non-DOI JSON
            doi = work.get('doi')
            non_doi_json = None if doi else work
            
            # Search for full names for all authors in this work at once
            print(f"Calling search_for_author_first_name with authors: {authors_to_process}")
            # Update the last browser call time
            self.last_browser_call = datetime.now()
            full_names_result = await search_for_author_first_name(authors_to_process, doi, non_doi_json)
            print(f"Result from search_for_author_first_name: {full_names_result}")
            
            if full_names_result:
                # Parse the result and update name mappings
                try:
                    # The result should be in the format "[first1, last1], [first2, last2], ..."
                    # We need to parse this and match with our original authors
                    name_pairs = self._parse_full_names_result(full_names_result)
                    print(f"Parsed name pairs: {name_pairs}")
                    
                    # Extract last names from our original authors for matching
                    original_authors_info = []
                    for author in authors_to_process:
                        parts = author.split()
                        if len(parts) >= 2:
                            first_name = parts[0]
                            last_name = parts[-1]  # Take the last part as the last name
                            original_authors_info.append((author, last_name))
                    
                    print("Original authors with last names:")
                    for author, last_name in original_authors_info:
                        print(f"  {author} (last name: {last_name})")
                    
                    # Match the results with our original authors based on last name
                    matched_authors = set()
                    for author, last_name in original_authors_info:
                        best_match = None
                        best_similarity = 0
                        
                        for i, (first, last) in enumerate(name_pairs):
                            if i in matched_authors:
                                continue  # Skip already matched names
                            
                            # Check if last names match
                            if self._last_names_match(last_name, last):
                                similarity = self._name_similarity(last_name, last)
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    best_match = (i, first, last)
                        
                        if best_match:
                            idx, first, last = best_match
                            matched_authors.add(idx)
                            full_name = f"{first} {last}"
                            name_mappings[author] = full_name
                            print(f"Matched {author} -> {full_name} (similarity: {best_similarity:.2f})")
                        else:
                            # If no match found, keep the original
                            name_mappings[author] = author
                            print(f"No match found for {author}, keeping original")
                    
                    # Also store the full result string with the work key for future reference
                    work_key = f"WORK:{work.get('doi', work.get('title', 'unknown'))}"
                    name_mappings[work_key] = full_names_result
                    
                except Exception as e:
                    print(f"Error parsing full names result: {e}")
                    # If parsing fails, keep the original names
                    for author in authors_to_process:
                        name_mappings[author] = author
            else:
                # If no result, keep the original names
                print("No result from search_for_author_first_name")
                for author in authors_to_process:
                    name_mappings[author] = author
                    print(f"No result for {author}, keeping original")
            
            # Add a delay between works
            if work_count < len(works_with_initial_authors):
                delay_time = delay + random.uniform(0, 1)
                print(f"Waiting {delay_time:.2f} seconds before next work...")
                time.sleep(delay_time)
    
    def _parse_full_names_result(self, result):
        """Parse the full names result from the API"""
        # Remove any "No author first names found" message
        if result is None or "No author first names found" in result:
            print("No author first names found in result")
            return []
        
        print(f"Parsing result: '{result}'")
        
        # Parse the result in the format "[first1, last1], [first2, last2], ..."
        name_pairs = []
        try:
            # Clean up the result string
            result = result.strip().strip("'")
            
            # Try to extract name pairs using regex for more robust parsing
            import re
            pattern = r'\[([^,]+),\s*([^\]]+)\]'
            matches = re.findall(pattern, result)
            
            if matches:
                print(f"Found {len(matches)} name pairs using regex")
                for first, last in matches:
                    first = first.strip()
                    last = last.strip()
                    name_pairs.append((first, last))
                    print(f"Added name pair: ({first}, {last})")
                return name_pairs
            
            # If regex fails, try the original parsing methods
            if result.startswith("[") and result.endswith("]"):
                print("Detected bracket format")
                # Format: "[first1, last1], [first2, last2], ..."
                # Remove outer brackets if the entire string is enclosed
                if result.count("[") > 1:  # Multiple name pairs
                    print(f"Detected multiple name pairs: {result.count('[')} pairs")
                    # Split by "], [" to get individual name pairs
                    pairs = result.split("], [")
                    # Clean up the first and last pair
                    pairs[0] = pairs[0][1:]  # Remove leading "["
                    pairs[-1] = pairs[-1][:-1]  # Remove trailing "]"
                else:  # Single name pair
                    print("Detected single name pair")
                    pairs = [result[1:-1]]  # Remove surrounding brackets
                
                for pair in pairs:
                    if "," in pair:
                        parts = pair.split(",", 1)
                        if len(parts) == 2:
                            first, last = parts
                            first = first.strip()
                            last = last.strip()
                            name_pairs.append((first, last))
                            print(f"Added name pair: ({first}, {last})")
            else:
                print("No bracket format detected, trying alternative parsing")
                
                # Check if this is a comma-separated list of full names
                # This handles formats like "Susan Dynarski, C.J. Libassi, Katherine Michelmore, Stephanie Owen"
                full_names = [name.strip() for name in result.split(',')]
                print(f"Split into {len(full_names)} full names: {full_names}")
                
                for full_name in full_names:
                    # Split each full name into first and last parts
                    name_parts = full_name.split()
                    if len(name_parts) >= 2:
                        # First name is the first part, last name is everything else
                        first_name = name_parts[0]
                        last_name = ' '.join(name_parts[1:])
                        name_pairs.append((first_name, last_name))
                        print(f"Added name pair from full name: ({first_name}, {last_name})")
                    elif len(name_parts) == 1:
                        # If there's only one part, it's probably a last name
                        print(f"Warning: Could not split '{full_name}' into first and last name")
        
            print(f"Final parsed name pairs: {name_pairs}")
            return name_pairs
        except Exception as e:
            print(f"Error parsing name pairs: {e}")
            print(f"Original result: {result}")
            return []
    
    def _update_author_list(self, author_list, name_mappings):
        """Update author list with resolved names"""
        updated = False
        
        # First, collect all authors with initial-only first names
        initial_authors = []
        for i, author in enumerate(author_list):
            if len(author) >= 2 and all(author[:2]):
                if is_initial_name(author[0]):
                    initial_authors.append((i, author))
        
        # If we have initial authors and resolved names
        if initial_authors:
            # Try to get the work key first
            work_key = None
            if "doi" in self.current_work:
                work_key = f"WORK:{self.current_work['doi']}"
            elif "title" in self.current_work:
                work_key = f"WORK:{self.current_work['title']}"
            
            # Check if we have the full result for this work
            if work_key and work_key in name_mappings:
                full_result = name_mappings[work_key]
                print(f"Found full result for work: {full_result}")
                resolved_names = self._parse_full_names_result(full_result)
            else:
                # Fall back to individual author mappings
                resolved_names = []
                for _, author in initial_authors:
                    name = format_author_name(author)
                    if name in name_mappings:
                        # Get the resolved name
                        resolved_name = name_mappings[name]
                        # Parse it into first and last name
                        parts = resolved_name.split()
                        if len(parts) >= 2:
                            first_name = parts[0]
                            last_name = ' '.join(parts[1:])
                            resolved_names.append((first_name, last_name))
                            print(f"Using individual mapping for {name} -> {resolved_name}")
            
            print(f"Parsed {len(resolved_names)} resolved names for {len(initial_authors)} initial authors")
            
            # Print all initial authors and their corresponding resolved names
            print("Initial authors:")
            for i, (idx, author) in enumerate(initial_authors):
                original_name = format_author_name(author)
                print(f"  {i+1}. {original_name}")
            
            print("Resolved names:")
            for i, name_pair in enumerate(resolved_names):
                first, last = name_pair
                print(f"  {i+1}. {first} {last}")
            
            # Only proceed if the number of resolved names matches the number of initial authors
            if len(resolved_names) == len(initial_authors):
                print("\nMapping authors:")
                for i, (idx, author) in enumerate(initial_authors):
                    if i < len(resolved_names):
                        first, last = resolved_names[i]
                        original_name = format_author_name(author)
                        # Verify the last name matches approximately (to avoid mismatches)
                        if self._last_names_match(author[1], last):
                            # Update the first name
                            author[0] = first
                            print(f"  {i+1}. {original_name} -> {first} {author[1]}")
                            updated = True
                        else:
                            print(f"  {i+1}. WARNING: Last name mismatch - {author[1]} vs {last}. Not updating {original_name}")
            else:
                print(f"WARNING: Number of resolved names ({len(resolved_names)}) doesn't match number of initial authors ({len(initial_authors)}). Not updating.")
        
        return updated

    def _last_names_match(self, last1, last2):
        """Check if two last names match approximately (to avoid mismatches)"""
        # Clean up the names
        last1 = last1.lower().strip().rstrip(']')
        last2 = last2.lower().strip().rstrip(']')
        
        # Check for exact match
        if last1 == last2:
            return True
        
        # Check if one is contained in the other
        if last1 in last2 or last2 in last1:
            return True
        
        # Check for similarity (e.g., "Stiglitz" vs "Stiglitz]")
        similarity = 0
        for c1, c2 in zip(last1, last2):
            if c1 == c2:
                similarity += 1
        
        max_len = max(len(last1), len(last2))
        if max_len > 0 and similarity / max_len > 0.8:  # 80% similarity threshold
            return True
        
        return False

    def _name_similarity(self, name1, name2):
        """Calculate similarity between two names (0-1 scale)"""
        # Clean up the names
        name1 = name1.lower().strip().rstrip(']')
        name2 = name2.lower().strip().rstrip(']')
        
        # Check for exact match
        if name1 == name2:
            return 1.0
        
        # Check if one is contained in the other
        if name1 in name2:
            return len(name1) / len(name2)
        if name2 in name1:
            return len(name2) / len(name1)
        
        # Calculate character-by-character similarity
        similarity = 0
        for c1, c2 in zip(name1, name2):
            if c1 == c2:
                similarity += 1
        
        max_len = max(len(name1), len(name2))
        if max_len > 0:
            return similarity / max_len
        return 0

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
        resolver = AuthorNameResolver(args.jsonl_file)
        
        # Run the async process_file method
        name_mappings = asyncio.run(resolver.process_file(args.batch_size, args.delay))
        
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
