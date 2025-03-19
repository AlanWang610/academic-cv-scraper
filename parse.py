import pdfplumber
import re
import openai
import spacy
import os
import dotenv

dotenv.load_dotenv()
# Load NLP model (you can use 'en_core_web_sm' for a lighter model)
nlp = spacy.load("en_core_web_md")

# Predefined mapping for specific aliases (e.g., business school to university)
UNIVERSITY_ALIAS_MAP = {
    "MIT Sloan": "Massachusetts Institute of Technology",
    "Harvard Business School": "Harvard University",
    "Stanford GSB": "Stanford University",
    # Add more specific aliases as needed
}

# Set of possible university names (to be added later)
KNOWN_UNIVERSITIES_SET = set()

# OpenAI API key (set your environment variable or replace directly)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(pdf_path):
    """Extract text and formatting from a PDF file."""
    structured_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            chars = page.chars
            chars.sort(key=lambda c: (c['top'], c['x0']))
            
            current_line = []
            current_y = None
            current_size = None
            current_font = None
            current_bold_status = []
            current_x = None  # Track x-position for indentation
            
            for char in chars:
                # Check if this is a new line based on y-position
                if current_y is None or abs(char['top'] - current_y) > 3:
                    if current_line:
                        is_bold = any('bold' in font.lower() for font in current_bold_status)
                        structured_text.append({
                            'text': ''.join(current_line),
                            'size': current_size,
                            'top': current_y,
                            'x0': current_x,  # Store the x-position
                            'is_bold': is_bold,
                            'font': current_font
                        })
                    current_line = []
                    current_bold_status = []
                    current_y = char['top']
                    current_size = char['size']
                    current_font = char['fontname']
                    current_x = char['x0']  # Set x-position for new line
                
                current_line.append(char['text'])
                current_bold_status.append(char['fontname'])
            
            # Don't forget the last line
            if current_line:
                is_bold = any('bold' in font.lower() for font in current_bold_status)
                structured_text.append({
                    'text': ''.join(current_line),
                    'size': current_size,
                    'top': current_y,
                    'x0': current_x,
                    'is_bold': is_bold,
                    'font': current_font
                })
    
    return structured_text

def filter_relevant_sections(structured_text):
    """Extract sections related to education and employment using font size and bold hints."""
    sections = []
    # Define exact section headers we want to match (case-insensitive)
    target_keywords = {
        "education", "employment", "academic positions", 
        "academic appointments", "professional experience",
        "degrees", "work experience"
    }
    
    # First pass: analyze font sizes and find headers
    font_sizes = {}
    for line in structured_text:
        size = line['size']
        if size not in font_sizes:
            font_sizes[size] = 0
        font_sizes[size] += 1
    
    # Find the most common font size (likely body text)
    body_size = max(font_sizes.items(), key=lambda x: x[1])[0]
    
    # First, find our target section headers and their formatting characteristics
    section_header_formats = []
    for line in structured_text:
        text = line['text'].strip()
        if text.lower() in target_keywords:
            section_header_formats.append({
                'size': line['size'],
                'is_bold': line['is_bold'],
                'font': line['font']
            })
            print(f"Found target section: {text}")
            print(f"Format: size={line['size']}, bold={line['is_bold']}, font={line['font']}")
    
    if not section_header_formats:
        return ""
    
    # Second pass: identify sections and their content
    current_section = None
    current_content = []
    sections_dict = {}
    
    def is_similar_format(line, header_format):
        """Check if a line has similar formatting to our section headers"""
        size_match = abs(line['size'] - header_format['size']) < 0.1
        bold_match = line['is_bold'] == header_format['is_bold']
        font_match = line['font'] == header_format['font']
        return size_match and bold_match and font_match
    
    for line in structured_text:
        text = line['text'].strip()
        if not text:  # Skip empty lines
            continue
        
        # Check if this is one of our target section headers
        is_target_header = text.lower() in target_keywords and \
                         any(is_similar_format(line, format) for format in section_header_formats)
        
        # Check if this is any peer-level header (similar formatting to our section headers)
        is_peer_header = any(is_similar_format(line, format) for format in section_header_formats) and \
                        not is_target_header
        
        if is_target_header:
            # Save previous section if exists
            if current_section and current_content:
                sections_dict[current_section] = current_content
            
            current_section = text
            current_content = []
        
        elif current_section is not None:
            if is_peer_header:
                # We've hit another section header of similar formatting
                if current_content:
                    sections_dict[current_section] = current_content
                current_section = None
                current_content = []
            else:
                current_content.append(text)
    
    # Don't forget the last section
    if current_section and current_content:
        sections_dict[current_section] = current_content
    
    # Format the output
    formatted_sections = []
    for header, content in sections_dict.items():
        formatted_sections.append(f"{header}\n" + "\n".join(content))
    
    return "\n\n".join(formatted_sections)

def extract_affiliations(text, structured_text):
    """Extract structured entries with affiliations, years, and roles, considering indentation."""
    structured_entries = []
    
    # Split text into lines and process each line
    lines = text.split('\n')
    current_institution = None
    current_entry = None
    base_x = None  # Track the leftmost x-position
    
    for line in lines:
        if not line.strip():
            continue
        
        # Find the line in structured_text to get its x-position
        matching_lines = [l for l in structured_text if l['text'].strip() == line.strip()]
        if not matching_lines:
            continue
            
        line_info = matching_lines[0]
        x_pos = line_info['x0']
        
        # Initialize base_x with the first line's position if not set
        if base_x is None:
            base_x = x_pos
        
        # Determine if this is an indented line
        is_indented = x_pos > base_x + 10  # Allow for some tolerance
        
        # Process the line with spaCy
        doc = nlp(line)
        
        # Extract years
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', line)
        
        # If this is a non-indented line, treat it as a potential new institution
        if not is_indented:
            # Save previous entry if it exists
            if current_entry:
                structured_entries.append(current_entry)
            
            # Look for institution entities
            institutions = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE"]]
            
            if institutions:
                current_institution = institutions[0]
                current_entry = {
                    'institution': current_institution,
                    'positions': [],
                    'raw_text': line.strip()
                }
        
        # If this is an indented line and we have a current institution
        elif current_entry is not None:
            position = {
                'text': line.strip(),
                'years': years,
                'role': None
            }
            
            # Try to extract role
            text_parts = line.split(',')
            for part in text_parts:
                if not any(year in part for year in years):
                    potential_role = part.strip()
                    if potential_role:
                        position['role'] = potential_role
                        break
            
            current_entry['positions'].append(position)
    
    # Don't forget the last entry
    if current_entry:
        structured_entries.append(current_entry)
    
    return structured_entries

def classify_with_openai(structured_entries):
    """Use GPT-4 to classify education and employment details with preserved structure."""
    formatted_entries = []
    for entry in structured_entries:
        formatted_entry = f"Institution: {entry['institution']}\n"
        if entry['positions']:
            formatted_entry += "Positions:\n"
            for pos in entry['positions']:
                formatted_entry += f"- {pos['text']}\n"
        else:
            formatted_entry += f"Main line: {entry['raw_text']}\n"
        formatted_entries.append(formatted_entry)
    
    prompt = f"""
    Classify the following academic entries into education (undergrad, master's, PhD) 
    and employment (postdoc, faculty, visiting) categories. Each entry shows an institution
    and may include multiple positions or roles at that institution.

    Entries:
    {'\n'.join(formatted_entries)}
    
    Please format the output as:
    EDUCATION:
    - Institution (Years): Degree/Program
    
    EMPLOYMENT:
    - Institution:
      * Role (Years)
      * Role (Years)
    """
    
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI that classifies academic CV data."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

def process_cv(pdf_path):
    """Main pipeline for processing a CV PDF."""
    structured_text = extract_text_from_pdf(pdf_path)
    
    # First just analyze the sections
    filtered_text = filter_relevant_sections(structured_text)
    
    # Ask user if they want to continue with the full processing
    input("\nPress Enter to continue with full processing...")
    
    # Extract structured entries instead of separate lists
    structured_entries = extract_affiliations(filtered_text, structured_text)
    
    # Debug output to verify structure
    print("\nStructured Entries:")
    for entry in structured_entries:
        print("-" * 50)
        print(f"Raw text: {entry['raw_text']}")
        print(f"Institutions: {entry['institution']}")
        print(f"Positions: {entry['positions']}")
    
    classified_data = classify_with_openai(structured_entries)
    return classified_data

# Example usage
pdf_path = "downloaded_CVs/Antoinette Schoar.pdf"
result = process_cv(pdf_path)
print(result)
