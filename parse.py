import pdfplumber
import re
import openai
import spacy
import os
import dotenv
import sys

pdf_path = sys.argv[1]

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
        "education",
        "employment",
        "academic positions",
        "academic appointments",
        "professional experience", 
        "degrees",
        "work experience",
        # Versions without spaces
        "academicpositions",
        "academicappointments", 
        "professionalexperience",
        "workexperience"
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
        # Remove spaces for comparison since some PDFs might not have spaces
        text_no_spaces = text.replace(" ", "").lower()
        if text_no_spaces in target_keywords:
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
    
    # Define section headers to skip
    section_headers = {
        "education", "academic positions", "academicpositions",
        "academic appointments", "academicappointments",
        "professional experience", "professionalexperience",
        "work experience", "workexperience"
    }
    
    def clean_text(text):
        """Add spaces before capital letters to help with parsing"""
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'(\d)([A-Z])', r'\1 \2', text)
        return text
    
    def extract_education_info(line):
        """Special handler for education entries"""
        # Clean up the text first
        line = clean_text(line)
        
        # Try to match common education patterns
        education_pattern = r'(\d{4})\s*([A-Za-z.]+|Diploma)\s*[.,]?\s*(?:\((.*?)\))?\s*([^,]+)?\s*,\s*([^,]+)(?:,\s*(.+))?'
        match = re.match(education_pattern, line)
        
        if match:
            year = match.group(1)
            degree = match.group(2)
            honors = match.group(3)  # e.g., "Summa Cum Laude"
            field = match.group(4)   # e.g., "Financial Economics"
            institution = match.group(5)  # e.g., "MIT Sloan School of Management"
            location = match.group(6)     # e.g., "Massachusetts"
            
            # Process the line with spaCy for backup institution detection
            doc = nlp(line)
            spacy_institutions = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE"]]
            
            # Use the longest matching institution name
            if spacy_institutions:
                institution = max(spacy_institutions, key=len)
            
            return {
                'year': year,
                'degree': degree,
                'institution': institution,
                'field': field,
                'honors': honors,
                'raw_text': line.strip()
            }
        return None
    
    def extract_position_info(line, institutions=None):
        """Helper function to extract position information from a line"""
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', line)
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
                if potential_role and (not institutions or not any(inst in potential_role for inst in institutions)):
                    position['role'] = potential_role
                    break
        
        return position
    
    # Split text into lines and process each line
    lines = text.split('\n')
    current_section = None
    current_entry = None
    
    for line in lines:
        if not line.strip():
            continue
        
        # Check if this is a section header
        clean_line = line.replace(" ", "").lower()
        if clean_line in section_headers:
            current_section = clean_line
            continue
        
        # Handle education entries differently
        if current_section == "education":
            edu_info = extract_education_info(line)
            if edu_info:
                structured_entries.append({
                    'institution': edu_info['institution'],
                    'education_info': edu_info,
                    'raw_text': line.strip()
                })
            continue
        
        # Process regular position entries
        doc = nlp(clean_text(line))
        institutions = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE"]]
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', line)
        
        # Rest of the existing position handling code...
        if years:
            if institutions:
                if not current_entry or institutions[0] != current_entry['institution']:
                    if current_entry:
                        structured_entries.append(current_entry)
                    current_entry = {
                        'institution': institutions[0],
                        'positions': [],
                        'raw_text': line.strip()
                    }
            
            if current_entry:
                position = extract_position_info(line, institutions)
                current_entry['positions'].append(position)
    
    # Don't forget the last entry
    if current_entry:
        structured_entries.append(current_entry)
    
    return structured_entries

def classify_with_openai(filtered_text):
    """Use GPT-4 to classify education and employment details with preserved structure."""
    
    json_structure = '''{
        "education": {
            "undergraduate": [
                {
                    "institution": "University Name",
                    "graduation_year": "YYYY"
                }
                // Can have multiple undergraduate degrees
            ],
            "masters": [
                {
                    "institution": "University Name",
                    "graduation_year": "YYYY"
                }
                // Can have multiple masters degrees
            ],
            "phd": [
                {
                    "institution": "University Name",
                    "graduation_year": "YYYY"
                }
                // Can have multiple PhDs
            ]
        },
        "affiliations": [
            {
                "institution": "University Name",
                "title": "one of: [Postdoctoral Researcher, Assistant Professor, Associate Professor, Professor]",
                "official_title": "Full official title as written in CV",
                "year_range": {
                    "start": "YYYY",
                    "end": "YYYY or present"
                }
            }
        ]
    }'''
    
    prompt = f"""
    Analyze these sections from an academic CV and organize them into a structured format.
    
    Input text:
    {filtered_text}
    
    Please output a JSON-like structure with these exact keys and hierarchy:
    
    {json_structure}
    
    Rules:
    1. For education entries:
       - Extract institution and graduation year
       - Include ALL degrees - a person may have multiple degrees at each level
       - If a graduation year isn't specified but a year range is given, use the end year
       - Classify "Diploma" degrees as undergraduate degrees unless explicitly stated otherwise
    2. For affiliations (postdoc and faculty positions):
       - Extract institution and year range
       - Store the complete position title under "official_title"
       - For "title", choose the most appropriate from these options ONLY:
         * "Postdoctoral Researcher" - for postdoc positions
         * "Assistant Professor" - for assistant professor roles
         * "Associate Professor" - for associate professor roles
         * "Professor" - for full professor roles
    3. Use "present" for current positions
    4. If any field is uncertain, use null
    5. Keep the exact order of entries as they appear in the text
    6. Include all positions for each institution
    7. For visiting positions, use the appropriate title (e.g., "Professor" for Visiting Professor)
    """
    
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI that structures academic CV data into a consistent format."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

def process_cv(pdf_path):
    """Main pipeline for processing a CV PDF."""
    structured_text = extract_text_from_pdf(pdf_path)
    
    # First just analyze the sections
    filtered_text = filter_relevant_sections(structured_text)
    print(filtered_text)
    
    # Ask user if they want to continue with the full processing
    input("\nPress Enter to continue with full processing...")
    
    classified_data = classify_with_openai(filtered_text)
    return classified_data

# Example usage
pdf_path = f"downloaded_CVs/{pdf_path}"
result = process_cv(pdf_path)
print(result)
