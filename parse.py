import pdfplumber
import re
import openai
import spacy
import os
import dotenv
import sys
import json

pdf_path = sys.argv[1]

dotenv.load_dotenv()
# Load NLP model (you can use 'en_core_web_sm' for a lighter model)
nlp = spacy.load("en_core_web_md")

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

def classify_with_openai(filtered_text):
    """Use GPT-4 to classify education and employment details with preserved structure."""
    
    prompt = f"""
    Analyze these sections from an academic CV and organize them into a structured format.
    
    Input text:
    {filtered_text}
    
    Output a valid JSON object with this exact structure:
    {{
        "education": {{
            "undergraduate": [
                {{
                    "institution": "University Name",
                    "graduation_year": "YYYY"
                }}
            ],
            "masters": [
                {{
                    "institution": "University Name",
                    "graduation_year": "YYYY"
                }}
            ],
            "phd": [
                {{
                    "institution": "University Name",
                    "graduation_year": "YYYY"
                }}
            ]
        }},
        "affiliations": [
            {{
                "institution": "University Name",
                "title": "one of: [Postdoctoral Researcher, Assistant Professor, Associate Professor, Professor]",
                "official_title": "Full official title as written in CV",
                "year_range": {{
                    "start": "YYYY",
                    "end": "YYYY or present"
                }}
            }}
        ]
    }}

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
    8. IMPORTANT: Ensure the output is a complete, valid JSON object starting with {{ and ending with }}
    """
    
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI that structures academic CV data into a consistent JSON format. Always output complete, valid JSON objects."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

def process_cv(pdf_path):
    """Main pipeline for processing a CV PDF."""
    structured_text = extract_text_from_pdf(pdf_path)
    filtered_text = filter_relevant_sections(structured_text)
    classified_data = classify_with_openai(filtered_text)
    
    # Clean up the response and parse JSON properly
    try:
        # Remove any markdown formatting if present
        if classified_data.startswith('```'):
            classified_data = classified_data.split('\n', 2)[2]  # Skip first two lines
        if classified_data.endswith('```'):
            classified_data = classified_data[:-3]  # Remove ending backticks
            
        # Parse the JSON string into a dictionary
        classified_data = json.loads(classified_data)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print("Raw response:", classified_data)
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    
    return classified_data

# Example usage
pdf_path = f"downloaded_CVs/{pdf_path}"
result = process_cv(pdf_path)
print(result)
