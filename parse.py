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
            # Get all characters with their properties
            chars = page.chars
            
            # Sort chars by y-position (top to bottom) and x-position (left to right)
            chars.sort(key=lambda c: (c['top'], c['x0']))
            
            current_line = []
            current_y = None
            current_size = None
            
            for char in chars:
                # Check if this is a new line based on y-position
                if current_y is None or abs(char['top'] - current_y) > 3:
                    if current_line:
                        # Store line with its properties
                        structured_text.append({
                            'text': ''.join(current_line),
                            'size': current_size,
                            'top': current_y
                        })
                    current_line = []
                    current_y = char['top']
                    current_size = char['size']
                
                current_line.append(char['text'])
            
            # Don't forget the last line
            if current_line:
                structured_text.append({
                    'text': ''.join(current_line),
                    'size': current_size,
                    'top': current_y
                })
    
    return structured_text

def filter_relevant_sections(structured_text):
    """Extract sections related to education and employment using font size hints."""
    sections = []
    keywords = ["Education", "Degrees", "Employment", "Experience", 
               "Academic Positions", "Academic Appointments", "Professional Experience"]
    
    # First pass: analyze font sizes and find headers
    font_sizes = {}
    for line in structured_text:
        size = line['size']
        if size not in font_sizes:
            font_sizes[size] = 0
        font_sizes[size] += 1
    
    # Find the most common font size (likely body text)
    body_size = max(font_sizes.items(), key=lambda x: x[1])[0]
    
    # Second pass: identify sections and their content
    current_section = None
    current_content = []
    sections_dict = {}  # To store sections and their content
    
    for line in structured_text:
        text = line['text'].strip()
        size = line['size']
        
        if not text:  # Skip empty lines
            continue
        
        # Check if this is a potential header
        is_header = (size > body_size * 1.1 or size < body_size * 0.9) and \
                   any(keyword.lower() in text.lower() for keyword in keywords)
        
        if is_header:
            # Save previous section if exists
            if current_section and current_content:
                sections_dict[current_section] = current_content
            
            current_section = text
            current_content = []
        
        elif current_section is not None:
            # Check if we've hit another major header (but not a keyword section)
            if size > body_size * 1.1:
                # Save current section and reset
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

def extract_affiliations(text):
    """Use NLP to extract institutions and years from text."""
    affiliations = []
    doc = nlp(text)
    
    for ent in doc.ents:
        if ent.label_ in ["ORG", "GPE"]:  # Organizations and locations (for universities)
            affiliations.append(ent.text)
    
    years = re.findall(r'\b(19\d{2}|20\d{2})\b', text)  # Extracts years
    
    return affiliations, years

def standardize_university_names(affiliations):
    """Map extracted affiliations to known university names or aliases."""
    standardized = []
    for name in affiliations:
        if name in UNIVERSITY_ALIAS_MAP:
            standardized.append(UNIVERSITY_ALIAS_MAP[name])
        elif name in KNOWN_UNIVERSITIES_SET:
            standardized.append(name)
        else:
            standardized.append(name)  # Keep as is if not found
    return standardized

def classify_with_openai(affiliations, years):
    """Use GPT-4 to classify education and employment details."""
    prompt = f"""
    Extract and classify the following academic affiliations into education (undergrad, master's, PhD) and employment (postdoc, faculty). 
    Standardize university names and format years as YYYY–YYYY.
    
    Affiliations: {affiliations}
    Years: {years}
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
    filtered_text = filter_relevant_sections(structured_text)
    print("Filtered sections:")
    print(filtered_text)
    print("\n---\n")
    affiliations, years = extract_affiliations(filtered_text)
    print("Found affiliations:", affiliations)
    print("Found years:", years)
    standardized_affiliations = standardize_university_names(affiliations)
    classified_data = classify_with_openai(standardized_affiliations, years)
    return classified_data

# Example usage
pdf_path = "downloaded_CVs/Antoinette Schoar.pdf"
result = process_cv(pdf_path)
print(result)
