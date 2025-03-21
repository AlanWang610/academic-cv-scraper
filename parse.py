import pdfplumber
import re
import openai
import spacy
import os
import dotenv
import sys
import json

dotenv.load_dotenv()
# Load NLP model (you can use 'en_core_web_sm' for a lighter model)
nlp = spacy.load("en_core_web_md")

# OpenAI API key (set your environment variable or replace directly)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(pdf_path):
    """Extract text and formatting from a PDF file."""
    structured_text = []
    
    def is_bold_font(fontname):
        """More accurate bold font detection"""
        bold_indicators = {'bold', 'bd', 'b', 'heavy', 'black', 'demi'}
        # Convert to lowercase and remove spaces for comparison
        font_lower = fontname.lower().replace(' ', '')
        # Check if any bold indicator is a substring, but be careful of words like "bold"
        return any(f'-{ind}' in font_lower or 
                  f'{ind}-' in font_lower or 
                  font_lower.endswith(ind) 
                  for ind in bold_indicators)
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            chars = page.chars
            chars.sort(key=lambda c: (c['top'], c['x0']))
            
            current_line = []
            current_y = None
            current_size = None
            current_font = None
            current_fonts = []  # Track all fonts in the line
            current_x = None
            
            for char in chars:
                # Check if this is a new line based on y-position
                if current_y is None or abs(char['top'] - current_y) > 3:
                    if current_line:
                        # More accurate bold detection using all fonts in the line
                        is_bold = any(is_bold_font(font) for font in current_fonts)
                        
                        structured_text.append({
                            'text': ''.join(current_line),
                            'size': current_size,
                            'top': current_y,
                            'x0': current_x,
                            'is_bold': is_bold,
                            'font': current_font,
                            'fonts': list(set(current_fonts))  # Store unique fonts
                        })
                    current_line = []
                    current_fonts = []
                    current_y = char['top']
                    current_size = char['size']
                    current_font = char['fontname']
                    current_x = char['x0']
                
                current_line.append(char['text'])
                current_fonts.append(char['fontname'])
            
            # Don't forget the last line
            if current_line:
                is_bold = any(is_bold_font(font) for font in current_fonts)
                structured_text.append({
                    'text': ''.join(current_line),
                    'size': current_size,
                    'top': current_y,
                    'x0': current_x,
                    'is_bold': is_bold,
                    'font': current_font,
                    'fonts': list(set(current_fonts))
                })
    
    return structured_text

def filter_relevant_sections(structured_text):
    """Extract sections related to education and employment using hierarchical structure."""
    
    def get_line_characteristics(line):
        """Get key characteristics of a line for hierarchy detection"""
        # More sophisticated header detection
        is_header_format = (
            line['is_bold'] or
            any('bold' in f.lower() for f in line['fonts']) or
            (line['text'].strip().endswith(':') and len(line['text'].strip()) > 2) or
            (line['text'].strip().isupper() and len(line['text'].strip()) > 3)
        )
        
        return {
            'size': line['size'],
            'is_bold': line['is_bold'],
            'font': line['font'],
            'fonts': line['fonts'],
            'x0': line['x0'],
            'has_colon': line['text'].strip().endswith(':'),
            'all_caps': line['text'].strip().isupper(),
            'is_header_format': is_header_format
        }
    
    def find_hierarchy_levels(structured_text):
        """Identify distinct hierarchy levels in the document"""
        header_formats = set()
        
        # First pass: collect potential header formats
        for line in structured_text:
            text = line['text'].strip()
            if not text:
                continue
            
            chars = get_line_characteristics(line)
            
            # Only consider lines that are likely headers
            if chars['is_header_format']:
                level_key = (
                    chars['size'],
                    chars['is_bold'],
                    any('bold' in f.lower() for f in chars['fonts']),
                    chars['x0'],
                    chars['all_caps']
                )
                header_formats.add(level_key)
        
        # Sort formats by hierarchy level
        sorted_formats = sorted(
            header_formats,
            key=lambda x: (-x[0], -int(x[1]), -int(x[2]), x[3])
        )
        
        return sorted_formats
    
    def normalize_text(text):
        """Normalize text by removing spaces and making lowercase"""
        return text.lower().replace(" ", "")
    
    def is_target_section(text, level_idx, hierarchy_levels):
        """Check if this section is one we want to extract"""
        target_keywords = {
            "education",
            "employment",
            "academic positions",
            "academic appointments",
            "professional experience",
            "experience",
            "academic degrees",
            "degrees",
            "work experience"
        }
        
        text_lower = text.lower()
        # Direct match or fuzzy match with target keywords
        return any(keyword in text_lower for keyword in target_keywords)
    
    # Find hierarchy levels
    hierarchy_levels = find_hierarchy_levels(structured_text)
    print("\nDetected hierarchy levels:")
    for i, level in enumerate(hierarchy_levels):
        print(f"Level {i}: size={level[0]}, bold={level[1]}, bold_font={level[2]}, indent={level[3]}, caps={level[4]}")
    
    # Process text using hierarchy
    sections = []
    current_section = None
    current_level = None
    
    for line in structured_text:
        text = line['text'].strip()
        if not text:
            continue
        
        chars = get_line_characteristics(line)
        level_key = (chars['size'], chars['is_bold'], 
                    any('bold' in f.lower() for f in chars['fonts']),
                    chars['x0'], chars['all_caps'])
        
        # Check if this line is a header
        if level_key in hierarchy_levels:
            level_idx = hierarchy_levels.index(level_key)
            print(f"\nFound potential header: '{text}'")
            print(f"Level {level_idx}: {chars}")
            
            # If this is a header at same or higher level, end current section
            if current_section and level_idx <= current_level:
                sections.append(current_section)
                current_section = None
                current_level = None
            
            # Check if this is a target section
            if is_target_section(text, level_idx, hierarchy_levels):
                print("-> Matched target section")
                current_section = {'header': text, 'content': []}
                current_level = level_idx
            
        # Add content to current section if we're in one
        elif current_section:
            current_section['content'].append(text)
    
    # Add final section
    if current_section:
        sections.append(current_section)
    
    # Combine sections into formatted text
    formatted_text = ""
    for section in sections:
        formatted_text += f"\n{section['header']}\n"
        formatted_text += "\n".join(section['content'])
        formatted_text += "\n"
    
    return formatted_text.strip()

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
    print(f"\nProcessing: {pdf_path}")
    
    structured_text = extract_text_from_pdf(pdf_path)
    filtered_text = filter_relevant_sections(structured_text)
    
    print("\nFiltered Text:")
    print("-------------")
    print(filtered_text)
    print("-------------\n")
    
    if not filtered_text.strip():
        print("Warning: No relevant sections found in CV")
        return None
        
    try:
        classified_data = classify_with_openai(filtered_text)
        
        # Ensure we have a proper JSON string
        if not classified_data.startswith('{'):
            classified_data = '{' + classified_data
            
        # Parse the JSON string into a dictionary
        return json.loads(classified_data)
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        print("Raw response:", classified_data)
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def process_folder(folder_path):
    """Process all PDFs in the specified folder."""
    successful = 0
    total = 0
    
    # Get all PDF files in the folder
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Create output file
    output_file = 'parsed_CVs.jsonl'
    
    # Process each PDF and write results immediately to JSONL file
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, pdf_file in enumerate(pdf_files, 1):
            print(f"\nProcessing {idx}/{len(pdf_files)}: {pdf_file}")
            full_path = os.path.join(folder_path, pdf_file)
            total += 1
            
            try:
                result = process_cv(full_path)
                if result:
                    # Get name from filename without .pdf extension
                    name = os.path.splitext(pdf_file)[0]
                    
                    # Create the entry in desired format
                    entry = {
                        "name": name,
                        "education": result["education"],
                        "affiliations": result["affiliations"]
                    }
                    
                    # Write the entry as a single line of JSON
                    json.dump(entry, f, ensure_ascii=False)
                    f.write('\n')
                    successful += 1
                else:
                    print(f"Failed to process {pdf_file}")
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
    
    print(f"\nProcessing complete. Results saved to {output_file}")
    print(f"Successfully processed {successful}/{total} CVs")

def main():
    if len(sys.argv) != 2:
        print("Usage: python parse.py <folder_path>")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} not found")
        sys.exit(1)
    
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a directory")
        sys.exit(1)
    
    process_folder(folder_path)

if __name__ == "__main__":
    print(process_cv("downloaded_CVs/Tong Liu.pdf"))
    # main()
