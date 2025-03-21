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
    """Extract raw text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

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
    
    # Extract raw text
    raw_text = extract_text_from_pdf(pdf_path)
    
    if not raw_text.strip():
        print("Warning: No text found in PDF")
        return None
        
    try:
        # Pass directly to OpenAI
        classified_data = classify_with_openai(raw_text)
        
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
