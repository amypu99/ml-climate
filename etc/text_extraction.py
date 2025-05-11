import os
import re
from collections import Counter
from pdfminer.high_level import extract_text
import pdfplumber

def extract_relevant_text(pdf_path, header_height=70, footer_height=70):
    """
    Extracts text from a PDF using pdfplumber by cropping out the header and footer areas.
    Adjust header_height and footer_height as needed.
    """
    all_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_height = page.height
                # Define a crop box to ignore header and footer areas.
                crop_bbox = (0, header_height, page.width, page_height - footer_height)
                cropped_page = page.within_bbox(crop_bbox)
                page_text = cropped_page.extract_text()
                if page_text:
                    all_text.append(page_text)
        return "\n".join(all_text)
    except Exception as e:
        print(f"Error processing {pdf_path} with pdfplumber: {e}")
        return ""

def filter_boilerplate(text):
    """
    Applies heuristic filtering to remove common boilerplate lines such as:
      - Lines that are only numbers (likely page numbers)
      - Lines that reference pages (e.g., "Page 1")
      - Lines containing phrases like "table of contents"
    """
    cleaned_lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip lines that are only numbers.
        if re.fullmatch(r'\d+', line):
            continue
        # Skip lines referencing page numbers.
        if re.search(r'page\s+\d+', line, re.IGNORECASE):
            continue
        # Skip lines that mention "table of contents".
        if "table of contents" in line.lower():
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def post_clean_text(text):
    """
    Further cleans the text by normalizing quotes, removing extra spaces,
    and optionally removing duplicate consecutive words.
    """
    # Normalize smart quotes and other unicode quotes.
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    # Remove extra spaces.
    text = re.sub(r'\s+', ' ', text)
    # Optionally, remove duplicate consecutive words (e.g., "About About" -> "About").
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE)
    # Clean stray punctuation spacing.
    text = re.sub(r'\s+,', ',', text)
    text = re.sub(r',\s+', ', ', text)
    return text.strip()

def remove_top_words(text, top_n=20):
    """
    Counts word frequencies (ignoring case), identifies the top `top_n` words,
    and removes them from the text. Returns the word counts and the cleaned text.
    """
    # Extract words (in lower-case) for counting.
    words = re.findall(r'\b\w+\b', text.lower())
    counts = Counter(words)
    # Identify the top N words.
    top_words = set(word for word, _ in counts.most_common(top_n))
    # Build a regex to match any of these words, ignoring case.
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, top_words)) + r')\b', flags=re.IGNORECASE)
    # Remove the top words.
    cleaned_text = pattern.sub("", text)
    # Clean up extra spaces.
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return counts, cleaned_text

def process_pdfs(input_folder, output_folder, top_n=20, header_height=50, footer_height=50):
    """
    Processes each PDF file in the input folder by:
      - Extracting text with layout-based extraction.
      - Applying heuristic filtering to remove common boilerplate.
      - Further cleaning up the text.
      - Saving two files:
           1. The cleaned, relevant text.
           2. The same text with the top `top_n` words removed.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_folder, filename)
            print(f"Processing: {pdf_path}")
            
            # Extract text while ignoring headers and footers.
            extracted_text = extract_relevant_text(pdf_path, header_height, footer_height)
            if not extracted_text:
                continue

            full_output_filename = os.path.splitext(filename)[0] + ".txt"
            full_output_path = os.path.join(output_folder, full_output_filename)
            with open(full_output_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            print(f"Extracted text saved to: {full_output_path}")
            
            filtered_text = filter_boilerplate(extracted_text)
            cleaned_text = post_clean_text(filtered_text)
            
            # Remove the top N occurring words.
            counts, cleaned_top_removed_text = remove_top_words(cleaned_text, top_n=top_n)
            print(f"Top {top_n} words in {filename}:")
            for word, count in counts.most_common(top_n):
                print(f"  {word}: {count}")
            
            # Save the text with the top words removed.
            cleaned_output_filename = os.path.splitext(filename)[0] + "_cleaned.txt"
            cleaned_output_path = os.path.join(output_folder, cleaned_output_filename)
            with open(cleaned_output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_top_removed_text)
            print(f"Text with top words removed saved to: {cleaned_output_path}\n")

if __name__ == "__main__":
    input_folder = "./climate_reports/ccrm_2024"
    output_head_dir = "./climate_reports"
    output_folder = os.path.join(output_head_dir, "extracted_text")
    
    process_pdfs(input_folder, output_folder, top_n=20, header_height=50, footer_height=50)
