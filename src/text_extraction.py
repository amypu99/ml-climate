import os
import re
from collections import Counter
from pdfminer.high_level import extract_text

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using pdfminer.six.
    Only extracts text in the PDF’s text layer (ignores text in images).
    """
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return ""

def remove_top_words(text, top_n=20):
    """
    Identifies the top `top_n` occurring words and removes them from the text.
    Returns both the word counts and the cleaned text.
    """
    # Find all words (converted to lower-case for counting).
    words = re.findall(r'\b\w+\b', text.lower())
    counts = Counter(words)
    
    # Identify the top N words.
    top_words = set(word for word, _ in counts.most_common(top_n))
    
    # Build a regex pattern to match any of the top words (ignoring case).
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, top_words)) + r')\b', flags=re.IGNORECASE)
    
    # Remove the top words from the original text.
    cleaned_text = pattern.sub("", text)
    
    # Clean up extra spaces.
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return counts, cleaned_text

def process_pdfs(input_folder, output_folder, top_n=20):
    """
    Processes each PDF file in the input folder:
      - Extracts text and saves it to a file.
      - Computes and prints the top N words and their counts.
      - Removes these top words from the text and saves the cleaned text to a separate file.
    """
    # Create the output folder if it doesn't exist.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through each PDF file in the input folder.
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_folder, filename)
            print(f"Processing: {pdf_path}")
            
            # Extract text from the PDF.
            text = extract_text_from_pdf(pdf_path)
            if not text:
                continue
            
            # Save the full extracted text.
            full_output_filename = os.path.splitext(filename)[0] + ".txt"
            full_output_path = os.path.join(output_folder, full_output_filename)
            with open(full_output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Full extracted text saved to: {full_output_path}")
            
            # Remove the top N words.
            counts, cleaned_text = remove_top_words(text, top_n=top_n)
            
            # Print out the top words and their counts.
            print(f"Top {top_n} words in {filename}:")
            for word, count in counts.most_common(top_n):
                print(f"  {word}: {count}")
            
            # Save the cleaned text.
            cleaned_output_filename = os.path.splitext(filename)[0] + "_cleaned.txt"
            cleaned_output_path = os.path.join(output_folder, cleaned_output_filename)
            with open(cleaned_output_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            print(f"Cleaned text saved to: {cleaned_output_path}\n")

if __name__ == "__main__":
    input_folder = "./climate_reports"
    output_folder = os.path.join(input_folder, "extracted_text")
    
    process_pdfs(input_folder, output_folder, top_n=20)
