import os
from pdfminer.high_level import extract_text

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using pdfminer.six.
    Note: This will only extract text that is in the PDF’s text layer,
    so any text that is part of an image will be ignored.
    """
    try:
        text = extract_text(pdf_path)
        return text
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return ""

def process_pdfs(input_folder, output_folder):
    """
    Iterates over all PDF files in the input_folder, extracts text,
    and writes the result to separate text files in the output_folder.
    """
    # Create the output folder if it doesn't exist.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through each file in the input folder.
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(input_folder, filename)
            print(f"Processing: {pdf_path}")
            text = extract_text_from_pdf(pdf_path)
            
            # Define the output file name (change extension to .txt)
            output_filename = os.path.splitext(filename)[0] + ".txt"
            output_path = os.path.join(output_folder, output_filename)
            
            # Write the extracted text to the output file.
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Saved extracted text to: {output_path}")

if __name__ == "__main__":
    input_folder = "/Users/nicolelin/Desktop/climate_reports"
    # Create a nested folder named 'extracted_text' inside the input folder.
    output_folder = os.path.join(input_folder, "extracted_text")
    
    process_pdfs(input_folder, output_folder)
