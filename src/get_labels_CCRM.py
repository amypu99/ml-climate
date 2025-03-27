import fitz  # PyMuPDF for handling PDFs
from pdf2image import convert_from_path  # Convert PDF pages to images
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd

pdf_path = "./CCRM2024.pdf"
doc = fitz.open(pdf_path)

pages_to_process = [73, 75, 77, 80, 83, 89, 91, 93, 95, 97, 103, 105, 107, 109, 111, 117, 119, 121, 123, 125]

# Color-based boxes (for rating classification)
color_boxes = [
    (470, 240, 680, 250, "1. Transparency and Integrity"),
    (1850, 240, 2030, 250, "2. Transparency"),
    (2050, 240, 2220, 250, "2. Integrity"),
    (900, 1040, 1000, 1050, "3. Transparency"),
    (1080, 1040, 1150, 1050, "3. Integrity"),
    (1850, 1040, 2030, 1050, "4. Transparency"),
    (2050, 1040, 2220, 1050, "4. Integrity"),
]

# Color classification (mapping average RGB values to labels)
color_mapping = {
    (196, 47, 39): "Very Poor",
    (241, 106, 33): "Poor",
    (250, 179, 22): "Moderate",
    (195, 216, 52): "Reasonable",
    (92, 185, 71): "High",
}

# Text-based bounding boxes
text_boxes = [
    (220, 100, 1100, 230, "Name"),
    (1300, 150, 1450, 220, "What is their revenue?"),
    (1470, 150, 1630, 220, "What are their emissions?"),
    (1650, 140, 1840, 220, "What is their pledge?"),
    (1850, 150, 2030, 220, "What is their transparency score?"),
    (2050, 150, 2220, 220, "What is their overall integrity score?"),
    (300, 320, 650, 380, "Tracking and disclosure"),
    (300, 390, 650, 530, "Major emission sources"),
    (1000, 320, 1800, 370, "Headline target or pledge"),
    (1000, 370, 1800, 460, "Short-term targets - grey"),
    (1260, 470, 1800, 600, "Short-term targets - white"),
    (1000, 600, 1800, 680, "Medium-term targets - grey"),
    (1260, 680, 1800, 800, "Medium-term targets - white"),
    (1000, 800, 1800, 900, "Longer-term targets - grey"),
    (1260, 900, 1800, 1010, "Longer-term targets - white"),
    (350, 1100, 480, 1190, "Operational emissions (scope 1) number"),
    (480, 1100, 870, 1190, "Operational emissions (scope 1) verbiage"),
    (350, 1190, 480, 1280, "Renewable electricity (scope 2) number"),
    (480, 1190, 870, 1280, "Renewable electricity (scope 2) verbiage"),
    (350, 1280, 480, 1380, "Upstream emissions (scope 3) number"),
    (480, 1280, 870, 1380, "Upstream emissions (scope 3) verbiage"),
    (350, 1380, 480, 1460, "Downstream emissions (scope 3) number"),
    (480, 1380, 870, 1470, "Downstream emissions (scope 3) verbiage"),
    (1500, 1100, 1850, 1220, "Climate contributions today"),
    (1500, 1220, 1850, 1330, "Misleading offsetting claims today"),
]

def classify_color(avg_color):
    avg_color = tuple(map(int, avg_color))
    for known_color, label in color_mapping.items():
        if np.linalg.norm(np.array(avg_color) - np.array(known_color)) < 15:
            return label
    return "Unknown"

# Create an empty DataFrame with color ratings
color_columns = [label for _, _, _, _, label in color_boxes]
text_columns = ["Name"] + [label for _, _, _, _, label in text_boxes[1:]]
columns = text_columns + color_columns
df = pd.DataFrame(columns=columns)

# Loop through each page
for page_number in pages_to_process:
    print(f"Processing page {page_number+1}...")

    page = doc[page_number]

    # Convert PDF page to image
    images = convert_from_path(pdf_path, first_page=page_number+1, last_page=page_number+1)
    image = images[0]
    draw = ImageDraw.Draw(image)

    extracted_data = {}

    # Extract color from color-based bounding boxes
    for x0, y0, x1, y1, label in color_boxes:
        cropped_img = image.crop((x0, y0, x1, y1))
        img_array = np.array(cropped_img)
        avg_color = img_array.mean(axis=(0, 1))  # Get the average color
        color_label = classify_color(avg_color)  # Classify color

        # Store extracted color classification
        extracted_data[label] = color_label

    # Extract text from text-based bounding boxes
    for x0, y0, x1, y1, label in text_boxes:
        # Convert image coordinates to PDF coordinates
        x0_pdf, y0_pdf, x1_pdf, y1_pdf = [coord / (205/72) for coord in (x0, y0, x1, y1)]
        rect = fitz.Rect(x0_pdf, y0_pdf, x1_pdf, y1_pdf)

        words = page.get_text("words")
        selected_words = [
            w[4] for w in words if x0_pdf <= w[0] <= x1_pdf and y0_pdf <= w[1] <= y1_pdf
        ]
        extracted_text = " ".join(selected_words) 

        extracted_data[label] = extracted_text.strip()

    df = df._append(extracted_data, ignore_index=True)

csv_output_path = "extracted_CCRM2024.csv"
df.to_csv(csv_output_path, index=False)

print(f"Extraction complete. Data saved to {csv_output_path}")
