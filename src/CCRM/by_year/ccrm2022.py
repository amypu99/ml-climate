import fitz  # PyMuPDF for handling PDFs
from pdf2image import convert_from_path  # Convert PDF pages to images
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd

# CCRM 2023 Configuration
pdf_path = "./CCRM2022.pdf"
pages_to_process = [51, 53, 55, 58, 60, 62, 64, 67, 69, 71, 73, 75, 78, 80, 83, 85, 87, 90, 92, 94, 96, 99, 101, 104, 106]
csv_output_path = "extracted_CCRM2022.csv"

# Color-based boxes (for rating classification)
color_boxes = [
    (1050, 400, 1060, 450, "Overall transparency score"),
    (1270, 400, 1280, 450, "Overall integrity score"),
    (1050, 630, 1060, 640, "1. Transparency and Integrity"),
    (1050, 1040, 1060, 1050, "2. Transparency"),
    (1350, 1040, 1360, 1050, "2. Integrity"),
    (1050, 1515, 1060, 1520, "3. Transparency"),
    (1350, 1515, 1360, 1520, "3. Integrity"),
    (1050, 1800, 1060, 1810, "4. Transparency"),
    (1350, 1800, 1360, 1810, "4. Integrity"),
]

# Text-based bounding boxes
text_boxes = [
    (120, 150, 1000, 270, "Name"),
    (350, 320, 580, 450, "What is their revenue?"),
    (580, 320, 800, 450, "What are their emissions?"),
    (800, 320, 1100, 450, "What is their pledge?"),
    (550, 700, 1100, 790, "Major emission sources"),
    (550, 1100, 1100, 1160, "Headline target or pledge"),
    (550, 1160, 1100, 1240, "Coverage of emission sources"),
    (640, 1240, 1100, 1350, "Reduction of own emissions"),
    (550, 1530, 1100, 1650, "Emission Reduction Measures"),
    (550, 1650, 1100, 1750, "Renewable Electricity"),
    (550, 1820, 1100, 1910, "Climate contributions today"),
    (550, 1910, 1100, 1970, "Misleading offsetting claims today"),
]

# Color classification (mapping average RGB values to labels)
color_mapping = {
    (194, 39, 21): "Very Poor",
    (235, 97, 10): "Poor",
    (250, 182, 0): "Moderate",
    (206, 214, 6): "Reasonable",
    (96, 178, 47): "High",
}

# Function to classify color based on RGB values
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
doc = fitz.open(pdf_path)

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

        # Draw bounding box (Red for colors)
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        draw.text((x0 + 5, y0 - 15), label, fill="red")

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

        # Draw bounding box (Blue for text)
        draw.rectangle([x0, y0, x1, y1], outline="blue", width=3)
        draw.text((x0 + 5, y0 - 15), label, fill="blue")

    df = df._append(extracted_data, ignore_index=True)

    # Save annotated image
    annotated_image_path = f"annotated_page_{page_number+1}.png"
    image.save(annotated_image_path)
    print(f"Annotated image saved: {annotated_image_path}")

df.to_csv(csv_output_path, index=False)

print(f"✅ Extraction complete! Data saved to {csv_output_path}")
