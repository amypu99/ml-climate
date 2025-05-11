import fitz  # PyMuPDF for handling PDFs
from pdf2image import convert_from_path  # Convert PDF pages to images
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd

# CCRM 2023 Configuration
pdf_path = "./CCRM2023.pdf"
pages_to_process = list(range(71, 72, 2))
csv_output_path = "test.csv"

# Color-based boxes (for rating classification)
color_boxes = [
    (1100, 410, 1270, 415, "Overall transparency score"),
    (1320, 410, 1500, 415, "Overall integrity score"),
    (1100, 460, 1500, 470, "1. Transparency and Integrity"),
    (1100, 750, 1200, 760, "2. Transparency"),
    (1350, 750, 1400, 760, "2. Integrity"),
    (1100, 1320, 1200, 1330, "3. Transparency"),
    (1350, 1320, 1400, 1330, "3. Integrity"),
    (1100, 1610, 1200, 1620, "4. Transparency"),
    (1350, 1610, 1400, 1620, "4. Integrity"),
]

# Text-based bounding boxes
text_boxes = [
    (120, 150, 1000, 270, "Name"),
    (350, 320, 580, 450, "What is their revenue?"),
    (580, 320, 800, 450, "What are their emissions?"),
    (800, 320, 1050, 450, "What is their pledge?"),
    # (1050, 400, 1270, 450, "What is their transparency score?"),
    # (1270, 400, 1500, 450, "What is their overall integrity score?"),
    (1100, 410, 1270, 415, "What is their transparency score?"),
    (1320, 410, 1500, 415, "What is their overall integrity score?"),
    (450, 500, 1100, 630, "Major emission sources"),
    (450, 800, 1200, 870, "Headline target or pledge"),
    (465, 870, 1100, 960, "Short and medium term targets - grey"),
    (565, 960, 1100, 1100, "Short and medium term targets - white"),
    (465, 1100, 1100, 1170, "Longer-term targets - grey"),
    (565, 1170, 1100, 1300, "Longer-term targets - white"),
    (450, 1380, 1100, 1490, "Emission Reduction Measures"),
    (450, 1490, 1100, 1600, "Renewable Electricity"),
    (450, 1750, 1100, 1790, "Climate contributions today"),
    (450, 1790, 1100, 1860, "Misleading offsetting claims today"),
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
