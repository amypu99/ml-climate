import fitz  # PyMuPDF for handling PDFs
from pdf2image import convert_from_path  # Convert PDF pages to images
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd

# Define both PDFs and their respective page numbers
pdf_configs = [
    {"pdf_path": "./CCRM2024.pdf", "pages": [73, 75, 77, 80, 83, 89, 91, 93, 95, 97, 103, 105, 107, 109, 111, 117, 119, 121, 123, 125], "year": 2024},
    {"pdf_path": "./CCRM2023.pdf", "pages": list(range(71, 118, 2)), "year": 2023},
]

csv_output_path = "extracted_CCRM_combined.csv"

# Define different color mappings for each year
color_mappings = {
    2024: {
        (196, 47, 39): "Very Poor",
        (241, 106, 33): "Poor",
        (250, 179, 22): "Moderate",
        (195, 216, 52): "Reasonable",
        (92, 185, 71): "High",
    },
    2023: {
        (194, 39, 21): "Very Poor",
        (235, 97, 10): "Poor",
        (250, 182, 0): "Moderate",
        (206, 214, 6): "Reasonable",
        (96, 178, 47): "High",
    },
}

# Function to classify colors based on the specific year
def classify_color(avg_color, year):
    avg_color = tuple(map(int, avg_color))
    for known_color, label in color_mappings[year].items():
        if np.linalg.norm(np.array(avg_color) - np.array(known_color)) < 15:
            return label
    return "Unknown"

# Define text and color-based bounding boxes for each year
bounding_boxes = {
    2024: {
        "color_boxes": [
            (1850, 150, 1870, 220, "Overall transparency score"),
            (2050, 150, 2070, 220, "Overall integrity score"),
            (470, 240, 680, 250, "1. Transparency and Integrity"),
            (1850, 240, 2030, 250, "2. Transparency"),
            (2050, 240, 2220, 250, "2. Integrity"),
            (900, 1040, 1000, 1050, "3. Transparency"),
            (1080, 1040, 1150, 1050, "3. Integrity"),
            (1850, 1040, 2030, 1050, "4. Transparency"),
            (2050, 1040, 2220, 1050, "4. Integrity"),
        ],
        "text_boxes": [
            (220, 100, 1100, 230, "Name"),
            (1300, 150, 1450, 220, "What is their revenue?"),
            (1470, 150, 1630, 220, "What are their emissions?"),
            (1650, 140, 1840, 220, "What is their pledge?"),
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
    },
    2023: {
        "color_boxes": [
            (1100, 410, 1270, 415, "Overall transparency score"),
            (1320, 410, 1500, 415, "Overall integrity score"),
            (1100, 460, 1500, 470, "1. Transparency and Integrity"),
            (1280, 750, 1290, 760, "2. Transparency"),
            (1320, 750, 1330, 760, "2. Integrity"),
            (1280, 1350, 1290, 1360, "3. Transparency"),
            (1320, 1350, 1330, 1360, "3. Integrity"),
            (1100, 1610, 1200, 1620, "4. Transparency"),
            (1350, 1610, 1400, 1620, "4. Integrity"),
        ],
        "text_boxes": [
            (120, 150, 1000, 270, "Name"),
            (350, 320, 600, 430, "What is their revenue?"),
            (600, 320, 830, 430, "What are their emissions?"),
            (830, 320, 1080, 430, "What is their pledge?"),
            (450, 500, 1100, 630, "Major emission sources"),
            (450, 800, 1200, 870, "Headline target or pledge"),
            (465, 870, 1100, 960, "Short and medium term targets - grey"),
            (565, 960, 1100, 1100, "Short and medium term targets - white"),
            (465, 1100, 1100, 1170, "Longer-term targets - grey"),
            (565, 1170, 1100, 1300, "Longer-term targets - white"),
            (450, 1380, 1100, 1490, "Emission Reduction Measures"),
            (450, 1490, 1100, 1600, "Renewable electricity (scope 2) verbiage"),
            (450, 1750, 1100, 1790, "Climate contributions today"),
            (450, 1790, 1100, 1860, "Misleading offsetting claims today"),
        ]
    },
}

# Create a list to store extracted data
data_list = []

# Process each PDF
for pdf_config in pdf_configs:
    pdf_path = pdf_config["pdf_path"]
    pages_to_process = pdf_config["pages"]
    year = pdf_config["year"]

    doc = fitz.open(pdf_path)
    text_boxes = bounding_boxes[year]["text_boxes"]
    color_boxes = bounding_boxes[year]["color_boxes"]

    for page_number in pages_to_process:
        print(f"Processing {pdf_path} - Page {page_number} ({year})")

        page = doc[page_number]

        # Convert PDF page to image
        images = convert_from_path(pdf_path, first_page=page_number+1, last_page=page_number+1)
        image = images[0]
        extracted_data = {"Year": year}

        # Extract colors
        for x0, y0, x1, y1, label in color_boxes:
            cropped_img = image.crop((x0, y0, x1, y1))
            img_array = np.array(cropped_img)
            avg_color = img_array.mean(axis=(0, 1))
            extracted_data[label] = classify_color(avg_color, year)

        # Extract text
        for x0, y0, x1, y1, label in text_boxes:
            x0_pdf, y0_pdf, x1_pdf, y1_pdf = [coord / (205 / 72) for coord in (x0, y0, x1, y1)]
            rect = fitz.Rect(x0_pdf, y0_pdf, x1_pdf, y1_pdf)
            words = page.get_text("words")
            selected_words = [w[4] for w in words if x0_pdf <= w[0] <= x1_pdf and y0_pdf <= w[1] <= y1_pdf]
            extracted_data[label] = " ".join(selected_words).strip()

        data_list.append(extracted_data)

# Convert list of extracted data into a DataFrame
df = pd.DataFrame(data_list)

# Save the combined CSV
df.to_csv(csv_output_path, index=False)
print(f"Extraction complete! Data saved to {csv_output_path}")
