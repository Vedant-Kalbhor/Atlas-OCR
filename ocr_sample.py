import fitz  # PyMuPDF
import numpy as np
import cv2
import easyocr
import pandas as pd
import re

PDF_PATH = "./Sample.pdf"
DPI = 400

# -----------------------------
# Step 1: Load PDF and convert pages to images
# -----------------------------
doc = fitz.open(PDF_PATH)
images = []

for page in doc:
    pix = page.get_pixmap(dpi=DPI)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )

    # Convert to BGR for OpenCV / EasyOCR
    if pix.n == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:           # RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    images.append(img)

def ocr_with_rotations(reader, img, page_no):
    results_all = []

    rotations = [
        (img, 0),
        (cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 90),
        (cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), -90)
    ]

    for rotated_img, angle in rotations:
        ocr_results = reader.readtext(rotated_img)

        for bbox, text, confidence in ocr_results:
            if re.fullmatch(r"\d+(\.\d+)?", text):
                x_coords = [pt[0] for pt in bbox]
                y_coords = [pt[1] for pt in bbox]

                results_all.append({
                    "page": page_no,
                    "value": float(text),
                    "x": min(x_coords),
                    "y": min(y_coords),
                    "confidence": confidence,
                    "rotation": angle
                })

    return results_all

# -----------------------------
# Step 2: OCR
# -----------------------------
reader = easyocr.Reader(['en'], gpu=False)

ocr_results = []

for page_no, img in enumerate(images, start=1):
    ocr_results.extend(
        ocr_with_rotations(reader, img, page_no)
    )


# -----------------------------
# Step 3: Create DataFrame (THIS IS THE FIX)
# -----------------------------
df = pd.DataFrame(ocr_results)

# Remove duplicates based on proximity + value
df = df.sort_values(by="confidence", ascending=False)
df = df.drop_duplicates(subset=["page", "value", "x", "y"], keep="first")

df = df.sort_values(by=["page", "y", "x"])


# Save to Excel
df.to_excel("extracted_measurements.xlsx", index=False)

print("OCR completed successfully")
print(df)
