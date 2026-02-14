from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import pandas as pd
import tempfile
from datetime import datetime
from werkzeug.utils import secure_filename
import re

import fitz  # PyMuPDF
import pytesseract
import cv2
import ollama

# ---------------- CONFIG ----------------

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'bmp'}

# ---------------- UTILS ----------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)[1]
    return img


def extract_text_from_image(img_path):
    img = preprocess_image(img_path)
    return pytesseract.image_to_string(img, config="--psm 6")


def extract_text_from_pdf(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)

    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img_path = pdf_path + "_page.png"
        pix.save(img_path)
        text += extract_text_from_image(img_path)
        os.remove(img_path)

    return text


def filter_dimension_lines(text):
    lines = text.splitlines()
    keep = []

    pattern = re.compile(
        r'(\d+(\.\d+)?\s*(mm|cm|in|inch)|Ø|R\d+|±)',
        re.IGNORECASE
    )

    for line in lines:
        if pattern.search(line):
            keep.append(line.strip())

    return "\n".join(keep)


def call_llama2(text):
    prompt = f"""
Extract all engineering dimensions from the text below.

STRICT RULES:
- Output ONLY valid JSON
- No markdown
- No explanations
- Missing → "N/A"

FORMAT:
{{
  "columns": ["id", "dimension", "value", "unit"],
  "data": [
    {{"id": "1", "dimension": "outer_diameter", "value": "50", "unit": "mm"}}
  ]
}}

TEXT:
{text}
"""

    res = ollama.chat(
        model="llama2:latest",
        messages=[{"role": "user", "content": prompt}]
    )

    return res["message"]["content"]


def extract_json(text):
    m = re.search(r'\{[\s\S]*\}', text)
    return m.group(0) if m else None

# ---------------- ROUTES ----------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    temp_dir = tempfile.gettempdir()
    filename = secure_filename(file.filename)
    filepath = os.path.join(temp_dir, filename)

    try:
        file.save(filepath)

        if filename.lower().endswith('.pdf'):
            raw_text = extract_text_from_pdf(filepath)
        else:
            raw_text = extract_text_from_image(filepath)

        os.remove(filepath)

        dim_text = filter_dimension_lines(raw_text)

        if not dim_text.strip():
            return jsonify({'error': 'No dimensions detected'}), 400

        llama_output = call_llama2(dim_text)
        json_text = extract_json(llama_output)

        if not json_text:
            return jsonify({'error': 'Invalid LLM output', 'raw': llama_output}), 500

        data = json.loads(json_text)

        excel_name = f"dimensions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        excel_path = os.path.join(temp_dir, excel_name)

        df = pd.DataFrame(data["data"])
        df.to_excel(excel_path, index=False)

        return jsonify({
            "preview": data,
            "excel_file": excel_name
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download(filename):
    temp_dir = tempfile.gettempdir()
    path = os.path.join(temp_dir, secure_filename(filename))

    if not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404

    return send_file(path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
