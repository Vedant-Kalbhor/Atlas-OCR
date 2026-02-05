#OCR + LLAMA

from flask import Flask, render_template, request, jsonify, send_file
import os
import pandas as pd
import json
import tempfile
from datetime import datetime
from werkzeug.utils import secure_filename

import fitz  # PyMuPDF
from PIL import Image
from paddleocr import PaddleOCR
import ollama

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'bmp'}

ocr = PaddleOCR( lang='en',use_textline_orientation=True)

# ------------------ Utils ------------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_image(image_path):
    result = ocr.predict(image_path)
    texts = []
    for line in result:
        for word in line:
            texts.append(word[1][0])
    return "\n".join(texts)


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


def call_llama2(extracted_text):
    prompt = f"""
You are an expert mechanical drawing analyst.

From the following OCR-extracted text of an engineering drawing,
extract ALL dimensions and return ONLY valid JSON in this exact format:

{{
  "columns": ["id", "..."],
  "data": [
    {{"id": "1", "...": "..."}}
  ]
}}

Rules:
- Always include "id"
- Include units in column names (mm, inch, etc.)
- Use "N/A" for missing values
- NO explanation text
- ONLY JSON

OCR TEXT:
{extracted_text}
"""

    response = ollama.chat(
        model="llama2",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


# ------------------ Routes ------------------

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

    try:
        temp_dir = tempfile.gettempdir()
        filename = secure_filename(file.filename)
        filepath = os.path.join(temp_dir, filename)
        file.save(filepath)

        # -------- OCR --------
        if filename.lower().endswith('.pdf'):
            extracted_text = extract_text_from_pdf(filepath)
        else:
            extracted_text = extract_text_from_image(filepath)

        os.remove(filepath)

        if not extracted_text.strip():
            return jsonify({'error': 'No text detected from file'}), 400

        # -------- LLaMA 2 --------
        llama_response = call_llama2(extracted_text)

        # Clean markdown if exists
        if llama_response.startswith("```"):
            llama_response = llama_response.split("```")[1]
            llama_response = llama_response.replace("json", "").strip()

        data = json.loads(llama_response)

        # -------- Excel --------
        excel_name = f"dimensions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        excel_path = os.path.join(temp_dir, excel_name)

        df = pd.DataFrame(data["data"])
        df.to_excel(excel_path, index=False, engine="openpyxl")

        return jsonify({
            "preview": data,
            "excel_file": excel_name
        })

    except json.JSONDecodeError as e:
        return jsonify({
            "error": "Invalid JSON from LLaMA",
            "raw_response": llama_response
        }), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<filename>')
def download(filename):
    temp_dir = tempfile.gettempdir()
    path = os.path.join(temp_dir, secure_filename(filename))

    if not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404

    return send_file(
        path,
        as_attachment=True,
        download_name=filename,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000)
